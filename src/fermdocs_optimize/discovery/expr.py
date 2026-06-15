"""Safe symbolic compiler for agent-proposed ODE structure.

The discovery agent does NOT write Python — it emits each rate law as a math
expression string (e.g. "mu_max*S/(ks+S)*Max(0, 1-P/P_max)**2"). This module
parses those strings with sympy under a locked-down namespace and lambdifies
them into a fast numpy RHS. There is no `exec`/`eval` of arbitrary code: only
names we expose (state variables, operating conditions, fixed physics constants,
and the agent's own declared parameters) and a small whitelist of math functions
are allowed. Anything else raises — so the agent can rewrite the *equations*,
not run code.

By default the state order matches the LABS mechanistic model
(y = [X, S, P, M, O2, V]), but `compile_spec` accepts an arbitrary `state`
tuple (plus its own conditions/constants/defaults) so the SAME safe compiler can
build a kinetic model over whatever variables a dataset actually measured — not
only the LABS species. The LABS defaults reproduce the original behavior exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import parse_expr

# LABS state variables, in integration order (the default).
STATE = ("X", "S", "P", "M", "O2", "V")
# Operating conditions the loop supplies per batch (D is derived = F/V).
CONDITIONS = ("D", "F", "S_f", "M_f", "v0")
# Fixed physics constants (reactor config — NOT the oracle's kinetic params).
CONSTANTS = ("K_O2", "q_O2_max", "O2_sat", "kLa", "C_FEED")
# LABS ODE defaults (so the agent can omit the boilerplate balances) and the
# states it must define. Pass empties for a general (batch) model where every
# state needs an explicit ODE and there is no reactor physics.
_LABS_ODE_DEFAULTS = {
    "V": "F",
    "O2": "kLa*(O2_sat-O2) - q_O2_max*(O2/(K_O2+O2))*X - D*O2",
}
_LABS_REQUIRED = ("X", "S", "P", "M")

# Whitelisted math functions available inside expressions.
_FUNCS = {
    "Max": sp.Max, "Min": sp.Min, "exp": sp.exp, "log": sp.log,
    "sqrt": sp.sqrt, "Abs": sp.Abs, "Pow": sp.Pow,
}
_ALLOWED_FUNC_NAMES = {"Max", "Min", "exp", "log", "sqrt", "Abs", "Pow"}
# parse_expr's transformed code references Integer/Float/Symbol/Rational. We give
# it ONLY those, and an empty __builtins__, so no builtin (e.g. __import__) is
# reachable from the eval. This is what makes parsing untrusted strings safe.
_SAFE_GLOBALS = {
    "__builtins__": {},
    "Integer": sp.Integer, "Float": sp.Float,
    "Symbol": sp.Symbol, "Rational": sp.Rational,
}
# numpy backings for lambdify (binary np.maximum/minimum compose for n-ary Max/Min).
_LAMBDIFY_MODULES = [{"Max": np.maximum, "Min": np.minimum, "Abs": np.abs}, "numpy"]


class ExprError(ValueError):
    """A proposed expression referenced an unknown name or failed to parse."""


@dataclass
class CompiledModel:
    """A compiled, parameter-named RHS ready for odeint/least_squares."""

    param_names: tuple[str, ...]
    _funcs: list  # one lambdified callable per state, in `state` order
    _arg_order: tuple[str, ...]
    state: tuple[str, ...] = STATE
    _cond_names: tuple[str, ...] = field(default_factory=lambda: tuple(
        c for c in CONDITIONS if c != "D"))
    _has_D: bool = True  # derive D = F/V (LABS reactor dilution)

    def rhs(self, y, t, theta, cond, const):
        """dy/dt for odeint. `theta` aligns to `param_names`; `cond`/`const` are
        dicts for the (non-derived) conditions and constants. Env is built from
        the model's own `state` order, so this works for any state vector."""
        env = {s: y[i] for i, s in enumerate(self.state)}
        if self._has_D:
            V = env.get("V", cond.get("v0", 1.0))
            env["D"] = cond["F"] / V if V > 1e-12 else 0.0
        for c in self._cond_names:
            env[c] = cond[c]
        env.update(const)
        for n, v in zip(self.param_names, theta):
            env[n] = float(v)
        args = [env[a] for a in self._arg_order]
        return [float(f(*args)) for f in self._funcs]


def compile_spec(
    params: list[str], aux: dict[str, str], odes: dict[str, str], *,
    state: tuple[str, ...] = STATE,
    conditions: tuple[str, ...] = CONDITIONS,
    constants: tuple[str, ...] = CONSTANTS,
    ode_defaults: dict[str, str] | None = None,
    required: tuple[str, ...] | None = None,
) -> CompiledModel:
    """Compile an agent-proposed model into a CompiledModel.

    `params`  : names the agent introduces and wants fitted.
    `aux`     : intermediate expressions (e.g. mu, growth term); may reference
                state, conditions, constants, params, and earlier aux names.
    `odes`    : derivative expression per state.

    By default this builds the LABS model (state X,S,P,M,O2,V; X,S,P,M required;
    O2/V default to the standard balances; D=F/V derived). Pass `state` (and,
    for a batch model, `conditions=()`, `constants=()`, `ode_defaults={}`) to
    discover a kinetic model over arbitrary measured variables; then every state
    needs an explicit ODE."""
    ode_defaults = _LABS_ODE_DEFAULTS if ode_defaults is None else ode_defaults
    required = _LABS_REQUIRED if required is None else required
    has_D = "D" in conditions
    allowed = set(state) | set(conditions) | set(constants) | set(params) | set(aux)
    symtab = {n: sp.Symbol(n) for n in allowed}

    def _parse(name: str, text: str):
        try:
            e = parse_expr(str(text), local_dict={**symtab, **_FUNCS},
                           global_dict=_SAFE_GLOBALS, evaluate=True)
            e = sp.sympify(e)  # coerce any bare python number to a sympy object
        except Exception as exc:  # noqa: BLE001
            raise ExprError(f"{name!r}: cannot parse {text!r}: {exc}") from exc
        unknown = {s.name for s in e.free_symbols} - allowed
        if unknown:
            raise ExprError(f"{name!r}: unknown name(s) {sorted(unknown)} in {text!r}")
        # Reject any function not on the whitelist (incl. undefined f(...) calls) —
        # this is what keeps the agent to math, not code.
        if e.atoms(AppliedUndef):
            bad = {type(f).__name__ for f in e.atoms(AppliedUndef)}
            raise ExprError(f"{name!r}: unknown function call(s) {sorted(bad)} in {text!r}")
        used = {type(f).__name__ for f in e.atoms(sp.Function)}
        bad = used - _ALLOWED_FUNC_NAMES
        if bad:
            raise ExprError(f"{name!r}: disallowed function(s) {sorted(bad)} in {text!r}")
        return e

    # Resolve aux in declared order, substituting earlier aux so odes become
    # pure expressions over state/conditions/constants/params.
    aux_expr: dict[str, sp.Expr] = {}
    for name, text in aux.items():
        e = _parse(name, text)
        e = e.subs({sp.Symbol(k): v for k, v in aux_expr.items()})
        aux_expr[name] = e

    # Sensible defaults so the agent can focus on the interesting balances.
    full_odes = dict(odes)
    for s, expr in ode_defaults.items():
        full_odes.setdefault(s, expr)

    missing = [s for s in required if s not in full_odes]
    if missing:
        raise ExprError(f"odes must define {list(required)} (missing {missing})")
    undefined = [s for s in state if s not in full_odes]
    if undefined:
        raise ExprError(f"every state needs an ODE (missing {undefined})")

    cond_names = tuple(c for c in conditions if c != "D")
    arg_order = tuple(state) + (("D",) if has_D else ()) + cond_names \
        + tuple(constants) + tuple(params)
    arg_syms = [sp.Symbol(a) for a in arg_order]

    funcs = []
    for s in state:
        e = _parse(f"d{s}/dt", full_odes[s])
        e = e.subs({sp.Symbol(k): v for k, v in aux_expr.items()})
        funcs.append(sp.lambdify(arg_syms, e, modules=_LAMBDIFY_MODULES))

    return CompiledModel(param_names=tuple(params), _funcs=funcs, _arg_order=arg_order,
                         state=tuple(state), _cond_names=cond_names, _has_D=has_D)
