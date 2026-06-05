"""Safe symbolic compiler for agent-proposed ODE structure.

The discovery agent does NOT write Python — it emits each rate law as a math
expression string (e.g. "mu_max*S/(ks+S)*Max(0, 1-P/P_max)**2"). This module
parses those strings with sympy under a locked-down namespace and lambdifies
them into a fast numpy RHS. There is no `exec`/`eval` of arbitrary code: only
names we expose (state variables, operating conditions, fixed physics constants,
and the agent's own declared parameters) and a small whitelist of math functions
are allowed. Anything else raises — so the agent can rewrite the *equations*,
not run code.

State order is fixed to match the mechanistic model: y = [X, S, P, M, O2, V].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import parse_expr

# State variables, in integration order.
STATE = ("X", "S", "P", "M", "O2", "V")
# Operating conditions the loop supplies per batch (D is derived = F/V).
CONDITIONS = ("D", "F", "S_f", "M_f", "v0")
# Fixed physics constants (reactor config — NOT the oracle's kinetic params).
CONSTANTS = ("K_O2", "q_O2_max", "O2_sat", "kLa", "C_FEED")

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
    _funcs: list  # one lambdified callable per state, in STATE order
    _arg_order: tuple[str, ...]

    def rhs(self, y, t, theta, cond, const):
        """dy/dt for odeint. `theta` aligns to `param_names`; `cond`/`const` are
        dicts for CONDITIONS (minus D, derived here) and CONSTANTS."""
        X, S, P, M, O2, V = y
        D = cond["F"] / V if V > 1e-12 else 0.0
        env = {
            "X": X, "S": S, "P": P, "M": M, "O2": O2, "V": V,
            "D": D, "F": cond["F"], "S_f": cond["S_f"], "M_f": cond["M_f"],
            "v0": cond["v0"], **const,
            **{n: float(v) for n, v in zip(self.param_names, theta)},
        }
        args = [env[a] for a in self._arg_order]
        return [float(f(*args)) for f in self._funcs]


def compile_spec(params: list[str], aux: dict[str, str], odes: dict[str, str]) -> CompiledModel:
    """Compile an agent-proposed model into a CompiledModel.

    `params`  : names the agent introduces and wants fitted.
    `aux`     : intermediate expressions (e.g. mu, growth term); may reference
                state, conditions, constants, params, and earlier aux names.
    `odes`    : derivative expression per state. X,S,P,M are required; O2 and V
                default to the standard aeration / volume balance if omitted.
    """
    allowed = set(STATE) | set(CONDITIONS) | set(CONSTANTS) | set(params) | set(aux)
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
    full_odes.setdefault("V", "F")
    full_odes.setdefault("O2", "kLa*(O2_sat-O2) - q_O2_max*(O2/(K_O2+O2))*X - D*O2")

    missing = [s for s in ("X", "S", "P", "M") if s not in full_odes]
    if missing:
        raise ExprError(f"odes must define X,S,P,M (missing {missing})")

    arg_order = tuple(STATE) + ("D",) + tuple(c for c in CONDITIONS if c != "D") \
        + tuple(CONSTANTS) + tuple(params)
    arg_syms = [sp.Symbol(a) for a in arg_order]

    funcs = []
    for s in STATE:
        e = _parse(f"d{s}/dt", full_odes[s])
        e = e.subs({sp.Symbol(k): v for k, v in aux_expr.items()})
        funcs.append(sp.lambdify(arg_syms, e, modules=_LAMBDIFY_MODULES))

    return CompiledModel(param_names=tuple(params), _funcs=funcs, _arg_order=arg_order)
