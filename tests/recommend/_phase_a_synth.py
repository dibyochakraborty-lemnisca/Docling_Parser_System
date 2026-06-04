"""Phase A: prove the confident path on clean synthetic Monod batch data.

Simulate a TRUE Monod batch (known mu_max/Ks) with brewtwin, two runs with
slightly different initial conditions + light noise, dense sampling. Write it as
a bundle observations.csv, then run the real fit kit + rubric. Expect: a
confident MECHANISTIC recommendation with recovered params near truth and
held-out R^2 > 0.75.
"""
import json, tempfile, os
import numpy as np
import jax; jax.config.update("jax_enable_x64", True)
from brewtwin.species import ChemicalSpecies, BiologicalSpecies
from brewtwin.reactions.reaction import Reaction
from brewtwin.reactions.network import ReactionNetwork
from brewtwin.rate_models.kinetic import Monod, Concentration, Constant
from brewtwin.rate_models.composite import CompositeRateLaw
from brewtwin.reactors.batch import BatchReactor
from brewtwin.solvers.jax_solver import JaxSolver

TRUE_MU, TRUE_KS, Y = 0.40, 0.50, 0.5
rng = np.random.default_rng(7)

def simulate(X0, S0):
    X = BiologicalSpecies("biomass_g_l", conc=X0)
    S = ChemicalSpecies("substrate_g_l", conc=S0)
    net = ReactionNetwork("truth"); net.add_species(X); net.add_species(S)
    rate = CompositeRateLaw(Constant(TRUE_MU), Monod(S, Ks=TRUE_KS), Concentration(X))
    net.add_reaction(Reaction(name="growth", stoichiometry={"substrate_g_l": -1.0/Y, "biomass_g_l": 1.0}, rate_model=rate))
    reactor = BatchReactor(net)
    t = np.linspace(0.0, 24.0, 25)
    sim = JaxSolver("kvaerno5", rtol=1e-8, atol=1e-10, max_steps=200000).solve(reactor, t_span=(0.0, 24.0), t_eval=t)
    xb = np.asarray(sim.y[:, sim.variables.index("biomass_g_l")])
    sb = np.asarray(sim.y[:, sim.variables.index("substrate_g_l")])
    return t, xb, sb

rows = ["run_id,variable,time_h,value,imputed,unit"]
for run, (X0, S0) in {"RUN-0001": (0.10, 10.0), "RUN-0002": (0.12, 9.5)}.items():
    t, xb, sb = simulate(X0, S0)
    xb = xb * (1 + 0.03 * rng.standard_normal(xb.shape))
    sb = np.clip(sb * (1 + 0.03 * rng.standard_normal(sb.shape)), 0, None)
    for i, ti in enumerate(t):
        rows.append(f"{run},biomass_g_l,{ti:.3f},{xb[i]:.5f},0,g/L")
        rows.append(f"{run},substrate_g_l,{ti:.3f},{sb[i]:.5f},0,g/L")

d = tempfile.mkdtemp()
csv = os.path.join(d, "observations.csv")
open(csv, "w").write("\n".join(rows))

from fermdocs_recommend.fit_kit import run_bakeoff
from fermdocs_recommend import rubric

res = run_bakeoff(csv, biomass="biomass_g_l", substrate="substrate_g_l",
                  feed_var=None, volume_var=None, n_adam=250, n_epochs=300)
cands = list(res.values())
print("TRUTH: mu_max=%.2f Ks=%.2f" % (TRUE_MU, TRUE_KS))
for c in cands:
    rep = c.get("report") or {}
    fq = rep.get("fit_quality", {})
    params = {k: round(v["value"], 3) for k, v in (rep.get("fitted_parameters") or {}).items()}
    print(" ", c["model_type"], "att=%s disq=%s" % (c["attempted"], c["disqualified"]),
          "R2=", {k: round(v.get("r2", float("nan")), 3) for k, v in fq.items()},
          "params=", params, (c.get("disqualification_reason") or "")[:50])

v = rubric.select(cands)
print("VERDICT:", v["recommended_model"], "| confident:", v["confident"], "| reason:", v["refusal_reason"])
print("rationale:", v["selection_rationale"])
