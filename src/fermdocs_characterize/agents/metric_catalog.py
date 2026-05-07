"""Declarative menu of fermentation metrics, ported from the langgraph
reference repo (`src/tools/analysis_catalog.py`) into our naming.

Architectural role: the trajectory analyzer's *planner-side* surface.
The LLM reads the catalog to decide which metrics apply to the current
bundle, then writes `execute_python` code that imports the corresponding
verified toolkit function and prints the result. Math never runs in the
LLM hot path; the catalog is the bridge from "metric_id" -> import path.

Tiering convention:
  A — pure trajectory math, organism-agnostic (mu, doubling, phases, ...)
  B — mass-balance / yield math, requires measured inputs (RQ, yields, C-balance)
  C — literature-assisted estimates with citations (Verduyn yields, Van't Riet kLa)

PR 1 scope: all 60 entries declared so the catalog is a complete
schema, but `toolkit_fn` is only populated for the kinetics functions
shipped in this PR (A8, A9, A10, A11). Other entries carry
`toolkit_fn=None` with `status="pending"`; roundtrip test only checks
non-None entries.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Literal

Tier = Literal["A", "B", "C", "P"]
OutputShape = Literal[
    "scalar",
    "scalar_with_metadata",
    "timecourse_csv",
    "table_csv",
    "multi_run_summary",
    "per_run_scalar",
]
EntryStatus = Literal["ready", "pending"]


@dataclass(frozen=True)
class InputSpec:
    """A required trajectory variable for a catalog entry.

    `variable` matches the golden_schema variable name (e.g.
    'biomass_g_l', 'glucose_g_l'). `min_points` is the smallest
    number of non-imputed observations the toolkit needs to compute
    a meaningful result; analyzer should emit a data-gap finding
    rather than calling the toolkit with too few points.

    `accepted_proxies` lists alternative variable names that satisfy
    this input. Used when the math is scale-invariant (e.g. specific
    growth rate μ = d ln(X)/dt is identical whether X is biomass_g_l,
    wcw_g_l, or od600_au — only the ratio matters). Analyzer is
    expected to pick the strongest available proxy and record which
    one in Finding.statistics so downstream consumers can reason
    about the citation honestly.
    """

    variable: str
    min_points: int = 5
    description: str = ""
    accepted_proxies: tuple[str, ...] = ()


# Biomass-equivalent proxies for scale-invariant kinetics (μ, doubling
# time, phase segmentation, phasewise μ). All three measure the same
# underlying quantity (cell mass) at different scales and with different
# constants, so derivative-based metrics work identically. ABSOLUTE-
# magnitude metrics (B6 byproduct yield, B16 carbon balance) still need
# biomass_g_l specifically and don't list these proxies.
BIOMASS_PROXIES: tuple[str, ...] = ("wcw_g_l", "od600_au")

# Dissolved oxygen proxies. dissolved_o2_mg_l is the absolute concentration
# but industrial probes typically report do_pct_saturation; the
# 30%-saturation threshold for O2 limitation IS in % saturation, so for
# A14 (DO margin profile) the proxy is arguably preferable.
DO_PROXIES: tuple[str, ...] = ("do_pct_saturation",)


@dataclass(frozen=True)
class ParamSpec:
    """A configurable parameter the analyzer may pass to the toolkit.

    Defaults are sourced from literature (cited in `note`) and should
    rarely be overridden. When overridden, the analyzer must record
    `params_used` in Finding.statistics so the audit trail is complete.
    """

    name: str
    default: float | int | str | None
    note: str = ""


@dataclass(frozen=True)
class CatalogEntry:
    metric_id: str
    tier: Tier
    short_description: str
    long_description: str
    applies_to: str
    output_shape: OutputShape
    output_columns: tuple[str, ...] = ()
    required_inputs: tuple[InputSpec, ...] = ()
    required_parameters: tuple[ParamSpec, ...] = ()
    toolkit_fn: str | None = None
    status: EntryStatus = "pending"
    literature_source: str | None = None

    def is_ready(self) -> bool:
        return self.status == "ready" and self.toolkit_fn is not None

    def resolve_toolkit_fn(self):
        """Import + return the toolkit callable. Raises if not ready."""
        if not self.toolkit_fn:
            raise ValueError(f"{self.metric_id}: no toolkit_fn declared")
        module_path, _, fn_name = self.toolkit_fn.partition(":")
        if not module_path or not fn_name:
            raise ValueError(
                f"{self.metric_id}: malformed toolkit_fn '{self.toolkit_fn}' "
                f"(expected 'module.path:function_name')"
            )
        module = importlib.import_module(module_path)
        return getattr(module, fn_name)


# -----------------------------------------------------------------------------
# Catalog
#
# IDs are stable across runs: A8 always means specific growth rate, B10
# always means RQ. Adding/removing entries is a code change. Renumbering
# is forbidden — downstream Finding.statistics["metric_id"] joins assume
# the IDs are immutable.
# -----------------------------------------------------------------------------


_CATALOG: dict[str, CatalogEntry] = {}


def _register(entry: CatalogEntry) -> None:
    if entry.metric_id in _CATALOG:
        raise ValueError(f"duplicate metric_id: {entry.metric_id}")
    _CATALOG[entry.metric_id] = entry


# ---------------- Tier A — kinetics (organism-agnostic trajectory math) ----------------


_register(
    CatalogEntry(
        metric_id="A1",
        tier="A",
        short_description="Run-level KPI summary",
        long_description=(
            "Per-run summary table: max biomass, max product, final yield, "
            "duration, time-to-plateau. Assembled from raw trajectories with "
            "no model assumptions."
        ),
        applies_to="any bundle with at least one biomass or product trajectory",
        output_shape="table_csv",
        output_columns=("run_id", "kpi_name", "value", "unit"),
        required_inputs=(InputSpec(variable="biomass_g_l", min_points=3),),
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A2",
        tier="A",
        short_description="Trajectory completeness map",
        long_description=(
            "Per-(run, variable) coverage report: count of non-null points, "
            "min/max time, gaps over 4h. Surfaces measurement-frequency "
            "variation across batches before any modelling."
        ),
        applies_to="any bundle",
        output_shape="table_csv",
        output_columns=("run_id", "variable", "n_points", "first_h", "last_h", "max_gap_h"),
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A3",
        tier="A",
        short_description="Per-run final values vs cohort",
        long_description=(
            "Z-score of each run's final biomass/product against the cohort. "
            "Flags runs more than 2 sigma from the population."
        ),
        applies_to="cohorts with 5+ runs",
        output_shape="multi_run_summary",
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A4",
        tier="A",
        short_description="Time-to-target curves",
        long_description=(
            "For each run, time to reach 50/75/95% of that run's max biomass. "
            "Useful for cross-batch comparison of growth speed."
        ),
        applies_to="bundles with biomass trajectories",
        output_shape="table_csv",
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A5",
        tier="A",
        short_description="Maximum value timing",
        long_description="Time at which each run reaches its maximum for each tracked variable.",
        applies_to="any bundle with trajectories",
        output_shape="table_csv",
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A6",
        tier="A",
        short_description="Sampling density profile",
        long_description="Per-run inter-sample interval distribution; flags runs with sparse sampling that limits derivative-based metrics.",
        applies_to="any bundle",
        output_shape="multi_run_summary",
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A7",
        tier="A",
        short_description="Imputation density",
        long_description="Fraction of imputed-vs-measured points per run, per variable. Caps confidence on downstream derived metrics.",
        applies_to="any bundle",
        output_shape="table_csv",
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A8",
        tier="A",
        short_description="Specific growth rate mu(t)",
        long_description=(
            "Time-resolved specific growth rate from biomass trajectory: "
            "mu = d(ln X)/dt computed via Savitzky-Golay smoothing of ln(biomass). "
            "Returns the timecourse plus mu_max and time-of-max."
        ),
        applies_to="any run with >= 5 biomass measurements",
        output_shape="scalar_with_metadata",
        output_columns=("mu_max", "t_mu_max_h", "n_points"),
        required_inputs=(
            InputSpec(
                variable="biomass_g_l",
                min_points=5,
                description="Cell biomass concentration over time",
                accepted_proxies=BIOMASS_PROXIES,
            ),
        ),
        required_parameters=(
            ParamSpec(
                name="window",
                default=7,
                note="Savitzky-Golay window length (odd integer); 7 ≈ 4-7h smoothing on hourly data.",
            ),
            ParamSpec(
                name="poly",
                default=2,
                note="Savitzky-Golay polynomial order; 2 keeps curvature in growth phase.",
            ),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.kinetics:compute_mu",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="A9",
        tier="A",
        short_description="Doubling time t_d",
        long_description=(
            "Doubling time during exponential growth: t_d = ln(2) / mu_max. "
            "Reports both the value and the time window over which mu_max held."
        ),
        applies_to="any run with a computable mu_max from A8",
        output_shape="scalar_with_metadata",
        output_columns=("t_doubling_h", "mu_max", "phase_start_h", "phase_end_h"),
        required_inputs=(
            InputSpec(
                variable="biomass_g_l", min_points=5, accepted_proxies=BIOMASS_PROXIES
            ),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.kinetics:doubling_time",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="A10",
        tier="A",
        short_description="Growth phase segmentation",
        long_description=(
            "Partitions the run into lag / exponential / linear / stationary / "
            "decline phases by thresholding mu(t). Returns per-phase start, "
            "end, mean mu, and biomass delta."
        ),
        applies_to="any run with biomass trajectory >= 8 points",
        output_shape="table_csv",
        output_columns=("phase", "start_h", "end_h", "mean_mu", "biomass_delta_g_l"),
        required_inputs=(
            InputSpec(
                variable="biomass_g_l", min_points=8, accepted_proxies=BIOMASS_PROXIES
            ),
        ),
        required_parameters=(
            ParamSpec(
                name="lag_threshold",
                default=0.05,
                note="mu below this counts as lag phase. Units: 1/h.",
            ),
            ParamSpec(
                name="exp_threshold",
                default=0.15,
                note="mu above this counts as exponential. Units: 1/h.",
            ),
            ParamSpec(
                name="decline_threshold",
                default=-0.02,
                note="mu below this (negative) counts as decline.",
            ),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.kinetics:segment_growth_phases",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="A11",
        tier="A",
        short_description="Phasewise mu summary",
        long_description=(
            "Mean specific growth rate within each segmented phase from A10. "
            "Used to compare cohorts on phase-resolved kinetics rather than "
            "single-number mu_max."
        ),
        applies_to="any run that produces phases via A10",
        output_shape="table_csv",
        output_columns=("phase", "mean_mu", "n_points"),
        required_inputs=(
            InputSpec(
                variable="biomass_g_l", min_points=8, accepted_proxies=BIOMASS_PROXIES
            ),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.kinetics:phasewise_mu",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="A12",
        tier="A",
        short_description="Volumetric productivity Qp",
        long_description="Whole-run volumetric productivity: (P_final - P_init) / duration. Reports g/L/h.",
        applies_to="runs with product concentration trajectory",
        output_shape="scalar",
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A13",
        tier="A",
        short_description="Phasewise volumetric productivity",
        long_description="Volumetric productivity within each growth phase from A10.",
        applies_to="runs that produce A10 phases and have product trajectory",
        output_shape="table_csv",
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A14",
        tier="A",
        short_description="DO margin profile",
        long_description=(
            "Profiles dissolved oxygen against an O2-limitation threshold. "
            "Returns frac_below, min_margin, time_below_h. Margin = DO - threshold."
        ),
        applies_to="runs with dissolved_o2 trajectory (mg/L or % saturation)",
        output_shape="scalar_with_metadata",
        output_columns=("frac_below", "min_margin", "min_do", "mean_do", "time_below_h"),
        required_inputs=(
            InputSpec(
                variable="dissolved_o2_mg_l",
                min_points=2,
                accepted_proxies=DO_PROXIES,
                description=(
                    "Either dissolved_o2_mg_l (absolute concentration) or "
                    "do_pct_saturation (% air saturation, more common in "
                    "industrial reports). The 30%-saturation default "
                    "threshold IS in % units, so do_pct_saturation is "
                    "actually the more appropriate input when both are "
                    "available."
                ),
            ),
        ),
        required_parameters=(
            ParamSpec(
                name="critical_threshold_pct",
                default=30.0,
                note="Lower DO bound for aerobic fermentation (textbook). Override per organism if documented.",
            ),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.operational:compute_do_margin",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="A15",
        tier="A",
        short_description="Controller excursion count",
        long_description=(
            "Count, duration, and peak deviation of windows where a measured "
            "controlled variable left the setpoint ±tolerance band."
        ),
        applies_to="runs with paired controlled variable + setpoint trajectories",
        output_shape="multi_run_summary",
        output_columns=("n_excursions", "total_time_out_h", "max_abs_deviation"),
        required_parameters=(
            ParamSpec(
                name="tolerance",
                default=None,
                note="±tolerance in measurement units; required, no universal default.",
            ),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.operational:controller_excursions",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="A16",
        tier="A",
        short_description="Foam-related agitation drops",
        long_description="Detect agitation_rpm dips correlated with foam events when foam pressure measurement available.",
        applies_to="runs with agitation + foam telemetry",
        output_shape="table_csv",
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A17",
        tier="A",
        short_description="Impeller tip speed",
        long_description="Tip speed = π · D · N (rps). Returns scalar (constant RPM) or timecourse.",
        applies_to="bioreactor runs with agitation_rpm and impeller_diameter_m",
        output_shape="timecourse_csv",
        output_columns=("tip_speed_m_s",),
        required_inputs=(InputSpec(variable="agitation_rpm", min_points=1),),
        required_parameters=(
            ParamSpec(name="impeller_diameter_m", default=None, note="Required from dossier or process_priors."),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.operational:tip_speed",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="A18",
        tier="A",
        short_description="Power per volume P/V",
        long_description=(
            "Ungassed volumetric power: P = Np · ρ · N³ · D⁵, divided by working volume. "
            "Default Np=5.0 for Rushton turbine in turbulent regime (Bates 1963)."
        ),
        applies_to="bioreactor runs with agitation_rpm + reactor geometry",
        output_shape="timecourse_csv",
        output_columns=("p_per_v_w_m3",),
        required_inputs=(InputSpec(variable="agitation_rpm", min_points=1),),
        required_parameters=(
            ParamSpec(name="impeller_diameter_m", default=None, note="From dossier or process_priors."),
            ParamSpec(name="fluid_density_kg_m3", default=1000.0, note="Water density default; override for high-osmolarity broth."),
            ParamSpec(name="working_volume_l", default=None, note="From dossier."),
            ParamSpec(name="power_number", default=5.0, note="Rushton turbine in turbulent regime (Bates 1963)."),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.operational:power_per_volume",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="A19",
        tier="A",
        short_description="Cross-run KPI table",
        long_description="Tidies per-run KPI dicts into a DataFrame indexed by run_id. Backbone for A20/A21.",
        applies_to="cohorts of 2+ runs",
        output_shape="multi_run_summary",
        output_columns=("run_id", "kpi_name", "value"),
        toolkit_fn="fermdocs_characterize.toolkit.cross_run:cross_run_kpi_table",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="A20",
        tier="A",
        short_description="Pairwise deviation between runs",
        long_description=(
            "Top-K most divergent run pairs per KPI by relative gap "
            "|xi - xj| / mean(xi, xj). Surfaces outlier runs without "
            "a parametric assumption."
        ),
        applies_to="cohorts of 3+ runs with a KPI table from A19",
        output_shape="multi_run_summary",
        output_columns=("kpi", "run_a", "run_b", "value_a", "value_b", "relative_gap"),
        required_parameters=(
            ParamSpec(name="top_k", default=3, note="How many divergent pairs to surface per KPI."),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.cross_run:pairwise_deviation",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="A21",
        tier="A",
        short_description="Variance decomposition (between vs within group)",
        long_description=(
            "Decomposes per-KPI variance into between-group and within-group "
            "components when a grouping column (e.g. process condition) is "
            "given; otherwise reports total variance only."
        ),
        applies_to="cohorts of 5+ runs",
        output_shape="multi_run_summary",
        output_columns=("kpi", "total_var", "between_var", "within_var", "between_frac"),
        toolkit_fn="fermdocs_characterize.toolkit.cross_run:variance_decomposition",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="A22",
        tier="A",
        short_description="Trajectory shape clustering",
        long_description="Hierarchical clustering of biomass trajectory shapes (Euclidean on resampled curves). Surfaces qualitatively different batch behaviours.",
        applies_to="cohorts of 5+ runs with biomass",
        output_shape="multi_run_summary",
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A23",
        tier="A",
        short_description="Productivity reduction signature",
        long_description="Detects runs where productivity drops sharply mid-run (slope inversion in product timecourse).",
        applies_to="runs with product trajectory >= 10 points",
        output_shape="scalar_with_metadata",
        status="pending",
    )
)

_register(
    CatalogEntry(
        metric_id="A24",
        tier="A",
        short_description="Data quality flags",
        long_description="Per-run/per-variable flags for sensor saturation, frozen-value windows, and physical impossibility (e.g. biomass decreasing during stated growth phase). Caps downstream confidence.",
        applies_to="any bundle",
        output_shape="table_csv",
        status="pending",
    )
)


# ---------------- Tier B — mass balance / yields (require measured inputs) ----------------

_B_READY_IN_PR2 = {"B6", "B10", "B16"}

for _id, _short in [
    ("B1", "Substrate consumption rate qs"),
    ("B2", "Product formation rate qp"),
    ("B3", "Oxygen uptake rate OUR"),
    ("B4", "Carbon dioxide evolution rate CER"),
    ("B5", "Biomass yield Yxs"),
    ("B7", "Product yield Yps"),
    ("B8", "Oxygen yield Yxo"),
    ("B9", "Maintenance coefficient ms"),
    ("B11", "kLa back-calculation from OUR"),
    ("B12", "Substrate-to-product carbon split"),
    ("B13", "Ash-corrected dry-cell-weight balance"),
    ("B14", "Volumetric oxygen transfer OTR"),
    ("B15", "Volumetric mass transfer coefficient kLa"),
    ("B17", "Nitrogen balance closure"),
    ("B18", "Degree-of-reduction (gamma) balance"),
    ("B19", "Specific heat generation"),
    ("B20", "Energy balance closure"),
]:
    _register(
        CatalogEntry(
            metric_id=_id,
            tier="B",
            short_description=_short,
            long_description=f"{_short}. Implementation pending; not in PR 2 scope.",
            applies_to="depends on metric; inputs declared when wired",
            output_shape="scalar_with_metadata",
            status="pending",
        )
    )

_register(
    CatalogEntry(
        metric_id="B6",
        tier="B",
        short_description="Byproduct yield ΔP/ΔX",
        long_description=(
            "Ratio of byproduct mass produced to biomass produced over the "
            "run (g byproduct / g biomass). High ethanol/acetate yield is a "
            "classic overflow-metabolism signal."
        ),
        applies_to="runs with paired biomass + byproduct (ethanol/acetate/lactate) trajectories",
        output_shape="scalar_with_metadata",
        output_columns=("byproduct", "delta_p", "delta_x", "yield_g_per_g"),
        required_inputs=(
            # No accepted_proxies — yield needs absolute g/L.
            InputSpec(variable="biomass_g_l", min_points=2),
        ),
        required_parameters=(
            ParamSpec(name="byproduct_name", default="byproduct", note="Free-form label of the byproduct species."),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.balances:compute_byproduct_yield",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="B10",
        tier="B",
        short_description="Respiratory quotient RQ + overflow flag",
        long_description=(
            "RQ(t) = CER / OUR. Sustained RQ > 1.1 indicates respiro-fermentative "
            "metabolism (ethanol/acetate excretion). Reports mean, max, and the "
            "fraction of time above 1.1."
        ),
        applies_to="runs with paired OUR + CER trajectories",
        output_shape="scalar_with_metadata",
        output_columns=(
            "rq_mean",
            "rq_max",
            "rq_min",
            "frac_over_overflow_threshold",
            "overflow_flag",
        ),
        required_inputs=(
            InputSpec(variable="our_mmol_per_l_per_h", min_points=3),
            InputSpec(variable="cer_mmol_per_l_per_h", min_points=3),
        ),
        required_parameters=(
            ParamSpec(name="overflow_threshold", default=1.1, note="RQ above this counts as overflow (Crabtree literature)."),
            ParamSpec(name="overflow_fraction_floor", default=0.2, note="Fraction-of-run above threshold needed to set overflow_flag."),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.balances:compute_rq",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="B16",
        tier="B",
        short_description="Carbon balance closure",
        long_description=(
            "Run-level carbon balance: (C in biomass + C in products + C in CO2) "
            "/ C consumed. Healthy fermentation closes 0.90-1.05; below 0.85 or "
            "above 1.10 signals an unmeasured carbon pool."
        ),
        applies_to="runs with measured substrate consumption + biomass + at least one product/CO2 stream",
        output_shape="scalar_with_metadata",
        output_columns=(
            "closure",
            "c_consumed_g",
            "c_in_biomass_g",
            "c_in_products_g",
            "c_in_co2_g",
        ),
        required_inputs=(
            # NOTE: no accepted_proxies — carbon balance is absolute math.
            InputSpec(variable="biomass_g_l", min_points=2),
            InputSpec(variable="substrate_g_l", min_points=2),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.balances:compute_carbon_balance_closure",
        status="ready",
    )
)
del _B_READY_IN_PR2


# ---------------- Tier C — literature-assisted (require organism priors) ----------------

_C_READY_IN_PR3 = {"C2", "C3", "C4", "C5", "C9", "C10"}

for _id, _short in [
    ("C1", "Theoretical Yxs from Verduyn yields"),
    ("C6", "Theoretical qp from Yps_max"),
    ("C7", "Maintenance vs Verduyn ms reference"),
    ("C8", "Glucose uptake reference window"),
    ("C11", "Heat generation vs metabolic prediction"),
    ("C12", "CO2 stripping efficiency"),
    ("C13", "Dissolved-CO2 inhibition flag"),
    ("C14", "Reference doubling time vs observed"),
    ("C15", "Substrate inhibition window"),
    ("C16", "Product inhibition window"),
]:
    _register(
        CatalogEntry(
            metric_id=_id,
            tier="C",
            short_description=_short,
            long_description=(
                f"{_short}. Implementation pending; not in PR 3 scope. "
                "Will consume fermdocs.domain.process_priors when wired."
            ),
            applies_to="organism present in process_priors registry",
            output_shape="scalar_with_metadata",
            status="pending",
        )
    )

_register(
    CatalogEntry(
        metric_id="C2",
        tier="C",
        short_description="Reference mu_max vs observed",
        long_description=(
            "Compares observed mu_max from A8 against the organism's prior "
            "range (variable=mu_x_max_per_h). Flags 'observed mu_max=0.05 "
            "is 4x below Verduyn 1991 typical 0.20' style claims."
        ),
        applies_to="bundles with organism in priors registry + an A8 result",
        output_shape="scalar_with_metadata",
        output_columns=("ratio_observed_to_typical", "in_range", "typical"),
        toolkit_fn="fermdocs_characterize.toolkit.literature:mu_max_reference_vs_observed",
        status="ready",
        literature_source="Verduyn 1991; Hensing 1995; per organism in priors",
    )
)

_register(
    CatalogEntry(
        metric_id="C3",
        tier="C",
        short_description="Henry's-law-derived O2 saturation C*",
        long_description=(
            "C* = H(T,P) · p_O2 with optional Setschenow salt correction. "
            "Pure chemistry (no organism prior); audit trail cites Schumpe "
            "1982 + Sander 2015. Used as input to C9."
        ),
        applies_to="any bundle with temperature + pressure recorded",
        output_shape="scalar_with_metadata",
        output_columns=("c_star_mg_per_l", "henry_constant_mol_per_l_per_atm"),
        required_inputs=(
            InputSpec(
                variable="temperature_c",
                min_points=1,
                accepted_proxies=("temperature_k",),
                description="temperature in Celsius (or Kelvin — auto-converted).",
            ),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.literature:saturation_o2_concentration",
        status="ready",
        literature_source="Schumpe 1982; Sander 2015 review",
    )
)

_register(
    CatalogEntry(
        metric_id="C4",
        tier="C",
        short_description="Van't Riet kLa estimate",
        long_description=(
            "kLa = α · (P/V)^β · v_s^γ. Defaults α=0.026, β=0.4, γ=0.5 are "
            "for coalescing aqueous broths (Van't Riet 1979 Table II). "
            "Combine with A18 P/V for a full estimate."
        ),
        applies_to="bundles with computable P/V (A18) + reactor geometry",
        output_shape="scalar_with_metadata",
        output_columns=("kla_per_s", "kla_per_h"),
        toolkit_fn="fermdocs_characterize.toolkit.literature:vant_riet_kla",
        status="ready",
        literature_source="Van't Riet 1979 Table II",
    )
)

_register(
    CatalogEntry(
        metric_id="C5",
        tier="C",
        short_description="qs from Verduyn yields",
        long_description=(
            "Compares observed specific glucose uptake rate qs against the "
            "organism's prior range. Returns ratio observed/typical and "
            "in-range flag. When observed_qs is None, emits the prior "
            "typical/range as a reference."
        ),
        applies_to="bundles with biomass + substrate trajectories AND organism in priors",
        output_shape="scalar_with_metadata",
        output_columns=("ratio_observed_to_typical", "typical", "in_range"),
        toolkit_fn="fermdocs_characterize.toolkit.literature:qs_from_verduyn_yields",
        status="ready",
        literature_source="Sonnleitner 1986; Verduyn 1991; per organism",
    )
)

_register(
    CatalogEntry(
        metric_id="C9",
        tier="C",
        short_description="Oxygen demand vs supply ratio",
        long_description=(
            "ratio = OUR / [kLa · (C* - DO)]. ratio < 1 = O2 headroom; "
            "≈1 = O2-transfer limited; >1 = numerically impossible at "
            "steady state (flags OUR overestimate or kLa underestimate). "
            "Combine with B10 RQ for full overflow story."
        ),
        applies_to="bundles with OUR (B3), kLa (B15 or C4), DO trajectory, C* (C3)",
        output_shape="scalar_with_metadata",
        output_columns=("ratio_demand_over_supply", "otr_max_mg_per_l_per_h"),
        required_inputs=(
            InputSpec(variable="our_mmol_per_l_per_h", min_points=3),
            InputSpec(
                variable="dissolved_o2_mg_l", min_points=3, accepted_proxies=DO_PROXIES
            ),
        ),
        toolkit_fn="fermdocs_characterize.toolkit.literature:oxygen_demand_vs_supply",
        status="ready",
        literature_source="Mass-transfer envelope (textbook)",
    )
)

_register(
    CatalogEntry(
        metric_id="C10",
        tier="C",
        short_description="Overflow threshold (Crabtree / acetate switch)",
        long_description=(
            "Flags overflow when observed qs exceeds the top of the "
            "organism's qs prior range. Surfaces both the critical qs and "
            "the byproduct marker variable (e.g. ethanol_g_l for yeast) "
            "so the analyzer knows which species to inspect."
        ),
        applies_to="bundles with biomass + substrate trajectories AND organism in priors",
        output_shape="scalar_with_metadata",
        output_columns=("critical_qs", "overflow_signal", "marker_typical"),
        toolkit_fn="fermdocs_characterize.toolkit.literature:overflow_threshold",
        status="ready",
        literature_source="Sonnleitner 1986; per organism's qs prior",
    )
)
del _C_READY_IN_PR3


# ---------------- Tier P — product KPIs (process-family routed) ----------------
#
# Plan ref: plans/2026-05-07-characterize-determinism.md commit 2.
#
# These entries don't declare `required_inputs` against a specific
# variable name — the variable is process-family-dependent (penicillin
# fed-batch → penicillin_g_l, melanin batch → melanin_g_l, etc.).
# Adapters resolve the variable through `lookup_family()` at runtime
# and emit a [CONFIG_MISMATCH] data_gap when the routed variable is
# not in the bundle (A3 fix).

_register(
    CatalogEntry(
        metric_id="P1",
        tier="P",
        short_description="Final product titer",
        long_description=(
            "Last observed value on the process-family product trajectory."
            " The number that answers 'how much did this run produce?' in"
            " g/L. Process-family routed: penicillin_g_l, melanin_g_l,"
            " recombinant_protein_g_l, etc. (see process_families.yaml)."
        ),
        applies_to="runs with the family's product trajectory present",
        output_shape="scalar_with_metadata",
        output_columns=("final_titer_g_l", "t_final_h"),
        toolkit_fn="fermdocs_characterize.toolkit.products:compute_final_titer",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="P2",
        tier="P",
        short_description="Peak product titer",
        long_description=(
            "Maximum observed product titer + time of peak. Together with"
            " P3 (decline) flags lysis / hydrolysis: a high peak that"
            " doesn't hold means the product was made then degraded."
        ),
        applies_to="runs with the family's product trajectory present",
        output_shape="scalar_with_metadata",
        output_columns=("peak_titer_g_l", "t_peak_h"),
        toolkit_fn="fermdocs_characterize.toolkit.products:compute_peak_titer",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="P3",
        tier="P",
        short_description="Titer decline after peak",
        long_description=(
            "Fractional drop from peak product titer to final product"
            " titer: (peak - final) / peak. Range [0, 1]. Above 0.05"
            " flags 'something happened post-peak' — lysis releasing"
            " β-lactamase, product hydrolysis, or product re-uptake."
            " The IndPenSim RUN-1 signature: P peaked at 21.6 g/L @ 168h"
            " then declined to 14.3 g/L by 228h (decline_fraction=0.34)."
        ),
        applies_to="runs with the family's product trajectory present",
        output_shape="scalar_with_metadata",
        output_columns=(
            "decline_fraction", "peak_titer_g_l", "final_titer_g_l",
            "t_peak_h", "t_final_h", "is_declining",
        ),
        toolkit_fn="fermdocs_characterize.toolkit.products:compute_titer_decline",
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="P4",
        tier="P",
        short_description="Integral productivity ∫P/t dt",
        long_description=(
            "Trapezoidal integral of product titer over run duration."
            " Useful for comparing runs of different durations: a run"
            " that hits a high peak briefly may have lower integral than"
            " one that holds a moderate level for longer."
        ),
        applies_to="runs with the family's product trajectory present",
        output_shape="scalar_with_metadata",
        output_columns=(
            "integral_g_l_h", "mean_productivity_g_l_per_h", "duration_h",
        ),
        toolkit_fn=(
            "fermdocs_characterize.toolkit.products:compute_integral_productivity"
        ),
        status="ready",
    )
)

_register(
    CatalogEntry(
        metric_id="P5",
        tier="P",
        short_description="Precursor utilization fraction",
        long_description=(
            "For process families with a named precursor (penicillin's"
            " PAA, melanin's tyrosine): (peak - final) / peak. Polarity"
            " OPPOSITE of byproduct yield — high means precursor was"
            " consumed efficiently into product, low means it accumulated"
            " as waste. The IndPenSim RUN-2 PAA: peak → 634 mg/L"
            " (efficient); RUN-1: stayed at 5203 mg/L (wasted)."
        ),
        applies_to="runs whose process family declares precursor_variables",
        output_shape="scalar_with_metadata",
        output_columns=(
            "utilization_fraction", "peak_value", "final_value",
            "utilization_class", "precursor_variable",
        ),
        toolkit_fn=(
            "fermdocs_characterize.toolkit.products:compute_precursor_utilization"
        ),
        status="ready",
    )
)


# -----------------------------------------------------------------------------
# Public surface
# -----------------------------------------------------------------------------


CATALOG: dict[str, CatalogEntry] = dict(_CATALOG)


def get_entry(metric_id: str) -> CatalogEntry:
    if metric_id not in CATALOG:
        raise KeyError(f"unknown metric_id: {metric_id!r}")
    return CATALOG[metric_id]


def entries_by_tier(tier: Tier) -> list[CatalogEntry]:
    return [e for e in CATALOG.values() if e.tier == tier]


def ready_entries() -> list[CatalogEntry]:
    return [e for e in CATALOG.values() if e.is_ready()]


# Specialist routing — used in PR 3 by the hypothesis projector. Declared
# here so the metric_id -> domain mapping lives next to the catalog itself.

KINETICS_METRICS: frozenset[str] = frozenset(
    {"A8", "A9", "A10", "A11", "A13", "A23", "B10"}
)
MASS_TRANSFER_METRICS: frozenset[str] = frozenset(
    {"A14", "A15", "A17", "A18", "B8", "B11", "B14", "B15", "C3", "C4", "C9", "C13"}
)
METABOLIC_METRICS: frozenset[str] = frozenset(
    {"B6", "B10", "B16", "B17", "B18", "C5", "C6", "C7", "C10"}
)
DATA_QUALITY_METRICS: frozenset[str] = frozenset({"A24"})
