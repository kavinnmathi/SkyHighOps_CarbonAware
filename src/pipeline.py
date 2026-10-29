"""End-to-end experiment: load data -> train forecaster -> simulate -> evaluate -> save."""
from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .carbon_import import REAL_ORIGINS, read_provenance
from .config import resolve_path
from .data_sources import (STEPS_PER_DAY, generate_synthetic_carbon, generate_synthetic_workload,
                           load_carbon_csv, load_google_trace_extract, load_workload_csv,
                           workload_from_dataframe)
from .errors import DataError
from .forecasting import MAX_LAG, CarbonForecaster, CarbonView, evaluate_forecaster, rolling_origin_evaluation
from .models import NodeSpec, SiteSpec
from .schedulers import BaselineScheduler, CarbonAwareScheduler
from .simulator import Simulator

log = logging.getLogger("skyhighops")


@dataclass
class Prepared:
    ci_df: pd.DataFrame
    base_ci: np.ndarray
    sim_start: int
    sim_steps: int
    total_steps: int
    workload: pd.DataFrame
    carbon_origin: str
    workload_origin: str
    forecaster: CarbonForecaster
    forecast_metrics: pd.DataFrame
    forecast_rolling: pd.DataFrame | None = None


def _ensure_sample_files(cfg: dict) -> None:
    """Re-create the bundled synthetic sample files if the user deleted them."""
    d = cfg["data"]
    cpath = resolve_path(d["sample_carbon_csv"])
    if not cpath.exists():
        cpath.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic_carbon().to_csv(cpath, index=False)
        log.info("Re-generated %s", cpath)
    wpath = resolve_path(d["sample_workload_csv"])
    if not wpath.exists():
        sim_steps = d["simulation_days"] * STEPS_PER_DAY
        generate_synthetic_workload(sim_steps, cfg["synthetic_workload"], cfg["random_seed"]).to_csv(wpath, index=False)
        log.info("Re-generated %s", wpath)


def describe_carbon_origin(origin: str | None, prov: dict | None) -> str:
    """Human-readable provenance label; anything not clearly real is flagged loudly."""
    if origin in REAL_ORIGINS:
        src = (prov or {}).get("source", "unknown source")
        extra = f", newest observation {prov['end_utc']}" if prov and prov.get("end_utc") else ""
        if origin == "CACHED_REAL_OBSERVATION":
            return f"CACHED_REAL_OBSERVATION (previously imported via {src}{extra}; offline re-use)"
        return f"REAL_OBSERVATION (GB national series via {src}{extra})"
    if origin == "SYNTHETIC_FALLBACK":
        return "SYNTHETIC_FALLBACK (NOT real - the real-data download failed and synthetic data was used)"
    return f"{origin or 'UNKNOWN_ORIGIN'} (provenance unknown - do not treat as real)"


def prepare(cfg: dict) -> Prepared:
    d = cfg["data"]
    if cfg["time_step_minutes"] != 30:
        raise DataError("time_step_minutes must be 30 (the carbon-intensity data is half-hourly).")
    sim_steps = d["simulation_days"] * STEPS_PER_DAY
    total_steps = sim_steps + d["drain_steps"]
    mode = d["mode"]

    if mode == "sample":
        _ensure_sample_files(cfg)
        ci_df = load_carbon_csv(d["sample_carbon_csv"])
        carbon_origin = "SYNTHETIC_SAMPLE (generated, not real)"
        wl = load_workload_csv(d["sample_workload_csv"])
        workload_origin = "SYNTHETIC_SAMPLE (generated, not real)"
        if wl.arrival_step.max() >= sim_steps:
            log.warning("Sample workload longer than simulation window; regenerating in memory.")
            wl = generate_synthetic_workload(sim_steps, cfg["synthetic_workload"], cfg["random_seed"])
    else:
        ci_df = load_carbon_csv(d["real_carbon_csv"])
        carbon_origin = describe_carbon_origin(ci_df.attrs.get("origin"), read_provenance(d["real_carbon_csv"]))
        if resolve_path(d["real_workload_csv"]).exists():
            wl = load_google_trace_extract(d["real_workload_csv"], cfg, sim_steps, cfg["random_seed"])
            workload_origin = "REAL Google 2019 trace fields (arrival, duration, avg CPU/mem) + SIMULATED attributes"
        else:
            log.warning("No Google trace extract at %s; using SYNTHETIC workload (carbon data origin is reported separately).",
                        d["real_workload_csv"])
            wl = generate_synthetic_workload(sim_steps, cfg["synthetic_workload"], cfg["random_seed"])
            workload_origin = "SYNTHETIC_SAMPLE workload (real trace extract not provided)"

    base_ci = ci_df["ci"].to_numpy(dtype=float)
    sim_start = len(base_ci) - total_steps
    min_train = 7 * STEPS_PER_DAY
    if sim_start < min_train:
        raise DataError(f"Carbon series has {len(base_ci)} steps; need at least {min_train + total_steps} "
                        f"({min_train} for training + {total_steps} for simulation). Use more days of data.")
    f = cfg["forecaster"]
    fc = CarbonForecaster(f["max_horizon_steps"], f["max_iter"], f["learning_rate"],
                          cfg["time_step_minutes"], cfg["random_seed"])
    fc.fit(base_ci, sim_start, int(ci_df["minutes_of_day"].iloc[0]), int(ci_df["dow"].iloc[0]))
    metrics = evaluate_forecaster(fc, base_ci, sim_start)
    rolling = rolling_origin_evaluation(
        base_ci, int(ci_df["minutes_of_day"].iloc[0]), int(ci_df["dow"].iloc[0]),
        max_horizon=f["max_horizon_steps"], max_iter=f["max_iter"], learning_rate=f["learning_rate"],
        step_minutes=cfg["time_step_minutes"], seed=cfg["random_seed"])
    mpath = resolve_path(f["model_path"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fc, mpath)
    return Prepared(ci_df, base_ci, sim_start, sim_steps, total_steps, wl, carbon_origin,
                    workload_origin, fc, metrics, rolling)


def build_specs(cfg: dict):
    sites = [SiteSpec(s["site_id"], float(s["ci_scale"]), int(s["ci_shift_steps"])) for s in cfg["sites"]]
    nodes = [NodeSpec(**{k: n[k] for k in ("node_id", "site_id", "tier", "cpu_cores", "mem_gb", "idle_power_w",
                                          "max_power_w", "pue", "latency_ms")}) for n in cfg["nodes"]]
    return sites, nodes


def make_schedulers(cfg: dict):
    return [BaselineScheduler(), CarbonAwareScheduler(cfg, use_carbon=True),
            CarbonAwareScheduler(cfg, use_carbon=False)]


def simulate(cfg: dict, prep: Prepared, workload_df: pd.DataFrame, schedulers=None):
    sites, nodes = build_specs(cfg)
    jobs = workload_from_dataframe(workload_df)
    view = CarbonView(prep.base_ci, sites, prep.forecaster, prep.sim_start)
    results = []
    for sch in (schedulers or make_schedulers(cfg)):
        sim = Simulator(cfg, nodes, jobs, view, sch, prep.sim_steps, prep.total_steps)
        results.append(sim.run())
        log.info("Finished scheduler: %s", sch.name)
    wl_sha = assert_identical_workloads(results)
    ci_sha = hashlib.sha256(np.ascontiguousarray(prep.base_ci).tobytes()).hexdigest()
    for r in results:  # provenance of the experiment, written to scheduler_comparison.csv
        r.summary["workload_sha256"] = wl_sha[:12]
        r.summary["carbon_series_sha256"] = ci_sha[:12]
    return results


_WORKLOAD_KEY_COLS = ["job_id", "arrival_step", "duration_steps", "cpu_req", "mem_req_gb",
                      "latency_req_ms", "slack_steps"]


def workload_fingerprint(jobs_df: pd.DataFrame) -> str:
    """SHA-256 of the workload-defining columns (order-independent)."""
    canon = jobs_df[_WORKLOAD_KEY_COLS].sort_values("job_id").reset_index(drop=True)
    return hashlib.sha256(canon.to_csv(index=False).encode("utf-8")).hexdigest()


def assert_identical_workloads(results) -> str:
    """Fail loudly if schedulers were not compared on exactly the same jobs."""
    prints = [(r.scheduler, workload_fingerprint(r.jobs)) for r in results]
    if len({fp for _, fp in prints}) != 1:
        raise RuntimeError(f"Schedulers saw different workloads: {[(n, fp[:10]) for n, fp in prints]}")
    return prints[0][1]


def paired_scheduler_comparison(cfg: dict, prep: Prepared, n_seeds: int | None = None):
    """Baseline vs carbon-aware vs energy-only ablation on IDENTICAL workloads, repeated over
    seeds. Within a seed every scheduler gets the same jobs, carbon series and forecaster, so
    differences are paired. Workloads are SYNTHETIC (seeded); carbon follows the data mode.

    Returns (per_seed_table, summary_table). Reductions are relative to the baseline of the
    same seed; ``seeds_better`` counts seeds where the scheduler beat that baseline.
    """
    n_seeds = cfg["evaluation"]["robustness_seeds"] if n_seeds is None else n_seeds
    rows = []
    for i in range(n_seeds):
        seed = cfg["random_seed"] + 2000 + i
        wl = generate_synthetic_workload(prep.sim_steps, cfg["synthetic_workload"], seed)
        res = simulate(cfg, prep, wl)
        base = res[0].summary
        for r in res:
            sm = r.summary
            rows.append({
                "seed": seed, "scheduler": sm["scheduler"], "workload_sha256": sm["workload_sha256"],
                "jobs_total": sm["jobs_total"], "completion_rate": sm["completion_rate"],
                "job_emissions_g": sm["job_emissions_g"], "fleet_emissions_g": sm["fleet_emissions_g"],
                "job_emissions_reduction_pct": (base["job_emissions_g"] - sm["job_emissions_g"])
                / base["job_emissions_g"] * 100.0,
                "fleet_emissions_reduction_pct": (base["fleet_emissions_g"] - sm["fleet_emissions_g"])
                / base["fleet_emissions_g"] * 100.0,
                "mean_wait_min": sm["mean_wait_min"], "p95_wait_min": sm["p95_wait_min"],
                "deadline_miss_rate": sm["deadline_miss_rate"], "migrations": sm["migrations"]})
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("scheduler", sort=False).agg(
        n_seeds=("seed", "nunique"),
        job_reduction_mean_pct=("job_emissions_reduction_pct", "mean"),
        job_reduction_std_pct=("job_emissions_reduction_pct", "std"),
        fleet_reduction_mean_pct=("fleet_emissions_reduction_pct", "mean"),
        fleet_reduction_std_pct=("fleet_emissions_reduction_pct", "std"),
        mean_wait_min=("mean_wait_min", "mean"), deadline_miss_rate=("deadline_miss_rate", "mean"),
        completion_rate=("completion_rate", "mean"),
        seeds_better=("fleet_emissions_reduction_pct", lambda s: int((s > 0).sum()))).reset_index()
    return per_seed, summary


def comparison_table(results) -> pd.DataFrame:
    df = pd.DataFrame([r.summary for r in results]).set_index("scheduler")
    base = df.loc["baseline_fcfs_first_fit"]
    for col, new in (("job_emissions_g", "job_emissions_reduction_pct"),
                     ("fleet_emissions_g", "fleet_emissions_reduction_pct"),
                     ("job_energy_kwh", "job_energy_reduction_pct")):
        df[new] = (base[col] - df[col]) / base[col] * 100.0
    return df.reset_index()


def robustness(cfg: dict, prep: Prepared) -> pd.DataFrame:
    """Repeat the baseline vs carbon-aware comparison on freshly generated SYNTHETIC workloads."""
    rows = []
    n = cfg["evaluation"]["robustness_seeds"]
    for i in range(n):
        seed = cfg["random_seed"] + 1000 + i
        wl = generate_synthetic_workload(prep.sim_steps, cfg["synthetic_workload"], seed)
        res = simulate(cfg, prep, wl, [BaselineScheduler(), CarbonAwareScheduler(cfg, True)])
        b, c = res[0].summary, res[1].summary
        rows.append({"seed": seed, "jobs": b["jobs_total"],
                     "baseline_job_emissions_g": b["job_emissions_g"],
                     "carbon_aware_job_emissions_g": c["job_emissions_g"],
                     "job_emissions_reduction_pct": (b["job_emissions_g"] - c["job_emissions_g"]) / b["job_emissions_g"] * 100,
                     "baseline_deadline_miss_rate": b["deadline_miss_rate"],
                     "carbon_aware_deadline_miss_rate": c["deadline_miss_rate"],
                     "carbon_aware_migrations": c["migrations"]})
    return pd.DataFrame(rows)


def deferral_sensitivity(cfg: dict, prep: Prepared) -> pd.DataFrame:
    """Measure the emissions/waiting trade-off for several deferral caps on the same workload."""
    rows = []
    caps = sorted(set([0, 6, 12, 18, 24, 36, 48, int(cfg["carbon_aware_scheduler"].get("max_deferral_steps", 12))]))
    for cap in caps:
        trial_cfg = copy.deepcopy(cfg)
        trial_cfg["carbon_aware_scheduler"]["max_deferral_steps"] = cap
        results = simulate(trial_cfg, prep, prep.workload,
                           [BaselineScheduler(), CarbonAwareScheduler(trial_cfg, use_carbon=True)])
        baseline, carbon = [r.summary for r in results]
        rows.append({
            "max_deferral_steps": cap,
            "max_deferral_hours": cap * cfg["time_step_minutes"] / 60.0,
            "job_emissions_g": carbon["job_emissions_g"],
            "job_emissions_reduction_pct": (
                (baseline["job_emissions_g"] - carbon["job_emissions_g"])
                / baseline["job_emissions_g"] * 100.0 if baseline["job_emissions_g"] else np.nan
            ),
            "fleet_emissions_g": carbon["fleet_emissions_g"],
            "fleet_emissions_reduction_pct": (
                (baseline["fleet_emissions_g"] - carbon["fleet_emissions_g"])
                / baseline["fleet_emissions_g"] * 100.0 if baseline["fleet_emissions_g"] else np.nan
            ),
            "mean_wait_min": carbon["mean_wait_min"],
            "p95_wait_min": carbon["p95_wait_min"],
            "deadline_miss_rate": carbon["deadline_miss_rate"],
            "migrations": carbon["migrations"],
        })
    return pd.DataFrame(rows)


def scenario_evaluation(cfg: dict, prep: Prepared) -> pd.DataFrame:
    """Run targeted stress/sensitivity scenarios against baseline and carbon-aware schedulers.

    Scenario parameters are explicit assumptions; this is a simulator test suite, not a cloud benchmark.
    """
    scenarios = []

    # 1) No temporal or spatial carbon differences: carbon-aware decisions should fall back
    # toward energy-aware choices rather than benefit from an artificial cleaner window.
    constant_cfg = copy.deepcopy(cfg)
    for site in constant_cfg["sites"]:
        site["ci_scale"], site["ci_shift_steps"] = 1.0, 0
    constant_prep = copy.copy(prep)
    constant_prep.base_ci = np.full_like(prep.base_ci, float(np.mean(prep.base_ci)))
    f = constant_cfg["forecaster"]
    constant_fc = CarbonForecaster(f["max_horizon_steps"], f["max_iter"], f["learning_rate"],
                                   constant_cfg["time_step_minutes"], constant_cfg["random_seed"])
    constant_fc.fit(constant_prep.base_ci, constant_prep.sim_start,
                    int(prep.ci_df["minutes_of_day"].iloc[0]), int(prep.ci_df["dow"].iloc[0]))
    constant_prep.forecaster = constant_fc
    scenarios.append(("constant_carbon_equal_sites", constant_cfg, constant_prep, prep.workload.copy()))

    # 2) Stronger cleaner-remote-site assumption, to test whether the placement heuristic reacts.
    cleaner_cfg = copy.deepcopy(cfg)
    for site in cleaner_cfg["sites"]:
        if site["site_id"] == "cloud-region-1":
            site["ci_scale"] = 0.5
    scenarios.append(("cleaner_cloud_assumption", cleaner_cfg, prep, prep.workload.copy()))

    # 3) Tight deadline case: no deferral slack is granted to any generated job.
    tight_cfg = copy.deepcopy(cfg)
    tight_cfg["synthetic_workload"]["slack_choices_steps"] = [0]
    tight_wl = generate_synthetic_workload(prep.sim_steps, tight_cfg["synthetic_workload"],
                                           cfg["random_seed"] + 77)
    scenarios.append(("zero_slack_workload", tight_cfg, prep, tight_wl))

    # 4) Make migration transfer energy intentionally prohibitive; migration should not be selected
    # unless the simulated carbon savings are implausibly large.
    expensive_cfg = copy.deepcopy(cfg)
    expensive_cfg["migration"]["transfer_kwh_per_gb"] = 1000.0
    scenarios.append(("high_migration_cost", expensive_cfg, prep, prep.workload.copy()))

    # 5) Stress the scheduler with a systematic +30 gCO2/kWh forecast bias. Actual emissions
    # accounting remains based on the unmodified observed series.
    biased_prep = copy.copy(prep)
    class BiasedForecaster:
        def __init__(self, base, bias):
            self.base, self.H, self.bias = base, base.H, bias
        def predict_matrix(self, ci, anchors, ks):
            return np.clip(self.base.predict_matrix(ci, anchors, ks) + self.bias, 0.0, None)
    biased_prep.forecaster = BiasedForecaster(prep.forecaster, 30.0)
    scenarios.append(("forecast_bias_plus_30", copy.deepcopy(cfg), biased_prep, prep.workload.copy()))

    # 6) Resource pressure: shrink per-node CPU and memory capacity. Jobs that cannot be
    # placed remain pending and are reported as unfinished instead of disappearing.
    constrained_cfg = copy.deepcopy(cfg)
    for node in constrained_cfg["nodes"]:
        node["cpu_cores"] = max(1.0, float(node["cpu_cores"]) * 0.25)
        node["mem_gb"] = max(1.0, float(node["mem_gb"]) * 0.25)
    scenarios.append(("resource_constrained", constrained_cfg, prep, prep.workload.copy()))

    rows = []
    for scenario_name, trial_cfg, trial_prep, workload in scenarios:
        results = simulate(trial_cfg, trial_prep, workload,
                           [BaselineScheduler(), CarbonAwareScheduler(trial_cfg, use_carbon=True)])
        baseline, carbon = [r.summary for r in results]
        for summary in (baseline, carbon):
            is_carbon = summary["scheduler"] == "carbon_aware"
            base_job = baseline["job_emissions_g"]
            base_fleet = baseline["fleet_emissions_g"]
            rows.append({
                "scenario": scenario_name,
                "scheduler": summary["scheduler"],
                "jobs_total": summary["jobs_total"],
                "jobs_completed": summary["jobs_completed"],
                "completion_rate": summary["completion_rate"],
                "jobs_unfinished": summary["jobs_unfinished"],
                "job_emissions_g": summary["job_emissions_g"],
                "fleet_emissions_g": summary["fleet_emissions_g"],
                "job_emissions_reduction_vs_scenario_baseline_pct": (
                    (base_job - summary["job_emissions_g"]) / base_job * 100.0 if base_job else np.nan
                ),
                "fleet_emissions_reduction_vs_scenario_baseline_pct": (
                    (base_fleet - summary["fleet_emissions_g"]) / base_fleet * 100.0 if base_fleet else np.nan
                ),
                "mean_wait_min": summary["mean_wait_min"],
                "p95_wait_min": summary["p95_wait_min"],
                "deadline_miss_rate": summary["deadline_miss_rate"],
                "latency_violations": summary["latency_violations"],
                "migrations": summary["migrations"],
            })
    return pd.DataFrame(rows)
