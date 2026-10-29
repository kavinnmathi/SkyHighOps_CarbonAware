"""Save CSVs, PNG graphs and a Markdown results report."""
from __future__ import annotations

from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import resolve_path
from .forecasting import CarbonForecaster

_TAG = {"text": ""}
BASE, CA, EN = "baseline_fcfs_first_fit", "carbon_aware", "energy_aware_ablation"
COLORS = {BASE: "#888888", CA: "#2a9d4b", EN: "#e0a030"}


def _finish(fig, name: str) -> None:
    """Stamp every graph with its data origin, then save it."""
    fig.text(0.5, 0.005, _TAG["text"], ha="center", va="bottom", fontsize=7, color="#a03030")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(_out(name), dpi=130)
    plt.close(fig)


def _out(name: str):
    d = resolve_path("results")
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def save_csvs(prep, results, comparison, robust, sensitivity=None, scenarios=None) -> None:
    prep.forecast_metrics.to_csv(_out("forecaster_metrics.csv"), index=False)
    comparison.to_csv(_out("scheduler_comparison.csv"), index=False)
    pd.concat([r.jobs for r in results]).to_csv(_out("job_results.csv"), index=False)
    pd.concat([r.timeseries for r in results]).to_csv(_out("timeseries.csv"), index=False)
    mig = pd.concat([r.migrations for r in results if len(r.migrations)], ignore_index=True) \
        if any(len(r.migrations) for r in results) else pd.DataFrame(
            columns=["scheduler", "step", "job_id", "from_node", "to_node"])
    mig.to_csv(_out("migrations.csv"), index=False)
    if robust is not None:
        robust.to_csv(_out("robustness.csv"), index=False)
    if sensitivity is not None:
        sensitivity.to_csv(_out("deferral_sensitivity.csv"), index=False)
    if scenarios is not None:
        scenarios.to_csv(_out("scenario_results.csv"), index=False)


def plot_forecast(prep, k: int = 4) -> None:
    fc: CarbonForecaster = prep.forecaster
    anchors = np.arange(prep.sim_start, prep.sim_start + 3 * 48)
    pred = fc.predict_matrix(prep.base_ci, anchors, np.array([k]))[:, 0]
    truth = prep.base_ci[anchors + k]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(truth, label="Actual (observed series)", color="black")
    ax.plot(pred, label=f"Model prediction ({k * 0.5:g} h ahead)", color="#2a9d4b")
    ax.plot(prep.base_ci[anchors], label="Persistence baseline", color="#bbbbbb", linestyle="--")
    ax.set_xlabel("Half-hour steps from start of test period")
    ax.set_ylabel("gCO2/kWh")
    ax.set_title("Carbon-intensity forecast vs actual (held-out test period)")
    ax.legend()
    _finish(fig, "forecast_vs_actual.png")


def plot_forecast_error(prep) -> None:
    m = prep.forecast_metrics
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, g in m.groupby("method"):
        ax.plot(g.horizon_hours, g.mae, marker="o", label=name)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE (gCO2/kWh)")
    ax.set_title("Forecast error by horizon (held-out period)")
    ax.legend()
    _finish(fig, "forecast_error_by_horizon.png")


def plot_emissions(comparison) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in ((axes[0], "job_emissions_g", "Job-attributed emissions (incl. migration)"),
                           (axes[1], "fleet_emissions_g", "Whole-fleet emissions (incl. idle power)")):
        vals = comparison.set_index("scheduler")[col] / 1000.0
        ax.bar(range(len(vals)), vals.values, color=[COLORS.get(i, "#555") for i in vals.index])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([i.replace("_", "\n") for i in vals.index], fontsize=7)
        ax.set_ylabel("kg CO2")
        ax.set_title(title, fontsize=10)
    _finish(fig, "emissions_comparison.png")


def plot_cumulative(results) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in results:
        ax.plot(r.timeseries.fleet_emissions_g.cumsum() / 1000.0, label=r.scheduler, color=COLORS.get(r.scheduler))
    ax.set_xlabel("Half-hour step")
    ax.set_ylabel("Cumulative fleet kg CO2")
    ax.set_title("Cumulative emissions over the simulation")
    ax.legend()
    _finish(fig, "cumulative_emissions.png")


def plot_placement(results) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    nodes = sorted({n for r in results for n in r.jobs.final_node.dropna().unique()})
    w = 0.8 / len(results)
    for i, r in enumerate(results):
        e = r.jobs.groupby("final_node").run_energy_kwh.sum().reindex(nodes).fillna(0)
        ax.bar(np.arange(len(nodes)) + i * w, e.values, w, label=r.scheduler, color=COLORS.get(r.scheduler))
    ax.set_xticks(np.arange(len(nodes)) + w * (len(results) - 1) / 2)
    ax.set_xticklabels(nodes, rotation=30, ha="right")
    ax.set_ylabel("Job energy (kWh, simulated)")
    ax.set_title("Where each scheduler placed the work")
    ax.legend(fontsize=8)
    _finish(fig, "placement_by_node.png")


def write_report(cfg, prep, results, comparison, robust, sensitivity=None, scenarios=None) -> None:
    c = comparison.set_index("scheduler")
    cols = ["job_energy_kwh", "job_emissions_g", "fleet_emissions_g", "migrations", "mean_wait_min",
            "p95_wait_min", "deadline_miss_rate", "latency_violations", "job_emissions_reduction_pct",
            "fleet_emissions_reduction_pct"]
    table = c[cols].round(3).to_markdown() if _has_tabulate() else "```\n" + c[cols].round(3).to_string() + "\n```"
    fm = prep.forecast_metrics
    fm_tab = fm[fm.horizon_steps.isin([1, 4, 12, 48])].round(3)
    fm_txt = "```\n" + fm_tab.to_string(index=False) + "\n```"
    piv = fm.pivot(index="horizon_steps", columns="method", values="mae")
    best_base = piv[["persistence", "seasonal_naive_24h"]].min(axis=1)
    wins = [int(h) for h in piv.index[piv["gradient_boosting"] < best_base]]
    loses = [int(h) for h in piv.index[piv["gradient_boosting"] >= best_base]]
    verdict = (f"The model has lower MAE than the best simple baseline at horizons (steps) {wins or 'none'}; "
               f"it does NOT beat the best baseline at {loses or 'none'}. This is reported as-is.")
    rob_txt = "Not run (only available in sample mode)."
    if robust is not None and len(robust):
        r = robust.job_emissions_reduction_pct
        rob_txt = (f"Across {len(robust)} freshly generated SYNTHETIC workloads the job-emissions reduction of the "
                   f"carbon-aware scheduler vs. the baseline was {r.mean():.1f}% +/- {r.std(ddof=0):.1f}% "
                   f"(min {r.min():.1f}%, max {r.max():.1f}%). See `robustness.csv`.")
    sensitivity_txt = "Not run."
    if sensitivity is not None and len(sensitivity):
        best = sensitivity.loc[sensitivity.job_emissions_g.idxmin()]
        sensitivity_txt = (
            "```" + "\n" + sensitivity.round(3).to_string(index=False) + "\n```" +
            f"\nLowest job emissions in this sweep occurred at a {best.max_deferral_hours:.1f}-hour "
            f"deferral cap ({best.job_emissions_reduction_pct:.1f}% lower than baseline; "
            f"mean wait {best.mean_wait_min:.1f} min, p95 wait {best.p95_wait_min:.1f} min). "
            "This is a single synthetic workload sensitivity check, not a universal optimum."
        )
    scenario_txt = "Not run."
    if scenarios is not None and len(scenarios):
        sc = scenarios[scenarios.scheduler == CA].copy()
        keep = ["scenario", "jobs_completed", "jobs_unfinished", "job_emissions_g",
                "job_emissions_reduction_vs_scenario_baseline_pct", "deadline_miss_rate",
                "latency_violations", "migrations"]
        scenario_txt = "```" + "\n" + sc[keep].round(3).to_string(index=False) + "\n```\n"
        scenario_txt += (
            "Interpretation: these are targeted stress tests using altered assumptions. "
            "In particular, cleaner-site and forecast-bias scenarios are sensitivity experiments, "
            "not claims about actual cloud regions or forecast error distributions."
        )
    ca, base = c.loc[CA], c.loc[BASE]
    en = c.loc[EN] if EN in c.index else None
    en_txt = ""
    if en is not None:
        en_txt = (f"\nThe energy-only ablation (`{EN}`, same placement logic but blind to carbon) reduces job emissions "
                  f"by {en.job_emissions_reduction_pct:.1f}% vs. baseline. The *additional* effect of carbon-awareness "
                  f"(carbon-aware vs. energy-only) is "
                  f"{(en.job_emissions_g - ca.job_emissions_g) / en.job_emissions_g * 100:.1f}%.\n")
    sites = "\n".join(f"- {s['site_id']}: ci_scale={s['ci_scale']}, ci_shift_steps={s['ci_shift_steps']}" for s in cfg["sites"])
    txt = f"""# Results report - Carbon-Aware Edge Scheduling (Team SkyHighOps)

Generated: {datetime.now():%Y-%m-%d %H:%M}   |   Data mode: **{cfg['data']['mode']}**

## 1. What is real, predicted and simulated?

| Item | Status |
|---|---|
| Grid carbon intensity series | {prep.carbon_origin} |
| Workload | {prep.workload_origin} |
| Future carbon intensity used by the scheduler | **MODEL PREDICTION** (scikit-learn HistGradientBoostingRegressor) |
| Emissions accounting | uses the **actual** series in the data file (never the predictions) |
| Per-site carbon intensity | **SIMULATED** transform of the base series (scale / time shift per site, see below) |
| Node power, PUE, latency, migration overhead | **SIMULATED assumptions** from `config/default_config.json` |
| Migrations | **simulated bookkeeping only**; no physical cloud migration takes place |

Site transforms (assumptions):
{sites}

## 2. Carbon-intensity forecaster (real supervised regression target)

Target: carbon intensity k half-hour steps ahead. Trained only on data before the simulation window
({prep.sim_start} steps), evaluated on the held-out simulation window.

{fm_txt}

{verdict}

## 3. Scheduler comparison (simulation window: {cfg['data']['simulation_days']} days + {cfg['data']['drain_steps']} drain steps)

{table}

* `job_emissions_g`: emissions attributed to jobs (idle power shared by reserved CPU, dynamic power by used CPU) plus migration overhead.
* `fleet_emissions_g`: all nodes' power incl. idle (nodes are always on) plus migration overhead. Idle power is the same for every scheduler, so this metric shows smaller relative differences.
* Emissions = energy (kWh) x carbon intensity (gCO2/kWh).

Headline (sample/simulated results, not a claim about real clouds): the carbon-aware scheduler changed job emissions by
**{ca.job_emissions_reduction_pct:.1f}%** and fleet emissions by **{ca.fleet_emissions_reduction_pct:.1f}%** relative to the baseline,
with deadline-miss rate {ca.deadline_miss_rate:.3f} (baseline {base.deadline_miss_rate:.3f}) and {int(ca.latency_violations)} latency violations.
{en_txt}
## 4. Robustness

{rob_txt}

## 5. Deferral sensitivity (same workload, different wait caps)

{ sensitivity_txt }

## 6. Targeted stress scenarios

{scenario_txt}

Full rows for baseline and carbon-aware schedulers are in `scenario_results.csv`. Resource-constrained scenarios may leave jobs unfinished; this is reported explicitly rather than hiding them.

## 7. Limitations

* Results depend heavily on the assumed node power values and site carbon scales/shifts; change them in the config and re-run.
* Sample mode uses SYNTHETIC data. Numbers from sample mode demonstrate that the pipeline works, not real-world savings.
* The scheduler is told each job's duration and CPU utilisation (no estimation error).
* Migration overhead is a simple model (transfer time + fixed downtime + assumed kWh/GB).
* The scenario sweeps are controlled sensitivity tests, not statistically representative of real workloads; the 5-seed robustness check uses synthetic workloads in sample mode.

## 8. Files

`scheduler_comparison.csv`, `forecaster_metrics.csv`, `job_results.csv`, `timeseries.csv`, `migrations.csv`,
`robustness.csv`, `deferral_sensitivity.csv`, `scenario_results.csv`, and the PNG graphs in this folder.
"""
    _out("report.md").write_text(txt, encoding="utf-8")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def _tbl(df) -> str:
    return df.to_markdown(index=False) if _has_tabulate() else "```\n" + df.to_string(index=False) + "\n```"


def write_validation_sections(cfg, prep, paired=None) -> None:
    """Save rolling-origin forecast results + paired scheduler comparison and append them to report.md."""
    from .forecasting import summarise_rolling
    from .carbon_import import read_provenance
    parts = ["\n## 9. Leakage-safe forecast evaluation (rolling origin)\n"]
    if prep.forecast_rolling is not None:
        prep.forecast_rolling.to_csv(_out("forecast_rolling_folds.csv"), index=False)
        summ = summarise_rolling(prep.forecast_rolling)
        summ.to_csv(_out("forecast_rolling_summary.csv"), index=False)
        show = summ[summ.method != "persistence"].round(3)[
            ["horizon_steps", "method", "mae_mean", "mae_std", "skill_vs_persistence", "n_folds"]]
        parts.append("Each fold trains a fresh model on data strictly before its test block (expanding window, "
                     "no shuffling) and scores the next block. `skill_vs_persistence` > 0 means lower MAE than "
                     "persistence. At 24 h (48 steps) the seasonal-naive baseline equals persistence by "
                     "construction. Full detail: `forecast_rolling_folds.csv`, `forecast_rolling_summary.csv`.\n\n"
                     + _tbl(show))
    if paired is not None:
        per_seed, summ = paired
        per_seed.to_csv(_out("paired_comparison_by_seed.csv"), index=False)
        summ.to_csv(_out("paired_comparison_summary.csv"), index=False)
        parts.append("\n## 10. Paired scheduler comparison (identical workloads)\n\n"
                     "Within each seed all three schedulers run the same jobs (verified by a workload SHA-256), "
                     "the same carbon series and the same forecaster. Workloads are SYNTHETIC; carbon data origin: "
                     f"{prep.carbon_origin}.\n\n" + _tbl(summ.round(3)))
    prov = read_provenance(cfg["data"]["real_carbon_csv"]) if cfg["data"]["mode"] == "real" else None
    if prov:
        (_out("carbon_provenance.json")).write_text(__import__("json").dumps(prov, indent=2), encoding="utf-8")
        parts.append(f"\n\nCarbon provenance: source `{prov.get('source')}`, origin `{prov.get('data_origin')}`, "
                     f"newest observation {prov.get('end_utc')}, retrieved {prov.get('retrieved_utc')} "
                     "(`carbon_provenance.json`).")
    with open(_out("report.md"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(parts) + "\n")


def make_all(cfg, prep, results, comparison, robust, sensitivity=None, scenarios=None, paired=None) -> None:
    _TAG["text"] = (f"Data mode: {cfg['data']['mode']} | carbon data: {prep.carbon_origin.split(' (')[0]} | "
                    f"workload: {prep.workload_origin.split(' (')[0].split(' +')[0]} | simulated scheduling")
    save_csvs(prep, results, comparison, robust, sensitivity, scenarios)
    plot_forecast(prep)
    plot_forecast_error(prep)
    plot_emissions(comparison)
    plot_cumulative(results)
    plot_placement(results)
    write_report(cfg, prep, results, comparison, robust, sensitivity, scenarios)
    write_validation_sections(cfg, prep, paired)
    from . import dashboard
    dashboard.write(cfg, prep)
