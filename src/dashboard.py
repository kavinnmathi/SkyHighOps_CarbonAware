"""Build results/dashboard.html: a single self-contained page (no internet needed)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import resolve_path


def _records(df: pd.DataFrame) -> list:
    return json.loads(df.to_json(orient="records"))


def write(cfg: dict, prep) -> Path:
    r = resolve_path("results")
    comp = pd.read_csv(r / "scheduler_comparison.csv")
    ts = pd.read_csv(r / "timeseries.csv")
    jobs = pd.read_csv(r / "job_results.csv")
    rob_path = r / "robustness.csv"
    data = {
        "meta": {"mode": cfg["data"]["mode"], "days": cfg["data"]["simulation_days"],
                 "carbon_origin": prep.carbon_origin, "workload_origin": prep.workload_origin},
        "comparison": _records(comp),
        "forecast": _records(pd.read_csv(r / "forecaster_metrics.csv")),
        "cumulative": {s: (g.fleet_emissions_g.cumsum() / 1000).round(4).tolist() for s, g in ts.groupby("scheduler")},
        "placement": {s: g.groupby("final_node").run_energy_kwh.sum().round(4).to_dict() for s, g in jobs.groupby("scheduler")},
        "robustness": _records(pd.read_csv(rob_path)) if rob_path.exists() else [],
        "deferral_sensitivity": _records(pd.read_csv(r / "deferral_sensitivity.csv"))
            if (r / "deferral_sensitivity.csv").exists() else [],
        "scenarios": _records(pd.read_csv(r / "scenario_results.csv"))
            if (r / "scenario_results.csv").exists() else [],
    }
    tpl = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    out = r / "dashboard.html"
    out.write_text(tpl.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    return out
