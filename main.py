"""Carbon-Aware Edge Scheduling for Sustainable Cloud Computing - Team SkyHighOps.

Usage (from the project folder, with the virtual environment active):
    python main.py                       # run everything with the included SAMPLE data
    python main.py run --mode real       # use real data files in data/real/ (see README)
    python main.py download-carbon --start 2025-01-01 --end 2025-02-15
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.config import load_config
from src.carbon_import import DEFAULT_OUT, SOURCES, import_carbon
from src.data_sources import download_carbon_intensity
from src.errors import ProjectError
from src import pipeline, reporting


def cmd_run(args) -> int:
    overrides = {"data": {"mode": args.mode}} if args.mode else None
    cfg = load_config(args.config, overrides)
    if args.seed is not None:
        cfg["random_seed"] = args.seed
    print(f"[1/6] Loading data and training the carbon-intensity forecaster (mode={cfg['data']['mode']}) ...")
    prep = pipeline.prepare(cfg)
    print(f"      carbon data : {prep.carbon_origin}\n      workload    : {prep.workload_origin} ({len(prep.workload)} jobs)")
    print("[2/6] Running schedulers (baseline, carbon-aware, energy-only ablation) ...")
    results = pipeline.simulate(cfg, prep, prep.workload)
    comparison = pipeline.comparison_table(results)
    robust = None
    if cfg["data"]["mode"] == "sample" and cfg["evaluation"]["robustness_seeds"] > 0 and not args.skip_robustness:
        print("[3/6] Robustness check on extra synthetic workloads ...")
        robust = pipeline.robustness(cfg, prep)
    else:
        print("[3/6] Robustness check skipped.")
    print("[4/6] Measuring the deferral emissions/waiting trade-off ...")
    sensitivity = pipeline.deferral_sensitivity(cfg, prep)
    print("[5/6] Running targeted stress scenarios (carbon, deadlines, migration, forecast, capacity) ...")
    scenarios = pipeline.scenario_evaluation(cfg, prep)
    paired = None
    if cfg["evaluation"]["robustness_seeds"] > 0 and not args.skip_robustness:
        print("[5b] Paired scheduler comparison on identical synthetic workloads (3 schedulers x seeds) ...")
        paired = pipeline.paired_scheduler_comparison(cfg, prep)
    print("[6/6] Writing CSVs, graphs and report to results/ ...")
    reporting.make_all(cfg, prep, results, comparison, robust, sensitivity, scenarios, paired)
    show = comparison[["scheduler", "job_emissions_g", "fleet_emissions_g", "job_emissions_reduction_pct",
                       "deadline_miss_rate", "migrations"]].round(2)
    print("\n" + show.to_string(index=False))
    print("\nDone. Open results/dashboard.html in your browser (and results/report.md).")
    return 0


def cmd_download(args) -> int:
    n = download_carbon_intensity(args.start, args.end, args.out)
    print(f"Saved {n} half-hourly rows to {args.out}. Now run: python main.py run --mode real")
    return 0


def cmd_import(args) -> int:
    res = import_carbon(args.days, args.out, args.source)
    tag = "REAL" if res.is_real else "NOT REAL (synthetic fallback)"
    print(f"Saved {res.rows} rows to {res.out_path}\n  source : {res.source}\n  origin : {res.data_origin} [{tag}]\n"
          f"  window : {res.start_utc} -> {res.end_utc} (newest observation is {res.data_age_hours} h old)")
    for n in res.notes:
        print(f"  note   : {n}")
    if res.is_real:
        print("Now run: python main.py run --mode real")
    else:
        print("WARNING: no real data could be obtained; results from this file must not be reported as real.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command")
    r = sub.add_parser("run", help="run the full experiment (default)")
    r.add_argument("--mode", choices=["sample", "real"], help="override data.mode in the config")
    r.add_argument("--config", help="optional JSON file overriding default_config.json values")
    r.add_argument("--seed", type=int)
    r.add_argument("--skip-robustness", action="store_true")
    r.set_defaults(func=cmd_run)
    d = sub.add_parser("download-carbon", help="download real GB carbon intensity (no API key needed)")
    d.add_argument("--start", required=True, help="YYYY-MM-DD")
    d.add_argument("--end", required=True, help="YYYY-MM-DD")
    d.add_argument("--out", default="data/real/carbon_intensity_gb.csv")
    d.set_defaults(func=cmd_download)
    i = sub.add_parser("import-carbon", help="import real GB carbon data (NESO CSV / UK API) with offline fallback")
    i.add_argument("--days", type=int, default=60, help="most recent days to keep (>= 16)")
    i.add_argument("--source", choices=SOURCES, default="auto")
    i.add_argument("--out", default=DEFAULT_OUT)
    i.set_defaults(func=cmd_import)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")
    if args.command is None:  # plain `python main.py`
        args = parser.parse_args(["run"] + (argv or []))
    try:
        return args.func(args)
    except ProjectError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
