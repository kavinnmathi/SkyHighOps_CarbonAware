# Validation status — Team SkyHighOps

Verification performed on the packaged project in a fresh extraction, using Python 3.13.5 in the available environment.

## Completed checks

- [x] Compiled `main.py`, all `src/*.py`, and all tests with `python -m compileall -q main.py src tests`.
- [x] Ran `python -m unittest discover -s tests -t . -v`: **55 tests passed** (33 original + 22 new: importer/provenance, leakage, identical-workload).
- [x] Deleted generated model/results (preserving `.gitkeep`) and ran `python main.py` from the project directory; all six pipeline stages completed and all expected CSV, PNG, Markdown and HTML artifacts were created.
- [x] Repeated the same seeded full sample run. SHA-256 hashes matched for `scheduler_comparison.csv`, `forecaster_metrics.csv`, `deferral_sensitivity.csv`, `scenario_results.csv`, and `robustness.csv`, confirming repeatability of those outputs in this environment.
- [x] Tested CPU/memory/latency feasibility, wait/deadline metric consistency, migration pause behavior, migration configuration validation, stress-scenario reporting and explicit unfinished-job accounting.
- [x] Tested the real-mode missing-input behavior: `python main.py run --mode real` exits cleanly with code 2 and an actionable missing-carbon-file message when no real carbon CSV is present.
- [x] Tested downloader date validation and parsing/writing of a mocked API response without requiring network access.
- [x] Tested Google-trace extract conversion with a tiny fabricated fixture and invalid-column handling.
- [x] Added `docs/SCHEDULING_ALGORITHM.md` to document the FCFS/first-fit baseline, carbon-aware placement/deferral, energy-only ablation, migration heuristic, metrics, design choices and limitations.

## Results from the reproducible sample run

These are **synthetic simulation outputs**, not measured cloud savings:

| Scheduler | Job-attributed emissions | Fleet emissions | Job-emissions reduction vs baseline | Jobs completed | Migrations |
|---|---:|---:|---:|---:|---:|
| FCFS + first-fit baseline | 8,878.13 g | 23,123.36 g | 0.00% | 955/955 | 0 |
| Carbon-aware | 4,334.49 g | 20,548.30 g | 51.18% | 955/955 | 1 |
| Energy-only ablation | 4,531.53 g | 20,662.68 g | 48.96% | 955/955 | 0 |

The carbon-aware sample run has mean wait around 78.3 minutes and p95 wait of 6 hours. This is the emissions/waiting trade-off on the bundled synthetic workload; other workloads and real data can produce different outcomes.

## Not verified live

The live UK Carbon Intensity API could not be reached from this execution environment because DNS resolution failed (`Temporary failure in name resolution`). The downloader's input validation and mocked-response behavior are tested, but **live connectivity and current API response behavior remain unverified**. Retry the documented `download-carbon` command from a normal internet-connected machine before relying on real-data mode.

The Google Cluster Data extract is still a manual BigQuery export. The included starter SQL must be checked against the official schema and tested in BigQuery before treating that path as end-to-end validated.

## Update: importer, forecast evaluation, paired comparison

Verified in the build sandbox (no internet): 55 unit tests pass; sample run reproduces the earlier headline numbers exactly (8,878.13 / 4,334.49 / 4,531.53 g); `import-carbon` with no network falls back to a file labelled `SYNTHETIC_FALLBACK`, and `run --mode real` then reports that origin as NOT real.

**Still not verified:** live download from NESO (`api.neso.energy`) and the UK API - DNS failed here. The NESO parser is tested only on fabricated fixtures built from the documented `DATETIME` / `CARBON_INTENSITY` columns. Run `python main.py import-carbon --days 60 --source neso` on a connected machine and check the printed window and age.

Forecast findings on the synthetic sample (rolling origin, 4 folds): the model beats persistence at 0.5-12 h, is about level with seasonal-naive at 12 h, and is worse than the baselines at 24 h. Treat the 24 h horizon as unreliable until re-checked on real data.
