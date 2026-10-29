# Carbon-Aware Edge Scheduling for Sustainable Cloud Computing
**Team SkyHighOps** - college research project (simulation only)

A beginner-friendly Python project that simulates an edge/cloud cluster, forecasts grid carbon intensity with scikit-learn, and compares a **conventional baseline scheduler** with a **carbon-aware scheduler** (placement, delay of tolerant jobs, migration with overhead).

Everything is a **simulation**. No real cloud migration happens. CO2 = energy (kWh) x carbon intensity (gCO2/kWh).

* Base paper: Danach et al., *IET Smart Grid* 9(1), e70056, first published 14 Jan 2026, DOI 10.1049/stg2.70056 - see `docs/BASE_PAPER.md` (including what I could and could not verify).
* Datasets: UK National Grid Carbon Intensity API + Google Cluster Data 2019 - see `docs/DATASETS.md`.
* **Default run uses included SYNTHETIC sample data** (clearly labelled, not real). Real data is optional.

## 1. Install and run (Windows)
Install Python 3.10 or newer from python.org (tick "Add Python to PATH"). Then open **Command Prompt** in the project folder (type `cmd` in the folder's address bar):

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

PowerShell users: activate with `.venv\Scripts\Activate.ps1`. If PowerShell blocks it, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or just use Command Prompt.

The run takes about a minute on a normal laptop (no GPU, no internet). It creates in `results/`:

| File | Content |
|---|---|
| `dashboard.html` | **interactive-style dashboard**: double-click to open in any browser (works offline) |
| `report.md` | readable report: real vs predicted vs simulated, tables, limitations |
| `scheduler_comparison.csv` | metrics per scheduler incl. % reductions |
| `forecaster_metrics.csv` | MAE / RMSE / R2 of the model and 2 baselines per horizon |
| `job_results.csv`, `timeseries.csv`, `migrations.csv`, `robustness.csv`, `deferral_sensitivity.csv`, `scenario_results.csv` | detailed outputs and stress-test results |
| `forecast_rolling_folds.csv`, `forecast_rolling_summary.csv` | leakage-safe walk-forward forecast evaluation |
| `paired_comparison_by_seed.csv`, `paired_comparison_summary.csv` | baseline vs carbon-aware vs ablation on identical workloads per seed |
| `forecast_vs_actual.png`, `forecast_error_by_horizon.png`, `emissions_comparison.png`, `cumulative_emissions.png`, `placement_by_node.png` | graphs |

The trained forecaster is saved to `models/ci_forecaster.joblib`.

### Run the tests
```bat
python -m unittest discover -s tests -t . -v
```

## 2. Optional: use real data
**Real carbon intensity (free, no registration, offline fallback):**
```bat
python main.py import-carbon --days 60
python main.py run --mode real
```
`import-carbon` tries, in order: the NESO historic generation-mix CSV (`CARBON_INTENSITY` column), then the UK Carbon Intensity API, then a previously imported real file, then - as a last resort - a synthetic file labelled `SYNTHETIC_FALLBACK`. Use `--source neso|ukapi|offline` to force one. Every file gets a `data_origin` column and a `carbon_intensity_gb.csv.provenance.json` (source, URL, retrieval time, newest observation, SHA-256). `run --mode real` prints the origin and writes it into `report.md`; a fallback is labelled **NOT real**. The NESO file is a periodically updated historical file, so "newest observation" and its age are recorded rather than claiming live data.

The older `download-carbon --start --end` command (UK API only) still works.

**Real workload (Google 2019 trace, manual):** needs a Google Cloud account because the trace is only on BigQuery. Follow `docs/DATASETS.md` and `docs/google_trace_bigquery.sql`, save the CSV as `data/real/google_trace_extract.csv`, then run `python main.py run --mode real`.

## 3. Changing the experiment
Edit `config/default_config.json` (nodes, power values, sites, migration overhead, scheduler margins) or pass your own file containing only the keys to change:
```bat
python main.py run --config my_changes.json
```
Other options: `--seed 7`, `--skip-robustness`, `-v` (verbose).

## 4. Main modules (`src/`)
| Module | Purpose |
|---|---|
| `config.py`, `errors.py` | load/validate JSON config, readable error messages |
| `carbon_import.py` | NESO / UK-API importer, offline fallback, provenance sidecar |
| `data_sources.py` | load/clean carbon CSV, synthetic sample generators, Google-trace converter, API downloader |
| `forecasting.py` | scikit-learn `HistGradientBoostingRegressor` forecasting carbon intensity 1-48 steps ahead; baselines; `CarbonView` (actual vs predicted per site) |
| `energy.py` | power model, kWh, gCO2, per-job attribution, migration overhead |
| `models.py` | `NodeSpec`, `JobSpec`, runtime state |
| `schedulers.py` | baseline FCFS first-fit; carbon-aware scheduler (also run with carbon switched off as an *energy-only ablation*) |
| `simulator.py` | 30-minute discrete-time simulation, accounting, metrics |
| `pipeline.py` | prepare data, train forecaster, run schedulers, robustness runs |
| `reporting.py` | CSVs, PNG graphs, `report.md` |

## 5. How to read the results honestly
* Sample-mode numbers prove the pipeline works; they are **not** real-world savings.
* Part of the saving comes from assumed efficient cloud hardware. Compare `carbon_aware` against `energy_aware_ablation` to see the share due to carbon-awareness itself.
* Carbon-aware delays tolerant jobs, so waiting time can rise; deadline-miss rate, completion rate, and wait times are reported next to emissions. Deferral is capped by `carbon_aware_scheduler.max_deferral_steps` (default 12 half-hour steps = 6 hours) in addition to each job's own slack.
* Migration downtime is modelled as a pause in execution, including fractional time steps; the job does not make progress during the paused portion. Migration transfer energy/emissions are recorded separately.
* Per-site carbon intensity is a simulated transformation of one national series.
Algorithm walkthrough and design decisions: `docs/SCHEDULING_ALGORITHM.md`. Full model assumptions: `docs/ASSUMPTIONS_AND_LIMITATIONS.md`. Final test/reproducibility status: `docs/VALIDATION_STATUS.md`.

## 6. Known limitations and manual steps
* The Google trace extract is a manual step (BigQuery account).
* The NESO importer and the UK API downloader are tested against mocked/fabricated responses only. **Live access has not been verified** (the build environment had no DNS). The NESO column names (`DATETIME`, `CARBON_INTENSITY`) are from the dataset's documented layout; if the live file differs, the importer raises a clear error listing the columns found.
* The BigQuery query must be checked against the official schema before use.
* The scheduler is a heuristic, not the paper's mixed-integer optimiser.

## 7. Project layout
```
main.py  requirements.txt  README.md
config/default_config.json
data/sample/  data/real/  data/README.md
docs/  models/  results/  src/  tests/
```
