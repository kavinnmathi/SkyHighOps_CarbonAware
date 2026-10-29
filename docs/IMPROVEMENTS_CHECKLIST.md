# Improvement and verification checklist

## Scheduler and simulation changes

- [x] Cap job deferral (`max_deferral_steps`, default 12 half-hour steps / 6 hours).
- [x] Restrict deferral comparisons to nodes feasible at the current step; document future contention as an approximation.
- [x] Make migration downtime pause execution rather than adding extra work while execution continues.
- [x] Attribute active power during execution and reserved idle share during downtime.
- [x] Report `completion_rate`, `jobs_started`, and `jobs_unfinished`.
- [x] Include every started job in waiting-time summaries and explicitly count unfinished jobs in deadline misses.
- [x] Validate deferral and migration configuration.
- [x] Add tests for resource/latency feasibility, waiting/deadline metric consistency, migration pauses, invalid config and constrained-capacity outcomes.
- [x] Add deferral-cap sensitivity and six targeted workload/assumption scenarios.
- [x] Regenerate result CSVs, plots, report and dashboard.

## Documentation and final verification

- [x] Add a step-by-step algorithm/design document: `SCHEDULING_ALGORITHM.md`.
- [x] Document assumptions, interpretation and limitations in `ASSUMPTIONS_AND_LIMITATIONS.md` and `VALIDATION_STATUS.md`.
- [x] Compile check passed.
- [x] Full unit-test suite passed: **33 tests**.
- [x] Clean sample run completed from a fresh extraction after removing generated model/results.
- [x] Repeated seeded run produced identical hashes for the key comparison/experiment CSV files.
- [x] Tested missing real-data input handling, API downloader input validation, mocked API response parsing and Google-trace converter fixtures.

## Remaining external/manual checks

- [ ] Re-run the UK Carbon Intensity API download from an internet-connected machine; this environment's DNS lookup failed, so live connectivity is not claimed as verified.
- [ ] Check the BigQuery starter SQL against Google's official ClusterData2019 schema, run it in BigQuery, and test with a real exported trace CSV.
- [ ] If presenting results externally, rerun with real carbon data and clearly label the workload as synthetic unless a real Google trace extract is also supplied.

## Carbon-import / evaluation round (this iteration)

- [x] `src/carbon_import.py`: NESO CSV importer, UK-API fallback, cached-real reuse, labelled synthetic fallback, provenance JSON; `python main.py import-carbon`.
- [x] Carbon origin read from the CSV's `data_origin` column and shown in console, report and dashboard tag; unknown/fallback origins are flagged as not real.
- [x] Rolling-origin (walk-forward) forecast evaluation with leakage tests (`forecast_rolling_*.csv`).
- [x] Identical-workload guarantee: SHA-256 check inside `pipeline.simulate`, hashes in `scheduler_comparison.csv`, paired multi-seed comparison incl. energy-only ablation.
- [x] 22 new tests (55 total).
- [ ] Live NESO / UK-API download from an internet-connected machine (not verified).
- [ ] Real Google-trace workload (manual BigQuery step) - unchanged.
