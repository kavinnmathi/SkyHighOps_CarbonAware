# Results report - Carbon-Aware Edge Scheduling (Team SkyHighOps)

Generated: 2026-10-29 10:02   |   Data mode: **real**

## 1. What is real, predicted and simulated?

| Item | Status |
|---|---|
| Grid carbon intensity series | REAL_OBSERVATION (GB national series via neso, newest observation 2026-10-29 03:00:00+00:00) |
| Workload | SYNTHETIC_SAMPLE workload (real trace extract not provided) |
| Future carbon intensity used by the scheduler | **MODEL PREDICTION** (scikit-learn HistGradientBoostingRegressor) |
| Emissions accounting | uses the **actual** series in the data file (never the predictions) |
| Per-site carbon intensity | **SIMULATED** transform of the base series (scale / time shift per site, see below) |
| Node power, PUE, latency, migration overhead | **SIMULATED assumptions** from `config/default_config.json` |
| Migrations | **simulated bookkeeping only**; no physical cloud migration takes place |

Site transforms (assumptions):
- edge-site-A: ci_scale=1.0, ci_shift_steps=0
- edge-site-B: ci_scale=0.9, ci_shift_steps=-12
- cloud-region-1: ci_scale=0.8, ci_shift_steps=-24

## 2. Carbon-intensity forecaster (real supervised regression target)

Target: carbon intensity k half-hour steps ahead. Trained only on data before the simulation window
(2448 steps), evaluated on the held-out simulation window.

```
 horizon_steps  horizon_hours             method    mae   rmse     r2  n_test_points
             1            0.5  gradient_boosting  8.446 12.075  0.923            431
             1            0.5        persistence  6.773 11.371  0.932            431
             1            0.5 seasonal_naive_24h 37.188 48.331 -0.226            431
             4            2.0  gradient_boosting 18.999 24.557  0.685            428
             4            2.0        persistence 19.393 26.723  0.627            428
             4            2.0 seasonal_naive_24h 37.227 48.427 -0.224            428
            12            6.0  gradient_boosting 30.195 35.310  0.360            420
            12            6.0        persistence 32.069 42.015  0.094            420
            12            6.0 seasonal_naive_24h 37.286 48.646 -0.214            420
            48           24.0  gradient_boosting 48.460 56.461 -0.539            384
            48           24.0        persistence 38.245 50.004 -0.207            384
            48           24.0 seasonal_naive_24h 38.245 50.004 -0.207            384
```

The model has lower MAE than the best simple baseline at horizons (steps) [4, 8, 12, 24]; it does NOT beat the best baseline at [1, 2, 48]. This is reported as-is.

## 3. Scheduler comparison (simulation window: 7 days + 96 drain steps)

```
                         job_energy_kwh  job_emissions_g  fleet_emissions_g  migrations  mean_wait_min  p95_wait_min  deadline_miss_rate  latency_violations  job_emissions_reduction_pct  fleet_emissions_reduction_pct
scheduler                                                                                                                                                                                                               
baseline_fcfs_first_fit          59.221         6999.398          17796.809           0          0.691           0.0               0.021                   0                        0.000                          0.000
carbon_aware                     35.296         3528.293          15841.505           1         75.613         360.0               0.001                   0                       49.591                         10.987
energy_aware_ablation            35.037         3668.624          15917.875           0          0.000           0.0               0.000                   0                       47.587                         10.558
```

* `job_emissions_g`: emissions attributed to jobs (idle power shared by reserved CPU, dynamic power by used CPU) plus migration overhead.
* `fleet_emissions_g`: all nodes' power incl. idle (nodes are always on) plus migration overhead. Idle power is the same for every scheduler, so this metric shows smaller relative differences.
* Emissions = energy (kWh) x carbon intensity (gCO2/kWh).

Headline (sample/simulated results, not a claim about real clouds): the carbon-aware scheduler changed job emissions by
**49.6%** and fleet emissions by **11.0%** relative to the baseline,
with deadline-miss rate 0.001 (baseline 0.021) and 0 latency violations.

The energy-only ablation (`energy_aware_ablation`, same placement logic but blind to carbon) reduces job emissions by 47.6% vs. baseline. The *additional* effect of carbon-awareness (carbon-aware vs. energy-only) is 3.8%.

## 4. Robustness

Not run (only available in sample mode).

## 5. Deferral sensitivity (same workload, different wait caps)

```
 max_deferral_steps  max_deferral_hours  job_emissions_g  job_emissions_reduction_pct  fleet_emissions_g  fleet_emissions_reduction_pct  mean_wait_min  p95_wait_min  deadline_miss_rate  migrations
                  0                 0.0         3659.469                       47.717          15913.490                         10.582          0.000           0.0               0.001           1
                  6                 3.0         3605.103                       48.494          15883.304                         10.752         47.089         180.0               0.001           1
                 12                 6.0         3528.293                       49.591          15841.505                         10.987         75.613         360.0               0.001           1
                 18                 9.0         3487.097                       50.180          15819.797                         11.109         93.832         540.0               0.001           1
                 24                12.0         3461.561                       50.545          15805.615                         11.188        106.586         720.0               0.001           1
                 36                18.0         3426.140                       51.051          15784.740                         11.306        116.293         720.0               0.001           1
                 48                24.0         3405.451                       51.347          15772.841                         11.373        119.152         720.0               0.001           1
```
Lowest job emissions in this sweep occurred at a 24.0-hour deferral cap (51.3% lower than baseline; mean wait 119.2 min, p95 wait 720.0 min). This is a single synthetic workload sensitivity check, not a universal optimum.

## 6. Targeted stress scenarios

```
                   scenario  jobs_completed  jobs_unfinished  job_emissions_g  job_emissions_reduction_vs_scenario_baseline_pct  deadline_miss_rate  latency_violations  migrations
constant_carbon_equal_sites             955                0         4455.984                                            40.948               0.001                   0           1
   cleaner_cloud_assumption             955                0         2922.992                                            57.761               0.001                   0           1
        zero_slack_workload            1003                0         3593.358                                            47.940               0.002                   0           2
        high_migration_cost             955                0         3527.026                                            49.610               0.000                   0           0
      forecast_bias_plus_30             955                0         3525.437                                            49.632               0.001                   0           1
       resource_constrained             931               24        13910.022                                            12.566               0.176                   0          32
```
Interpretation: these are targeted stress tests using altered assumptions. In particular, cleaner-site and forecast-bias scenarios are sensitivity experiments, not claims about actual cloud regions or forecast error distributions.

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

## 9. Leakage-safe forecast evaluation (rolling origin)

Each fold trains a fresh model on data strictly before its test block (expanding window, no shuffling) and scores the next block. `skill_vs_persistence` > 0 means lower MAE than persistence. At 24 h (48 steps) the seasonal-naive baseline equals persistence by construction. Full detail: `forecast_rolling_folds.csv`, `forecast_rolling_summary.csv`.

```
 horizon_steps             method  mae_mean  mae_std  skill_vs_persistence  n_folds
             1  gradient_boosting     8.462    1.328                -0.218        4
             1 seasonal_naive_24h    38.320    8.022                -4.517        4
             4  gradient_boosting    18.958    3.300                 0.044        4
             4 seasonal_naive_24h    38.612    9.644                -0.948        4
            12  gradient_boosting    30.649    3.829                 0.045        4
            12 seasonal_naive_24h    39.547   12.095                -0.232        4
            24  gradient_boosting    37.976    7.672                 0.082        4
            24 seasonal_naive_24h    40.534   12.941                 0.020        4
            48  gradient_boosting    53.463    9.524                -0.239        4
            48 seasonal_naive_24h    43.135   16.149                 0.000        4
```

## 10. Paired scheduler comparison (identical workloads)

Within each seed all three schedulers run the same jobs (verified by a workload SHA-256), the same carbon series and the same forecaster. Workloads are SYNTHETIC; carbon data origin: REAL_OBSERVATION (GB national series via neso, newest observation 2026-10-29 03:00:00+00:00).

```
              scheduler  n_seeds  job_reduction_mean_pct  job_reduction_std_pct  fleet_reduction_mean_pct  fleet_reduction_std_pct  mean_wait_min  deadline_miss_rate  completion_rate  seeds_better
baseline_fcfs_first_fit        5                   0.000                  0.000                     0.000                    0.000          0.532               0.012              1.0             0
           carbon_aware        5                  49.096                  1.258                    11.153                    0.585         78.021               0.002              1.0             5
  energy_aware_ablation        5                  47.456                  1.198                    10.786                    0.570          0.000               0.000              1.0             5
```


Carbon provenance: source `neso`, origin `REAL_OBSERVATION`, newest observation 2026-10-29 03:00:00+00:00, retrieved 2026-10-29T04:28:43+00:00 (`carbon_provenance.json`).
