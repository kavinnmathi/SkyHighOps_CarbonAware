# Assumptions and limitations

## Model
- **Time step:** 30 minutes (matches the National Grid series).
- **Server power:** `P = P_idle + (P_max - P_idle) x CPU utilisation` (linear model). Facility energy = `P x PUE x hours / 1000` kWh.
- **Emissions:** `gCO2 = kWh x gCO2/kWh`, always computed with the **actual** carbon-intensity series.
- **Job attribution:** idle power is shared by reserved CPU share, dynamic power by used CPU. `job_*` metrics use this; `fleet_*` metrics include all idle power (all nodes are always on) and therefore show smaller relative differences.
- **Migration (simulated):** time = `data_MB x 8 / bandwidth + fixed downtime`; the job pauses for that duration (fractional 30-minute steps are supported), and does not execute during the paused portion. Transfer energy = `GB x kWh_per_GB`, charged at the mean of source and destination actual intensity. No real migration exists.
- **Deferral guardrail:** in addition to each job's configured slack, scheduler deferral is capped by `carbon_aware_scheduler.max_deferral_steps` (default 12 steps = 6 hours). Future placement feasibility is approximated using nodes feasible at the current step; future contention is not known exactly.
- **Latency:** each node has a fixed assumed latency; a job may only run where `node latency <= job requirement` (hard constraint, verified by the `latency_violations` metric).

## Assumed (not measured) values
Node power ratings, PUE, node latency, bandwidth, kWh per GB, site `ci_scale` / `ci_shift_steps`, job `cpu_util`, `slack_steps`, `data_mb`, and in sample mode the whole workload and carbon series. All are in `config/default_config.json` and can be changed.

## Limitations
1. Sample-mode numbers come from synthetic data and show that the pipeline works, not real savings.
2. The headline savings are strongly influenced by the assumed hardware efficiency gap (cloud nodes use fewer watts per core). The **energy-aware ablation** row separates this out: the extra gain of carbon-awareness is the difference between `carbon_aware` and `energy_aware_ablation`.
3. Per-site carbon intensity is a transformation of one national series.
4. The scheduler knows each job's duration and CPU use exactly. Real systems must estimate them.
5. Migrations are rare with the default settings because the overhead model and margin are conservative; tune `migration` in the config to explore.
6. One simulated week per run, plus a 5-seed robustness check on synthetic workloads in sample mode.
7. The carbon-aware scheduler is a heuristic, not the paper's mixed-integer optimisation, so it is not a replication.
8. The forecaster is evaluated on one held-out period; with sample data it learns synthetic patterns, which proves nothing about the real grid. Run real mode to see real forecast errors (the model may then not beat the baselines at every horizon, and the report will say so).
9. Deferring jobs lowers emissions but increases waiting time; both are reported.
