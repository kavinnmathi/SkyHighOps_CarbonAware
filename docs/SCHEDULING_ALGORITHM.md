# Scheduling algorithm and design decisions

This document describes the implementation in `src/schedulers.py` and `src/simulator.py`. It is an explanation of this project's heuristic simulation, **not** a claim that the code reproduces a production cloud scheduler or the mixed-integer optimiser in the reference paper.

## 1. Shared simulation model

- Time advances in 30-minute steps. Jobs arrive at their configured `arrival_step` and request CPU cores, memory, a maximum acceptable node latency, duration, and optional waiting slack.
- A placement is legal only if the node has enough unreserved CPU and memory and its configured latency is no greater than the job's latency requirement. `Simulator.start_job` and `Simulator.migrate_job` assert this before changing reservations.
- Job execution and node energy are simulated. The model does not run user workloads or move real cloud state.
- Scheduling decisions use the forecaster's predicted carbon intensity. Energy/emissions reporting uses the actual carbon-intensity series selected for the run.

## 2. Baseline: FCFS + first-fit

1. Visit pending jobs in their existing pending-list order (the simulator adds arrivals in time order).
2. For each job, scan nodes in configuration order.
3. Start it on the first node that passes CPU, memory and latency checks.
4. If no node can fit it at that step, leave it pending for a later step.
5. Do not intentionally defer a feasible job for a greener period and do not migrate jobs.

**Design choice:** this is a simple conventional reference point. Node order matters; the supplied configuration lists edge nodes before cloud nodes.

## 3. Carbon-aware scheduler

At each step:

1. Consider migration opportunities first when migration is enabled.
2. Order pending jobs by earliest configured latest-start step (`arrival_step + slack_steps`), then arrival time.
3. Build the set of nodes that are feasible **now**. If the set is empty, leave the job pending.
4. Estimate each feasible node's job energy from the simplified power model and multiply it by predicted site carbon intensity over the job's expected duration. Choose the lowest predicted cost as the current best placement.
5. If the job has slack, compare its current predicted cost against predicted costs for future start offsets, bounded by both the job's remaining slack and the forecaster horizon. The candidate nodes are those feasible now; the code does not know future contention exactly.
6. Defer only when the best future cost improves on the current cost by the configured `defer_margin`, and the job has not exceeded `max_deferral_steps` (default 12 steps = 6 hours since arrival). Otherwise start on the current best node.
7. The simulator reserves requested CPU/memory on placement. It checks the hard latency constraint and releases reservations at completion.

**Cost function:**

- Carbon-aware: estimated job kWh × predicted carbon intensity summed over the job's expected execution window.
- Energy-only ablation: estimated job kWh only. It disables carbon-based deferral and migration, providing a comparison for the hardware-efficiency contribution.

## 4. Migration heuristic

1. Skip a job if it has reached `max_migrations_per_job` or has less than `min_remaining_steps` work left.
2. Estimate the cost of staying on the current node versus moving to each different node that can fit the job now.
3. Add estimated transfer energy/emissions and downtime overhead to the destination cost.
4. Migrate only when the predicted benefit is positive and exceeds `benefit_margin` times the stay cost.
5. Record the migration in the log and charge transfer energy/emissions separately. During migration downtime, execution progress pauses; fractional time-step pauses are supported. The job's CPU/memory reservation remains on its destination in this simplified model.

Migration is bookkeeping in the simulation. It is not a real VM/container migration, and the model does not reproduce network contention or all costs of live migration.

## 5. Metrics and why there are two emissions totals

- **Job-attributed emissions:** emissions assigned to jobs using active power plus a share of idle power based on reserved CPU. Migration transfer emissions are included in each job's total.
- **Fleet emissions:** energy and emissions of all configured nodes, including their idle power, plus migration transfer overhead. Nodes are treated as always on, so fleet savings are usually smaller than job-attributed savings.
- **Waiting:** reported for jobs that started; unfinished jobs are separately reported and are not silently marked complete.
- **Deadline miss rate:** a job meets its deadline only if it completed by `arrival + slack + duration`. Unfinished jobs count as deadline misses.
- **Latency violations:** placement on a node whose configured latency exceeds the job requirement. This should remain zero because latency is a hard placement constraint.
- **Completion rate / unfinished jobs:** makes capacity-limited experiments visible instead of comparing emissions while hiding lost work.

## 6. Experiments and validation

- `deferral_sensitivity.csv` varies the maximum deferral cap to show the emissions-versus-waiting trade-off.
- `scenario_results.csv` changes assumptions such as equal carbon across sites, a cleaner cloud, zero slack, high migration cost, biased forecasts, and constrained capacity.
- `robustness.csv` repeats comparisons on seeded synthetic workloads.
- Unit tests cover configuration validation, carbon-data gaps, Google-trace CSV mapping, forecast leakage checks, resource/latency feasibility, migration pause behavior, metric consistency, deterministic results, and stress-scenario outputs.

## 7. Limitations that affect interpretation

1. The default workload and carbon series are synthetic. Their results demonstrate that the pipeline runs, not real-world carbon savings.
2. Site carbon intensities are transformations of one base national series, not measured regional data.
3. Node power, PUE, latency, transfer bandwidth/energy, and some job attributes are assumptions.
4. The scheduler knows each job's duration and resource request exactly; a production system would often need estimates.
5. Future resource availability is approximated by current feasibility during deferral planning.
6. The forecast model beats the best simple baseline at some horizons but not necessarily all horizons; inspect `forecaster_metrics.csv` rather than assuming it always wins.
7. The Google trace conversion requires a manually exported BigQuery CSV, and the SQL schema must be checked against official documentation before use.
8. The carbon API downloader is covered by input-validation tests but requires a separate internet-enabled run to verify live API connectivity and current response behavior.
