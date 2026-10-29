"""Schedulers.

BaselineScheduler    : conventional FCFS + first-fit (edge nodes listed first). It ignores
                       carbon intensity, never delays a job and never migrates.
CarbonAwareScheduler : respects the same CPU / memory / latency constraints, but
                       (1) places each job on the node with the lowest PREDICTED emissions,
                       (2) may delay delay-tolerant jobs (slack) to a predicted greener time,
                       (3) may migrate a running job if predicted savings exceed the
                           migration overhead by a margin.
With use_carbon=False it becomes an energy-only ablation (cost = kWh, no deferral/migration),
which shows how much of the gain comes from efficient hardware rather than carbon-awareness.
"""
from __future__ import annotations

from .energy import energy_kwh, job_power_w, migration_cost


class Scheduler:
    name = "base"

    def step(self, sim, t: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class BaselineScheduler(Scheduler):
    name = "baseline_fcfs_first_fit"

    def step(self, sim, t: int) -> None:
        for run in list(sim.pending):
            for node in sim.nodes:
                if node.can_fit(run.spec):
                    sim.start_job(run, node, t)
                    break


class CarbonAwareScheduler(Scheduler):
    def __init__(self, cfg: dict, use_carbon: bool = True, name: str | None = None):
        self.cfg = cfg
        self.use_carbon = use_carbon
        self.defer_margin = cfg["carbon_aware_scheduler"]["defer_margin"]
        self.max_deferral_steps = int(cfg["carbon_aware_scheduler"].get("max_deferral_steps", 12))
        self.enable_deferral = use_carbon and cfg["carbon_aware_scheduler"]["enable_deferral"]
        self.enable_migration = use_carbon and cfg["carbon_aware_scheduler"]["enable_migration"]
        self.mcfg = cfg["migration"]
        self.step_h = cfg["time_step_minutes"] / 60.0
        self.step_s = cfg["time_step_minutes"] * 60.0
        self.H = cfg["forecaster"]["max_horizon_steps"]
        self.name = name or ("carbon_aware" if use_carbon else "energy_aware_ablation")
        self._cache: dict = {}

    # predicted emissions [g] (or energy [kWh] if not use_carbon) of running `length` steps
    def _cost(self, sim, node, job, t, start, length) -> float:
        kwh_per_ci_step = energy_kwh(job_power_w(job, node.spec), node.spec.pue, self.step_h)
        if not self.use_carbon:
            return kwh_per_ci_step * length
        return kwh_per_ci_step * sim.view.predicted_sum(node.spec.site_id, t, start, length)

    def _future_costs(self, sim, node, job, t, max_start):
        """Predicted emissions for every start offset 0..max_start (vector)."""
        key = (node.spec.site_id, t, job.duration_steps, max_start)
        if key not in self._cache:
            self._cache[key] = sim.view.sums_over_starts(node.spec.site_id, t, max_start, job.duration_steps)
        kwh = energy_kwh(job_power_w(job, node.spec), node.spec.pue, self.step_h)
        return kwh * self._cache[key]

    def step(self, sim, t: int) -> None:
        self._cache = {}
        if self.enable_migration:
            self._migrate(sim, t)
        order = sorted(sim.pending, key=lambda r: (r.spec.arrival_step + r.spec.slack_steps, r.spec.arrival_step))
        for run in order:
            job = run.spec
            feasible_now = [n for n in sim.nodes if n.can_fit(job)]
            if not feasible_now:
                continue
            costs_now = [(self._cost(sim, n, job, t, 0, job.duration_steps), n) for n in feasible_now]
            best_now, best_node = min(costs_now, key=lambda x: x[0])
            slack_left = job.arrival_step + job.slack_steps - t
            if self.enable_deferral and slack_left > 0:
                max_start = min(slack_left, self.H)
                # Compare future windows only on nodes that are feasible NOW. Considering
                # statically feasible but currently overloaded nodes can make the scheduler
                # defer for a destination that it cannot actually use.
                waited = t - job.arrival_step
                can_defer_more = waited < self.max_deferral_steps
                if can_defer_more:
                    future_candidates = [
                        float(self._future_costs(sim, n, job, t, max_start)[1:].min())
                        for n in feasible_now
                    ]
                    best_future = min(future_candidates) if future_candidates else best_now
                    if best_future < best_now * (1.0 - self.defer_margin):
                        continue  # wait for a predicted cleaner window, within the wait cap
            sim.start_job(run, best_node, t)

    def _migrate(self, sim, t: int) -> None:
        cands = []
        for run in sim.running:
            if run.migrations >= self.mcfg["max_migrations_per_job"] or run.remaining < self.mcfg["min_remaining_steps"]:
                continue
            job, cur = run.spec, sim.cluster[run.node_id]
            stay = self._cost(sim, cur, job, t, 0, run.remaining)
            best = None
            for dst in sim.nodes:
                if dst is cur or not dst.can_fit(job):
                    continue
                mc = migration_cost(job, self.mcfg, self.step_s)
                ci_mid = 0.5 * (sim.view.actual(cur.spec.site_id, t) + sim.view.actual(dst.spec.site_id, t))
                go = (self._cost(sim, dst, job, t, 0, run.remaining + mc.extra_work_steps)
                      + mc.energy_kwh * ci_mid)
                if best is None or go < best[0]:
                    best = (go, dst)
            if best and stay - best[0] > self.mcfg["benefit_margin"] * stay and stay - best[0] > 0:
                cands.append((stay - best[0], run, best[1]))
        for _, run, dst in sorted(cands, key=lambda x: -x[0]):
            if dst.can_fit(run.spec):  # capacity may have changed after earlier moves
                sim.migrate_job(run, dst, t)
