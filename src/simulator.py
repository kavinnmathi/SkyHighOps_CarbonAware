"""Discrete-time edge/cloud simulator (30-minute steps).

Everything here is a SIMULATION: jobs are not executed and nothing is physically
migrated. Carbon intensities used for accounting are the ACTUAL series of the
chosen data mode; scheduler decisions only see model PREDICTIONS of the future.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .energy import emissions_g, energy_kwh, job_power_w, migration_cost, node_power_w
from .models import JobRun, JobSpec, NodeSpec, NodeState


@dataclass
class RunResult:
    scheduler: str
    jobs: pd.DataFrame
    timeseries: pd.DataFrame
    migrations: pd.DataFrame
    summary: dict


class Simulator:
    def __init__(self, cfg: dict, node_specs: list[NodeSpec], jobs: list[JobSpec], view, scheduler,
                 sim_steps: int, total_steps: int):
        self.cfg = cfg
        self.view = view
        self.scheduler = scheduler
        self.sim_steps, self.total_steps = sim_steps, total_steps
        self.step_h = cfg["time_step_minutes"] / 60.0
        self.step_s = cfg["time_step_minutes"] * 60.0
        self.nodes = [NodeState(s) for s in node_specs]
        self.cluster = {n.spec.node_id: n for n in self.nodes}
        self.runs = [JobRun(j) for j in jobs]
        self._arrivals: dict[int, list[JobRun]] = {}
        for r in self.runs:
            self._arrivals.setdefault(r.spec.arrival_step, []).append(r)
        self.pending: list[JobRun] = []
        self.running: list[JobRun] = []
        self.migration_log: list[dict] = []
        self.mig_energy_total = 0.0
        self.mig_emissions_total = 0.0

    # ---- actions used by schedulers
    def start_job(self, run: JobRun, node: NodeState, t: int) -> None:
        assert node.can_fit(run.spec), "scheduler tried an infeasible placement"
        node.cpu_reserved += run.spec.cpu_req
        node.mem_reserved += run.spec.mem_req_gb
        node.jobs.append(run)
        run.node_id, run.start_step = node.spec.node_id, t
        self.pending.remove(run)
        self.running.append(run)

    def migrate_job(self, run: JobRun, dst: NodeState, t: int) -> None:
        assert dst.can_fit(run.spec), "scheduler tried an infeasible migration"
        src = self.cluster[run.node_id]
        src.cpu_reserved -= run.spec.cpu_req
        src.mem_reserved -= run.spec.mem_req_gb
        src.jobs.remove(run)
        dst.cpu_reserved += run.spec.cpu_req
        dst.mem_reserved += run.spec.mem_req_gb
        dst.jobs.append(run)
        mc = migration_cost(run.spec, self.cfg["migration"], self.step_s)
        ci_mid = 0.5 * (self.view.actual(src.spec.site_id, t) + self.view.actual(dst.spec.site_id, t))
        em = emissions_g(mc.energy_kwh, ci_mid)           # accounted with ACTUAL carbon intensity
        # Model downtime as wall-clock pause, not extra work. The old implementation
        # increased remaining work and still advanced the job in the same timestep,
        # effectively allowing execution during the migration pause.
        run.paused_steps_remaining += mc.extra_work_steps
        run.migrations += 1
        run.migration_energy_kwh += mc.energy_kwh
        run.migration_emissions_g += em
        self.mig_energy_total += mc.energy_kwh
        self.mig_emissions_total += em
        self.migration_log.append({"scheduler": self.scheduler.name, "step": t, "job_id": run.spec.job_id,
                                   "from_node": src.spec.node_id, "to_node": dst.spec.node_id,
                                   "downtime_s": mc.downtime_seconds, "transfer_energy_kwh": mc.energy_kwh,
                                   "transfer_emissions_g": em})
        run.node_id = dst.spec.node_id

    # ---- main loop
    def run(self) -> RunResult:
        ts_rows = []
        fleet_kwh = fleet_g = 0.0
        for t in range(self.total_steps):
            if t < self.sim_steps:
                self.pending.extend(self._arrivals.get(t, []))
            self.scheduler.step(self, t)
            used = {"edge": 0.0, "cloud": 0.0}
            cap = {"edge": 0.0, "cloud": 0.0}
            step_kwh = step_g = 0.0
            active_fraction_by_run = {}
            for node in self.nodes:
                s = node.spec
                # A migrating job remains reserved on the destination but does not consume
                # execution CPU during its downtime. Fractional downtime is supported.
                active_fraction_by_run = {
                    id(r): max(0.0, 1.0 - min(1.0, r.paused_steps_remaining))
                    for r in node.jobs
                }
                used_cpu = sum(r.spec.cpu_req * r.spec.cpu_util * active_fraction_by_run[id(r)]
                               for r in node.jobs)
                ci = self.view.actual(s.site_id, t)
                e = energy_kwh(node_power_w(s, used_cpu), s.pue, self.step_h)
                step_kwh += e
                step_g += emissions_g(e, ci)
                used[s.tier] += used_cpu
                cap[s.tier] += s.cpu_cores
                for r in node.jobs:  # attribute active power + reserved-idle share during downtime
                    active_fraction = active_fraction_by_run[id(r)]
                    idle_share_w = s.idle_power_w * r.spec.cpu_req / s.cpu_cores
                    active_power_w = job_power_w(r.spec, s)
                    attributed_power_w = (active_power_w * active_fraction
                                          + idle_share_w * (1.0 - active_fraction))
                    je = energy_kwh(attributed_power_w, s.pue, self.step_h)
                    r.energy_kwh += je
                    r.emissions_g += emissions_g(je, ci)
            fleet_kwh += step_kwh
            fleet_g += step_g
            ts_rows.append({"step": t, "fleet_energy_kwh": step_kwh, "fleet_emissions_g": step_g,
                            "running_jobs": len(self.running), "pending_jobs": len(self.pending),
                            "edge_cpu_utilisation": used["edge"] / cap["edge"] if cap["edge"] else np.nan,
                            "cloud_cpu_utilisation": used["cloud"] / cap["cloud"] if cap["cloud"] else np.nan})
            for r in list(self.running):  # progress only while not paused by migration
                active_fraction = max(0.0, 1.0 - min(1.0, r.paused_steps_remaining))
                r.remaining -= active_fraction
                r.paused_steps_remaining = max(0.0, r.paused_steps_remaining - 1.0)
                if r.remaining <= 1e-9:
                    r.finish_step = t + 1
                    n = self.cluster[r.node_id]
                    n.cpu_reserved -= r.spec.cpu_req
                    n.mem_reserved -= r.spec.mem_req_gb
                    if r not in n.jobs:
                        raise RuntimeError(f"completion bookkeeping mismatch: job={r.spec.job_id}, node_id={r.node_id}, node_jobs={[x.spec.job_id for x in n.jobs]}, node_ids={[x.spec.node_id for x in self.nodes]}")
                    n.jobs.remove(r)
                    self.running.remove(r)
        return self._collect(pd.DataFrame(ts_rows), fleet_kwh, fleet_g)

    def _collect(self, ts: pd.DataFrame, fleet_kwh: float, fleet_g: float) -> RunResult:
        rows = []
        for r in self.runs:
            sp = r.spec
            done = r.finish_step is not None
            started = r.start_step is not None
            node_lat = self.cluster[r.node_id].spec.latency_ms if r.node_id else math.nan
            deadline = sp.arrival_step + sp.slack_steps + sp.duration_steps
            rows.append({
                "scheduler": self.scheduler.name, "job_id": sp.job_id, "arrival_step": sp.arrival_step,
                "duration_steps": sp.duration_steps, "cpu_req": sp.cpu_req, "mem_req_gb": sp.mem_req_gb,
                "latency_req_ms": sp.latency_req_ms, "slack_steps": sp.slack_steps,
                "start_step": r.start_step, "finish_step": r.finish_step, "final_node": r.node_id,
                "wait_steps": (r.start_step - sp.arrival_step) if started else np.nan,
                "completed": done, "deadline_met": bool(done and r.finish_step <= deadline),
                "latency_violation": bool(started and node_lat > sp.latency_req_ms),
                "migrations": r.migrations, "run_energy_kwh": r.energy_kwh,
                "run_emissions_g": r.emissions_g, "migration_energy_kwh": r.migration_energy_kwh,
                "migration_emissions_g": r.migration_emissions_g,
                "total_emissions_g": r.emissions_g + r.migration_emissions_g})
        jobs = pd.DataFrame(rows)
        done = jobs[jobs.completed]
        job_kwh = jobs.run_energy_kwh.sum() + jobs.migration_energy_kwh.sum()
        job_g = jobs.total_emissions_g.sum()
        summary = {
            "scheduler": self.scheduler.name,
            "jobs_total": len(jobs), "jobs_completed": int(jobs.completed.sum()),
            "completion_rate": float(jobs.completed.mean()) if len(jobs) else 1.0,
            "jobs_started": int(jobs.start_step.notna().sum()),
            "jobs_unfinished": int((~jobs.completed).sum()),
            "job_energy_kwh": job_kwh, "job_emissions_g": job_g,
            "job_avg_carbon_intensity_g_per_kwh": job_g / job_kwh if job_kwh else np.nan,
            "migration_energy_kwh": self.mig_energy_total, "migration_emissions_g": self.mig_emissions_total,
            "fleet_energy_kwh": fleet_kwh + self.mig_energy_total,
            "fleet_emissions_g": fleet_g + self.mig_emissions_total,
            "migrations": int(jobs.migrations.sum()),
            "mean_wait_min": float(jobs.loc[jobs.start_step.notna(), "wait_steps"].mean()
                                   * self.cfg["time_step_minutes"]) if jobs.start_step.notna().any() else np.nan,
            "p95_wait_min": float(jobs.loc[jobs.start_step.notna(), "wait_steps"].quantile(0.95)
                                  * self.cfg["time_step_minutes"]) if jobs.start_step.notna().any() else np.nan,
            "deadline_miss_rate": float(1.0 - jobs.deadline_met.mean()),
            "latency_violations": int(jobs.latency_violation.sum()),
            "mean_edge_cpu_util": float(ts.edge_cpu_utilisation.mean()),
            "mean_cloud_cpu_util": float(ts.cloud_cpu_utilisation.mean()),
        }
        ts.insert(0, "scheduler", self.scheduler.name)
        return RunResult(self.scheduler.name, jobs, ts, pd.DataFrame(self.migration_log), summary)
