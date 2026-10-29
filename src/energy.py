"""Energy, carbon and migration-overhead models.

Server power (linear model, a common simplification):
    P_node = P_idle + (P_max - P_idle) * utilisation            [W]
Facility energy over one step:
    E_kWh = P_node * PUE * step_hours / 1000
Carbon (the relationship required by the project):
    CO2 [g] = E [kWh] * carbon_intensity [gCO2/kWh]

Per-job attribution (so that schedulers can be compared job by job):
    idle power is shared by the CPU the job RESERVES, dynamic power by the CPU it USES.
All wattages are ASSUMED values from the config, not measurements.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import JobSpec, NodeSpec


def node_power_w(spec: NodeSpec, used_cpu: float) -> float:
    util = min(max(used_cpu / spec.cpu_cores, 0.0), 1.0)
    return spec.idle_power_w + (spec.max_power_w - spec.idle_power_w) * util


def job_power_w(job: JobSpec, spec: NodeSpec) -> float:
    """IT power attributed to a job on a node (before PUE)."""
    idle_share = spec.idle_power_w * job.cpu_req / spec.cpu_cores
    dyn_share = (spec.max_power_w - spec.idle_power_w) * job.cpu_req * job.cpu_util / spec.cpu_cores
    return idle_share + dyn_share


def energy_kwh(power_w: float, pue: float, step_hours: float) -> float:
    return power_w * pue * step_hours / 1000.0


def emissions_g(kwh: float, ci_g_per_kwh: float) -> float:
    """CO2 emissions in grams = kWh x gCO2/kWh."""
    return kwh * ci_g_per_kwh


@dataclass(frozen=True)
class MigrationCost:
    transfer_seconds: float
    downtime_seconds: float
    extra_work_steps: float   # job loses this much progress (it is paused while moving)
    energy_kwh: float         # network/transfer energy (ASSUMED kWh per GB)


def migration_cost(job: JobSpec, mcfg: dict, step_seconds: float) -> MigrationCost:
    """SIMULATED overhead of moving a job's state. No real migration is performed."""
    transfer_s = job.data_mb * 8.0 / mcfg["bandwidth_mbps"]
    downtime_s = transfer_s + mcfg["fixed_downtime_s"]
    return MigrationCost(transfer_s, downtime_s, downtime_s / step_seconds,
                         job.data_mb / 1024.0 * mcfg["transfer_kwh_per_gb"])
