"""Plain data containers used by the simulator."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SiteSpec:
    site_id: str
    ci_scale: float       # SIMULATED assumption: multiplies the base grid series
    ci_shift_steps: int   # SIMULATED assumption: time offset of this site's grid pattern


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    site_id: str
    tier: str             # "edge" or "cloud"
    cpu_cores: float
    mem_gb: float
    idle_power_w: float   # ASSUMED, not measured
    max_power_w: float    # ASSUMED, not measured
    pue: float
    latency_ms: float     # ASSUMED user-to-node latency


@dataclass(frozen=True)
class JobSpec:
    job_id: int
    arrival_step: int
    duration_steps: int
    cpu_req: float
    mem_req_gb: float
    cpu_util: float       # fraction of cpu_req actually used (drives dynamic power)
    latency_req_ms: float # job needs node latency <= this
    slack_steps: int      # how many steps the job may be delayed (0 = not delay tolerant)
    data_mb: float        # state size moved on migration (SIMULATED)


@dataclass
class JobRun:
    """Mutable per-run state of a job."""
    spec: JobSpec
    remaining: float = 0.0
    node_id: str | None = None
    start_step: int | None = None
    finish_step: int | None = None
    migrations: int = 0
    energy_kwh: float = 0.0
    emissions_g: float = 0.0
    migration_energy_kwh: float = 0.0
    migration_emissions_g: float = 0.0
    paused_steps_remaining: float = 0.0  # fractional simulation steps left in migration downtime
    latency_violation: bool = False

    def __post_init__(self):
        self.remaining = float(self.spec.duration_steps)


@dataclass
class NodeState:
    spec: NodeSpec
    cpu_reserved: float = 0.0
    mem_reserved: float = 0.0
    jobs: list = field(default_factory=list)

    def can_fit(self, job: JobSpec) -> bool:
        s = self.spec
        return (self.cpu_reserved + job.cpu_req <= s.cpu_cores + 1e-9
                and self.mem_reserved + job.mem_req_gb <= s.mem_gb + 1e-9
                and s.latency_ms <= job.latency_req_ms)

    def statically_feasible(self, job: JobSpec) -> bool:
        """Would the job fit if the node were empty (used when planning deferral)?"""
        s = self.spec
        return (job.cpu_req <= s.cpu_cores and job.mem_req_gb <= s.mem_gb
                and s.latency_ms <= job.latency_req_ms)
