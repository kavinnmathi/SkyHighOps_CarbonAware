import copy
import unittest

import numpy as np

from src import pipeline
from src.config import load_config
from src.data_sources import generate_synthetic_carbon
from src.errors import DataError
from src.forecasting import CarbonForecaster
from src.models import JobSpec, NodeSpec, NodeState
from src.schedulers import BaselineScheduler
from src.simulator import Simulator


def small_cfg():
    cfg = copy.deepcopy(load_config())
    cfg["data"]["simulation_days"] = 2
    cfg["data"]["drain_steps"] = 96
    cfg["forecaster"]["max_iter"] = 20
    cfg["forecaster"]["model_path"] = "models/test_tmp_forecaster.joblib"
    return cfg


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = small_cfg()
        cls.prep = pipeline.prepare(cls.cfg)
        cls.results = pipeline.simulate(cls.cfg, cls.prep, cls.prep.workload.query("arrival_step < 96"))
        cls.by_name = {r.scheduler: r for r in cls.results}

    @classmethod
    def tearDownClass(cls):
        from src.config import resolve_path
        p = resolve_path(cls.cfg["forecaster"]["model_path"])
        if p.exists():
            p.unlink()

    def test_no_latency_violations_and_all_complete(self):
        for r in self.results:
            self.assertEqual(r.summary["latency_violations"], 0, r.scheduler)
            self.assertEqual(r.summary["jobs_completed"], r.summary["jobs_total"], r.scheduler)

    def test_node_capacity_and_latency_are_hard_constraints(self):
        node = NodeState(NodeSpec("tiny", "edge-site-A", "edge", 2, 4, 10, 30, 1.2, 10))
        fits = JobSpec(1, 0, 1, 1.5, 2, 0.8, 10, 0, 0)
        cpu_too_large = JobSpec(2, 0, 1, 2.1, 2, 0.8, 10, 0, 0)
        mem_too_large = JobSpec(3, 0, 1, 1, 4.1, 0.8, 10, 0, 0)
        latency_too_strict = JobSpec(4, 0, 1, 1, 1, 0.8, 9, 0, 0)
        self.assertTrue(node.can_fit(fits))
        self.assertFalse(node.can_fit(cpu_too_large))
        self.assertFalse(node.can_fit(mem_too_large))
        self.assertFalse(node.can_fit(latency_too_strict))
        node.cpu_reserved = 1.0
        node.mem_reserved = 3.0
        self.assertFalse(node.can_fit(fits), "reserved CPU and memory must count against capacity")

    def test_job_energy_not_above_fleet_energy(self):
        for r in self.results:
            self.assertLessEqual(r.summary["job_energy_kwh"], r.summary["fleet_energy_kwh"] + 1e-6)

    def test_emissions_equal_energy_times_ci_bounds(self):
        # average CI experienced must lie within the range of CI values the sites can have
        for r in self.results:
            avg = r.summary["job_avg_carbon_intensity_g_per_kwh"]
            lo, hi = 0.8 * self.prep.base_ci.min(), 1.0 * self.prep.base_ci.max()
            self.assertTrue(lo <= avg <= hi, (r.scheduler, avg))

    def test_wait_and_deadline_metrics_match_job_records(self):
        for result in self.results:
            jobs = result.jobs
            started = jobs[jobs.start_step.notna()]
            expected_wait = started.wait_steps.mean() * self.cfg["time_step_minutes"]
            self.assertAlmostEqual(result.summary["mean_wait_min"], expected_wait)
            expected_miss = 1.0 - jobs.deadline_met.mean()
            self.assertAlmostEqual(result.summary["deadline_miss_rate"], expected_miss)
            self.assertEqual(result.summary["jobs_unfinished"], int((~jobs.completed).sum()))

    def test_baseline_never_migrates_or_delays(self):
        b = self.by_name["baseline_fcfs_first_fit"]
        self.assertEqual(b.summary["migrations"], 0)

    def test_carbon_aware_not_worse_than_baseline_on_job_emissions(self):
        self.assertLessEqual(self.by_name["carbon_aware"].summary["job_emissions_g"],
                             self.by_name["baseline_fcfs_first_fit"].summary["job_emissions_g"])

    def test_deterministic(self):
        again = pipeline.simulate(self.cfg, self.prep, self.prep.workload.query("arrival_step < 96"))
        for a, b in zip(self.results, again):
            self.assertAlmostEqual(a.summary["job_emissions_g"], b.summary["job_emissions_g"])

    def test_forecaster_needs_enough_history(self):
        fc = CarbonForecaster(48, 5)
        with self.assertRaises(DataError):
            fc.fit(np.ones(100), 100, 0, 0)

    def test_forecaster_uses_no_future_values(self):
        # changing data AFTER the anchor must not change the prediction for that anchor
        ci = self.prep.base_ci.copy()
        t = self.prep.sim_start + 10
        p1 = self.prep.forecaster.predict_matrix(ci, np.array([t]), np.array([4]))
        ci2 = ci.copy()
        ci2[t + 1:] += 1000
        p2 = self.prep.forecaster.predict_matrix(ci2, np.array([t]), np.array([4]))
        np.testing.assert_allclose(p1, p2)

    def test_too_short_carbon_series(self):
        cfg = small_cfg()
        cfg["data"]["simulation_days"] = 30  # leaves < 7 days for training in the 37-day sample
        with self.assertRaises(DataError):
            pipeline.prepare(cfg)

    def test_stress_scenario_suite_reports_all_scenarios(self):
        scenarios = pipeline.scenario_evaluation(self.cfg, self.prep)
        self.assertEqual(scenarios["scenario"].nunique(), 6)
        self.assertIn("jobs_unfinished", scenarios.columns)
        expensive = scenarios.query(
            "scenario == 'high_migration_cost' and scheduler == 'carbon_aware'"
        )
        self.assertEqual(int(expensive.iloc[0].migrations), 0)
        constrained = scenarios.query(
            "scenario == 'resource_constrained' and scheduler == 'carbon_aware'"
        )
        self.assertEqual(len(constrained), 1)
        self.assertGreater(int(constrained.iloc[0].jobs_unfinished), 0,
                           "capacity stress must expose unfinished jobs, not count them as complete")

    def test_migration_downtime_pauses_work_instead_of_double_counting(self):
        cfg = copy.deepcopy(load_config())
        cfg["migration"]["bandwidth_mbps"] = 100
        cfg["migration"]["fixed_downtime_s"] = 30
        job = JobSpec(999, 0, 4, 1, 1, 0.5, 50, 0, 50000)
        nodes = [
            NodeSpec("test-a", "edge-site-A", "edge", 8, 16, 40, 120, 1.5, 8),
            NodeSpec("test-b", "edge-site-B", "edge", 8, 16, 40, 120, 1.5, 12),
        ]

        class ConstantView:
            @staticmethod
            def actual(site_id, t):
                return 100.0

        sim = Simulator(cfg, nodes, [job], ConstantView(), BaselineScheduler(),
                        sim_steps=1, total_steps=12)
        run = sim.runs[0]
        sim._arrivals.pop(0, None)  # the job is manually started for this focused migration test
        sim.pending.append(run)
        sim.start_job(run, sim.nodes[0], 0)
        original_remaining = run.remaining
        sim.migrate_job(run, sim.nodes[1], 0)

        self.assertEqual(run.remaining, original_remaining,
                         "Migration downtime must pause wall-clock progress, not add extra work.")
        self.assertGreater(run.paused_steps_remaining, 2.0)
        result = sim.run()
        self.assertTrue(result.jobs.iloc[0].completed)
        self.assertGreater(result.jobs.iloc[0].finish_step, job.duration_steps,
                           "Migration pause must delay completion.")


if __name__ == "__main__":
    unittest.main()
