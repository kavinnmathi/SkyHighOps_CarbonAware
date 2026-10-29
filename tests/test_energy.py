import unittest

from src.energy import emissions_g, energy_kwh, job_power_w, migration_cost, node_power_w
from src.models import JobSpec, NodeSpec

NODE = NodeSpec("n", "s", "edge", 8, 16, 40, 120, 1.5, 8)
JOB = JobSpec(1, 0, 4, 2, 4, 0.5, 50, 0, 512)


class EnergyTests(unittest.TestCase):
    def test_power_model_endpoints(self):
        self.assertAlmostEqual(node_power_w(NODE, 0), 40)
        self.assertAlmostEqual(node_power_w(NODE, 8), 120)
        self.assertAlmostEqual(node_power_w(NODE, 100), 120)  # capped at 100% utilisation

    def test_emissions_formula(self):
        kwh = energy_kwh(100, 1.5, 0.5)           # 100 W * 1.5 PUE * 0.5 h = 0.075 kWh
        self.assertAlmostEqual(kwh, 0.075)
        self.assertAlmostEqual(emissions_g(kwh, 200), 15.0)  # 0.075 kWh * 200 g/kWh

    def test_job_attribution_not_above_node_power(self):
        full_node_power = node_power_w(NODE, JOB.cpu_req * JOB.cpu_util)
        self.assertLessEqual(job_power_w(JOB, NODE), full_node_power + 1e-9)

    def test_migration_cost(self):
        m = {"bandwidth_mbps": 100, "fixed_downtime_s": 30, "transfer_kwh_per_gb": 0.03}
        c = migration_cost(JOB, m, 1800)
        self.assertAlmostEqual(c.transfer_seconds, 512 * 8 / 100)
        self.assertAlmostEqual(c.downtime_seconds, 512 * 8 / 100 + 30)
        self.assertAlmostEqual(c.energy_kwh, 0.5 * 0.03)
        self.assertAlmostEqual(c.extra_work_steps, c.downtime_seconds / 1800)


if __name__ == "__main__":
    unittest.main()
