"""Identical-workload guarantees for scheduler comparisons (synthetic data)."""
import copy
import unittest

from src import pipeline
from src.config import resolve_path
from tests.test_simulation import small_cfg


class IdenticalWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = small_cfg()
        cls.cfg["evaluation"]["robustness_seeds"] = 2
        cls.prep = pipeline.prepare(cls.cfg)
        cls.wl = cls.prep.workload.query("arrival_step < 96")
        cls.results = pipeline.simulate(cls.cfg, cls.prep, cls.wl)

    @classmethod
    def tearDownClass(cls):
        p = resolve_path(cls.cfg["forecaster"]["model_path"])
        if p.exists():
            p.unlink()

    def test_all_schedulers_see_the_same_jobs(self):
        fps = {pipeline.workload_fingerprint(r.jobs) for r in self.results}
        self.assertEqual(len(fps), 1)
        self.assertEqual(len({len(r.jobs) for r in self.results}), 1)
        self.assertEqual(len(self.results), 3)

    def test_comparison_table_records_workload_and_carbon_hash(self):
        table = pipeline.comparison_table(self.results)
        self.assertEqual(table.workload_sha256.nunique(), 1)
        self.assertEqual(table.carbon_series_sha256.nunique(), 1)

    def test_mismatched_workloads_are_rejected(self):
        other = pipeline.simulate(self.cfg, self.prep, self.wl.iloc[:-5], [pipeline.BaselineScheduler()])
        with self.assertRaises(RuntimeError):
            pipeline.assert_identical_workloads([self.results[0], other[0]])

    def test_fingerprint_is_order_independent_and_sensitive(self):
        j = self.results[0].jobs
        self.assertEqual(pipeline.workload_fingerprint(j), pipeline.workload_fingerprint(j.sample(frac=1, random_state=1)))
        j2 = j.copy()
        j2.loc[j2.index[0], "cpu_req"] += 1
        self.assertNotEqual(pipeline.workload_fingerprint(j), pipeline.workload_fingerprint(j2))

    def test_paired_comparison_is_deterministic_and_paired(self):
        a, sa = pipeline.paired_scheduler_comparison(self.cfg, self.prep, 2)
        b, _ = pipeline.paired_scheduler_comparison(self.cfg, self.prep, 2)
        self.assertEqual(a.job_emissions_g.round(6).tolist(), b.job_emissions_g.round(6).tolist())
        for _, grp in a.groupby("seed"):
            self.assertEqual(grp.workload_sha256.nunique(), 1)      # same jobs for all 3 schedulers in a seed
            self.assertEqual(len(grp), 3)
        self.assertEqual(set(sa.scheduler), {"baseline_fcfs_first_fit", "carbon_aware", "energy_aware_ablation"})
        base = a[a.scheduler == "baseline_fcfs_first_fit"].job_emissions_reduction_pct
        self.assertTrue((base.abs() < 1e-9).all())


if __name__ == "__main__":
    unittest.main()
