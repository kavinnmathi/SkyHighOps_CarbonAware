import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import load_config, validate_config
from src.data_sources import (download_carbon_intensity, generate_synthetic_carbon, generate_synthetic_workload,
                              load_carbon_csv, load_google_trace_extract, workload_from_dataframe)
from src.errors import ConfigError, DataError


class ConfigTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        self.assertIn("nodes", load_config())

    def test_bad_node_is_rejected(self):
        cfg = copy.deepcopy(load_config())
        cfg["nodes"][0]["idle_power_w"] = 999
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_unknown_site_is_rejected(self):
        cfg = copy.deepcopy(load_config())
        cfg["nodes"][0]["site_id"] = "nope"
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_negative_deferral_cap_is_rejected(self):
        cfg = copy.deepcopy(load_config())
        cfg["carbon_aware_scheduler"]["max_deferral_steps"] = -1
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_invalid_deferral_margin_is_rejected(self):
        cfg = copy.deepcopy(load_config())
        cfg["carbon_aware_scheduler"]["defer_margin"] = 1.0
        with self.assertRaises(ConfigError):
            validate_config(cfg)

    def test_invalid_migration_settings_are_rejected(self):
        for key, value in (("max_migrations_per_job", -1), ("bandwidth_mbps", 0),
                           ("transfer_kwh_per_gb", -0.1)):
            with self.subTest(key=key):
                cfg = copy.deepcopy(load_config())
                cfg["migration"][key] = value
                with self.assertRaises(ConfigError):
                    validate_config(cfg)

    def test_missing_user_config_file(self):
        with self.assertRaises(ConfigError):
            load_config("does_not_exist.json")


class DataTests(unittest.TestCase):
    def test_missing_carbon_file(self):
        with self.assertRaises(DataError):
            load_carbon_csv("data/real/nope.csv")

    def test_downloader_rejects_bad_dates_without_network_access(self):
        with self.assertRaises(DataError):
            download_carbon_intensity("not-a-date", "2026-08-01", "unused.csv")
        with self.assertRaises(DataError):
            download_carbon_intensity("2026-08-02", "2026-08-01", "unused.csv")

    def test_downloader_parses_mock_api_response(self):
        # Offline contract test: validates API response parsing and CSV writing,
        # but does not claim the live public API was reached.
        import io
        import json

        payload = {"data": [
            {"from": "2026-01-01T00:00Z", "to": "2026-01-01T00:30Z",
             "intensity": {"forecast": 100, "actual": 95, "index": "moderate"}},
            {"from": "2026-01-01T00:30Z", "to": "2026-01-01T01:00Z",
             "intensity": {"forecast": 110, "actual": 105, "index": "moderate"}},
        ]}

        class FakeResponse:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return json.dumps(payload).encode("utf-8")

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "carbon.csv"
            with patch("src.data_sources.urllib.request.urlopen", return_value=FakeResponse()):
                n = download_carbon_intensity("2026-01-01", "2026-01-02", out)
            self.assertEqual(n, 2)
            saved = pd.read_csv(out)
            self.assertEqual(list(saved.actual), [95, 105])
            self.assertTrue((saved.data_origin == "REAL_OBSERVATION").all())

    def test_synthetic_carbon_is_deterministic_and_labelled(self):
        a, b = generate_synthetic_carbon(5, seed=1), generate_synthetic_carbon(5, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue((a.data_origin == "SYNTHETIC_SAMPLE").all())

    def test_gap_filling(self):
        df = generate_synthetic_carbon(3, seed=2).drop(index=[10, 11])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.csv"
            df.to_csv(p, index=False)
            out = load_carbon_csv(p)
        self.assertEqual(len(out), 3 * 48)
        self.assertEqual(out.attrs["n_filled"], 2)

    def test_long_gap_raises(self):
        df = generate_synthetic_carbon(3, seed=2).drop(index=range(10, 30))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.csv"
            df.to_csv(p, index=False)
            with self.assertRaises(DataError):
                load_carbon_csv(p)

    def test_workload_has_required_fields(self):
        cfg = load_config()
        wl = generate_synthetic_workload(48, cfg["synthetic_workload"], 1)
        jobs = workload_from_dataframe(wl)
        self.assertGreater(len(jobs), 0)
        self.assertTrue(all(j.cpu_req > 0 and j.mem_req_gb > 0 and j.latency_req_ms > 0 for j in jobs))

    def test_google_extract_mapping_with_tiny_fake_file(self):
        # TEST FIXTURE ONLY (fabricated rows used to test the converter, not real trace data)
        cfg = load_config()
        raw = pd.DataFrame({"collection_id": [1, 2], "instance_index": [0, 0],
                            "start_time": [6e8, 6e8 + 3600e6], "end_time": [6e8 + 1800e6, 6e8 + 7200e6],
                            "avg_cpus": [0.01, 0.05], "avg_memory": [0.002, 0.01]})
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            raw.to_csv(p, index=False)
            out = load_google_trace_extract(p, cfg, sim_steps=336)
        self.assertEqual(list(out.arrival_step), [0, 2])
        self.assertTrue((out.duration_steps >= 1).all())
        self.assertIn("SIMULATED", out.data_origin.iloc[0])

    def test_google_extract_missing_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"
            pd.DataFrame({"a": [1]}).to_csv(p, index=False)
            with self.assertRaises(DataError):
                load_google_trace_extract(p, load_config(), 336)


if __name__ == "__main__":
    unittest.main()
