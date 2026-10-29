"""Importer tests. All network access is mocked; NESO rows are FABRICATED test fixtures."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import carbon_import as ci
from src.data_sources import load_carbon_csv
from src.errors import DataError
from src.pipeline import describe_carbon_origin


def fake_neso_csv(path, days=20, start="2026-01-01", extra_old_days=30):
    idx = pd.date_range(start, periods=(days + extra_old_days) * 48, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"DATETIME": idx.strftime("%Y-%m-%dT%H:%M:%SZ"), "GAS": 1.0, "WIND": 2.0,
                       "CARBON_INTENSITY": np.round(150 + 40 * np.sin(np.arange(len(idx)) / 7) + rng.normal(0, 3, len(idx)), 0)})
    df.to_csv(path, index=False)
    return idx


def _fail(*a, **k):
    raise DataError("network down")


class ParseNesoTests(unittest.TestCase):
    def test_keeps_only_last_days_and_labels_real(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.csv"
            idx = fake_neso_csv(p, days=20, extra_old_days=30)
            out = ci.parse_neso_csv(p, days=20)
        self.assertEqual(len(out), 20 * 48)
        self.assertTrue((out.data_origin == "REAL_OBSERVATION").all())
        self.assertEqual(pd.to_datetime(out["from"].iloc[-1], utc=True), idx[-1])

    def test_case_insensitive_columns_and_bad_values_become_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.csv"
            fake_neso_csv(p, days=20, extra_old_days=0)
            df = pd.read_csv(p).rename(columns=str.lower)
            df.loc[100, "carbon_intensity"] = -5      # impossible -> dropped, then interpolated by loader
            df.to_csv(p, index=False)
            out = ci.parse_neso_csv(p, days=20)
            self.assertEqual(len(out), 20 * 48 - 1)
            q = Path(d) / "std.csv"
            out.to_csv(q, index=False)
            self.assertEqual(len(load_carbon_csv(q)), 20 * 48)

    def test_missing_column_and_too_few_rows(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.csv"
            pd.DataFrame({"DATETIME": ["2026-01-01T00:00:00Z"], "GAS": [1]}).to_csv(p, index=False)
            with self.assertRaises(DataError):
                ci.parse_neso_csv(p)
            fake_neso_csv(p, days=5, extra_old_days=0)
            with self.assertRaises(DataError):
                ci.parse_neso_csv(p, days=60)

    def test_url_has_no_tracking_suffix(self):
        self.assertNotIn("utm_", ci.NESO_URL)


class ImportFlowTests(unittest.TestCase):
    NOW = datetime(2026, 10, 29, tzinfo=timezone.utc)

    def _run(self, source="auto", **patches):
        d = tempfile.mkdtemp()
        out = Path(d) / "c.csv"
        with patch("src.carbon_import.resolve_path", lambda p: Path(p)):
            return d, out, ci.import_carbon(20, str(out), source, now=self.NOW)

    def test_neso_success_writes_csv_and_provenance(self):
        def fake_download(url, dest, timeout=60):
            fake_neso_csv(dest, days=20, extra_old_days=5)
        with patch.object(ci, "_download_file", fake_download):
            d, out, res = self._run("neso")
        self.assertTrue(res.is_real)
        self.assertEqual(res.source, "neso")
        self.assertEqual(res.rows, 20 * 48)
        prov = json.loads((Path(d) / "c.csv.provenance.json").read_text())
        self.assertEqual(prov["data_origin"], "REAL_OBSERVATION")
        self.assertIn("sha256", prov)
        self.assertGreater(prov["data_age_hours"], 0)
        self.assertEqual(load_carbon_csv(out).attrs["origin"], "REAL_OBSERVATION")

    def test_auto_falls_through_neso_to_ukapi(self):
        idx = pd.date_range("2026-10-09", periods=20 * 48, freq="30min", tz="UTC")
        api = pd.DataFrame({"from": idx.strftime("%Y-%m-%dT%H:%MZ"), "actual": 120.0, "data_origin": "REAL_OBSERVATION"})
        with patch.object(ci, "fetch_neso", _fail), patch.object(ci, "fetch_ukapi", lambda days: api):
            _, _, res = self._run("auto")
        self.assertEqual(res.source, "ukapi")
        self.assertTrue(any("neso failed" in n for n in res.notes))

    def test_offline_without_cache_is_labelled_synthetic_not_real(self):
        with patch.object(ci, "fetch_neso", _fail), patch.object(ci, "fetch_ukapi", _fail):
            _, out, res = self._run("auto")
        self.assertFalse(res.is_real)
        self.assertEqual(res.data_origin, "SYNTHETIC_FALLBACK")
        self.assertEqual(load_carbon_csv(out).attrs["origin"], "SYNTHETIC_FALLBACK")
        self.assertIn("NOT real", describe_carbon_origin("SYNTHETIC_FALLBACK", None))

    def test_offline_reuses_cached_real_file_and_relabels_it(self):
        d = tempfile.mkdtemp()
        out = Path(d) / "c.csv"
        with patch("src.carbon_import.resolve_path", lambda p: Path(p)):
            with patch.object(ci, "fetch_neso", lambda days, url=ci.NESO_URL: ci.parse_neso_csv(_neso(d), days)):
                first = ci.import_carbon(20, str(out), "neso", now=self.NOW)
            with patch.object(ci, "fetch_neso", _fail), patch.object(ci, "fetch_ukapi", _fail):
                again = ci.import_carbon(20, str(out), "auto", now=self.NOW)
        self.assertTrue(first.is_real)
        self.assertTrue(again.is_real)
        self.assertEqual(again.data_origin, "CACHED_REAL_OBSERVATION")
        self.assertEqual(again.rows, first.rows)            # real data was NOT overwritten by synthetic
        self.assertEqual(load_carbon_csv(out).attrs["origin"], "CACHED_REAL_OBSERVATION")

    def test_explicit_offline_source_and_validation(self):
        with patch("src.carbon_import.resolve_path", lambda p: Path(p)):
            with self.assertRaises(DataError):
                ci.import_carbon(5, "x.csv", "auto")
            with self.assertRaises(DataError):
                ci.import_carbon(20, "x.csv", "bogus")

    def test_unknown_origin_is_not_called_real(self):
        self.assertNotIn("REAL_OBSERVATION", describe_carbon_origin(None, None))
        self.assertTrue(describe_carbon_origin("REAL_OBSERVATION", {"source": "neso", "end_utc": "x"}).startswith("REAL_OBSERVATION"))


def _neso(d):
    p = Path(d) / "neso_src.csv"
    fake_neso_csv(p, days=20, extra_old_days=3)
    return p


if __name__ == "__main__":
    unittest.main()
