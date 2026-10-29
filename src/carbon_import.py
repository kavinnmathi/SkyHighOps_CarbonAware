"""Free, key-less real carbon-intensity importer with offline fallback and provenance labels.

Source order for ``source="auto"``:
  1. NESO "Historic generation mix and carbon intensity" CSV (df_fuel_ckan.csv)
  2. UK National Grid Carbon Intensity API (api.carbonintensity.org.uk)
  3. a previously imported REAL file already on disk   -> label CACHED_REAL_OBSERVATION
  4. the bundled synthetic sample                     -> label SYNTHETIC_FALLBACK (NOT real)

Every output CSV has a ``data_origin`` column and a ``<csv>.provenance.json`` sidecar
saying where the numbers came from, when they were fetched and the newest observation.
A fallback can never be mistaken for real data: the label is inside the CSV itself.

NOTE: the NESO file is a periodically updated *historical* file, not a live stream, so
the importer records ``latest_observation_utc`` and ``data_age_hours`` instead of
claiming the data is real-time.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_path
from .data_sources import STEPS_PER_DAY, download_carbon_intensity, generate_synthetic_carbon
from .errors import DataError

log = logging.getLogger("skyhighops")

# utm_source=chatgpt.com tracking suffix removed on purpose; it is not needed.
NESO_URL = ("https://api.neso.energy/dataset/88313ae5-94e4-4ddc-a790-593554d8c6b9/"
            "resource/f93d1835-75bc-43e5-84ad-12472b180a98/download/df_fuel_ckan.csv")

ORIGIN_NESO = "REAL_OBSERVATION"
ORIGIN_UKAPI = "REAL_OBSERVATION"
ORIGIN_CACHED = "CACHED_REAL_OBSERVATION"
ORIGIN_FALLBACK = "SYNTHETIC_FALLBACK"
REAL_ORIGINS = {ORIGIN_NESO, ORIGIN_UKAPI, ORIGIN_CACHED}
SOURCES = ("auto", "neso", "ukapi", "offline")
DEFAULT_OUT = "data/real/carbon_intensity_gb.csv"


@dataclass
class ImportResult:
    out_path: str
    source: str            # neso | ukapi | cache | synthetic_fallback
    data_origin: str       # label written into the CSV
    is_real: bool
    rows: int
    start_utc: str
    end_utc: str            # latest observation in the file
    retrieved_utc: str
    data_age_hours: float
    url: str | None
    sha256: str
    notes: list[str]


# ------------------------------------------------------------------- parsing
def _find_col(columns, wanted: str):
    for c in columns:
        if str(c).strip().lower() == wanted:
            return c
    return None


def parse_neso_csv(path, days: int = 60) -> pd.DataFrame:
    """Read the NESO file (can be ~100 MB) keeping only the newest ``days`` days.

    Needs the DATETIME and CARBON_INTENSITY columns (case-insensitive). Returns the
    project's standard columns: from, actual, data_origin.
    """
    p = Path(path)
    head = pd.read_csv(p, nrows=0)
    c_dt, c_ci = _find_col(head.columns, "datetime"), _find_col(head.columns, "carbon_intensity")
    if c_dt is None or c_ci is None:
        raise DataError("NESO file needs DATETIME and CARBON_INTENSITY columns; found "
                        f"{list(head.columns)[:8]}...")
    parts = []
    for chunk in pd.read_csv(p, usecols=[c_dt, c_ci], chunksize=100_000):
        chunk[c_dt] = pd.to_datetime(chunk[c_dt], utc=True, errors="coerce")
        parts.append(chunk.dropna(subset=[c_dt]))
    if not parts:
        raise DataError("NESO file contained no rows.")
    df = pd.concat(parts).drop_duplicates(c_dt).sort_values(c_dt)
    df[c_ci] = pd.to_numeric(df[c_ci], errors="coerce")
    df.loc[(df[c_ci] <= 0) | (df[c_ci] > 1500), c_ci] = np.nan   # impossible values -> gap
    df = df.dropna(subset=[c_ci])
    if df.empty:
        raise DataError("NESO file has no valid CARBON_INTENSITY values.")
    df = df[df[c_dt] > df[c_dt].max() - timedelta(days=days)]
    need = int(days * STEPS_PER_DAY * 0.9)
    if len(df) < need:
        raise DataError(f"NESO file gives only {len(df)} valid rows for the last {days} days (need >= {need}).")
    return pd.DataFrame({"from": df[c_dt].dt.strftime("%Y-%m-%dT%H:%MZ").values,
                         "actual": df[c_ci].round(1).values,
                         "data_origin": ORIGIN_NESO})


# --------------------------------------------------------------------- fetch
def _download_file(url: str, dest: Path, timeout: int = 60) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "SkyHighOps-research/1.0", "Accept": "text/csv"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh, 1 << 20)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataError(f"Download failed for {url}: {exc}") from exc


def fetch_neso(days: int, url: str = NESO_URL) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "neso.csv"
        _download_file(url, tmp)
        return parse_neso_csv(tmp, days)


def fetch_ukapi(days: int) -> pd.DataFrame:
    """UK Carbon Intensity API; trailing rows without an ``actual`` value are dropped so a
    *forecast* can never be silently used as observed truth."""
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "api.csv"
        download_carbon_intensity(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", tmp, retries=2)
        raw = pd.read_csv(tmp)
    raw["actual"] = pd.to_numeric(raw["actual"], errors="coerce")
    raw = raw.dropna(subset=["actual"])
    if len(raw) < int(days * STEPS_PER_DAY * 0.9):
        raise DataError(f"UK API returned only {len(raw)} rows with actual values.")
    return pd.DataFrame({"from": raw["from"].values, "actual": raw["actual"].values,
                         "data_origin": ORIGIN_UKAPI})


# ---------------------------------------------------------------- provenance
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def provenance_path(csv_path) -> Path:
    p = Path(csv_path)
    return p.with_name(p.name + ".provenance.json")


def read_provenance(csv_path) -> dict | None:
    pp = provenance_path(resolve_path(csv_path))
    if not pp.exists():
        return None
    try:
        return json.loads(pp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write(df: pd.DataFrame, out: Path, source: str, origin: str, url, notes, now) -> ImportResult:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    last = pd.to_datetime(df["from"].iloc[-1], utc=True)
    res = ImportResult(out_path=str(out), source=source, data_origin=origin, is_real=origin in REAL_ORIGINS,
                       rows=len(df), start_utc=str(pd.to_datetime(df["from"].iloc[0], utc=True)),
                       end_utc=str(last), retrieved_utc=now.isoformat(timespec="seconds"),
                       data_age_hours=round((now - last).total_seconds() / 3600.0, 1),
                       url=url, sha256=_sha256(out), notes=notes)
    provenance_path(out).write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
    return res


def import_carbon(days: int = 60, out_path: str = DEFAULT_OUT, source: str = "auto",
                  neso_url: str = NESO_URL, now: datetime | None = None) -> ImportResult:
    """Import ``days`` of GB half-hourly carbon intensity, falling back offline if needed."""
    if source not in SOURCES:
        raise DataError(f"source must be one of {SOURCES}.")
    if days < 16:
        raise DataError("days must be >= 16 (7 days training + 7 simulation days + drain).")
    now = now or datetime.now(timezone.utc)
    out = resolve_path(out_path)
    notes: list[str] = []

    attempts = {"auto": ["neso", "ukapi"], "neso": ["neso"], "ukapi": ["ukapi"], "offline": []}[source]
    for name in attempts:
        try:
            if name == "neso":
                df, url, origin = fetch_neso(days, neso_url), neso_url, ORIGIN_NESO
            else:
                df, url, origin = fetch_ukapi(days), "https://api.carbonintensity.org.uk", ORIGIN_UKAPI
            return _write(df, out, name, origin, url, notes, now)
        except DataError as exc:
            notes.append(f"{name} failed: {exc}")
            log.warning("Carbon import via %s failed: %s", name, exc)

    # ---- offline fallbacks (never overwrite real data with synthetic data)
    prev = read_provenance(out)
    if out.exists() and prev and prev.get("is_real"):
        notes.append("Using previously imported real file (offline).")
        prev.update(source="cache", data_origin=ORIGIN_CACHED, is_real=True, notes=notes)
        df = pd.read_csv(out)
        df["data_origin"] = ORIGIN_CACHED
        df.to_csv(out, index=False)
        res = ImportResult(**{k: prev[k] for k in ImportResult.__dataclass_fields__ if k in prev})
        res.sha256 = _sha256(out)
        res.data_age_hours = round((now - pd.to_datetime(res.end_utc, utc=True)).total_seconds() / 3600.0, 1)
        provenance_path(out).write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
        return res
    notes.append("No real data available: wrote SYNTHETIC fallback. Do NOT report results as real-world.")
    syn = generate_synthetic_carbon(days=max(days, 37))
    syn["data_origin"] = ORIGIN_FALLBACK
    return _write(syn, out, "synthetic_fallback", ORIGIN_FALLBACK, None, notes, now)
