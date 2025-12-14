#!/usr/bin/env python3
"""
Parallel Fertimap soil atlas exporter.

Version: dump full regional/generic recommendation text into CSV columns.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------
BASE_URL = "http://www.fertimap.ma"
INFO_ENDPOINT = f"{BASE_URL}/php/info.phtml"
RECO_ENDPOINT = f"{BASE_URL}/php/calcul.php"

TARGET_CROP_ID = "1"
TARGET_CROP_NAME = "Blé bour"

MAP_BOUNDS = {
    "lon_min": -12.5,
    "lon_max": -0.5,
    "lat_min": 27.0,
    "lat_max": 36.5,
}

MAX_RETRIES = 3

# default delay (overridden by CLI)
REQUEST_DELAY = 0.05

# we now add two columns: regional_recommendations, generic_recommendations
OUTPUT_COLUMNS = [
    "lat",
    "lon",
    "region",
    "province",
    "province_id",
    "commune",
    "soil_type",
    "texture",
    "ph",
    "organic_matter_pct",
    "p2o5_mgkg",
    "k2o_mgkg",
    "target_crop_id",
    "target_crop_name",
    "target_yield_min",
    "target_yield_max",
    "target_yield_step",
    "target_yield_unit",
    "target_expected_yield",
    "scenario_expected_yield",
    "rec_n_kg_ha",
    "rec_p_kg_ha",
    "rec_k_kg_ha",
    "rec_cost_dh_ha",
    "regional_recommendations",
    "generic_recommendations",
]

# precompiled regex for info page
REGION_RE = re.compile(r"Région\s*[:\s]+([^\n\r]+)", re.IGNORECASE)
PROVINCE_RE = re.compile(r"Préfecture\s*/\s*Province\s*[:\s]+([^\n\r]+)", re.IGNORECASE)
COMMUNE_RE = re.compile(r"Commune\s*[:\s]+([^\n\r]+)", re.IGNORECASE)
SOIL_RE = re.compile(r"Type de sol\s*[:\s]+([^\n\r]+)", re.IGNORECASE)
TEXTURE_RE = re.compile(r"Texture\s*[:\s]+([^\n\r]+)", re.IGNORECASE)

# thread-local requests session
thread_local = threading.local()


# ---------------------------------------------------------------------
# HTTP SESSION
# ---------------------------------------------------------------------
def build_session(user_agent: str, retries_total: int = 5) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=retries_total,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=200, pool_maxsize=200)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


def get_thread_session(user_agent: str) -> requests.Session:
    sess = getattr(thread_local, "session", None)
    if sess is None:
        sess = build_session(user_agent)
        thread_local.session = sess
    return sess


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip().replace(",", ".")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------
# PARSE INFO PAGE
# ---------------------------------------------------------------------
def parse_info_payload(html: str) -> Optional[Dict[str, Optional[float]]]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    def input_value(input_id: str) -> Optional[str]:
        tag = soup.find("input", {"id": input_id})
        return tag.get("value") if tag else None

    def hidden_value(field_id: str) -> Optional[str]:
        tag = soup.find("input", {"id": field_id})
        return tag.get("value") if tag else None

    province_id = hidden_value("id_province")
    if not province_id:
        province_select = soup.find("select", {"name": re.compile("province", re.I)})
        if province_select:
            selected = province_select.find("option", selected=True)
            if selected:
                province_id = selected.get("value")

    payload: Dict[str, Optional[float]] = {
        "province_id": province_id,
        "region": REGION_RE.search(text).group(1).strip() if REGION_RE.search(text) else "",
        "province": PROVINCE_RE.search(text).group(1).strip() if PROVINCE_RE.search(text) else "",
        "commune": COMMUNE_RE.search(text).group(1).strip() if COMMUNE_RE.search(text) else "",
        "soil_type": SOIL_RE.search(text).group(1).strip() if SOIL_RE.search(text) else "",
        "texture": TEXTURE_RE.search(text).group(1).strip() if TEXTURE_RE.search(text) else "",
        "ph": safe_float(input_value("ph")),
        "organic_matter_pct": safe_float(input_value("mo")),
        "p2o5_mgkg": safe_float(input_value("p")),
        "k2o_mgkg": safe_float(input_value("k")),
    }

    payload.update(
        {
            "target_crop_id": TARGET_CROP_ID,
            "target_crop_name": TARGET_CROP_NAME,
            "target_yield_min": safe_float(hidden_value(f"min{TARGET_CROP_ID}")),
            "target_yield_max": safe_float(hidden_value(f"max{TARGET_CROP_ID}")),
            "target_yield_step": safe_float(hidden_value(f"step{TARGET_CROP_ID}")),
            "target_yield_unit": hidden_value(f"unite{TARGET_CROP_ID}"),
        }
    )

    slider_value = safe_float(input_value("rdt"))
    if slider_value is None:
        ymin = payload["target_yield_min"]
        ymax = payload["target_yield_max"]
        slider_value = round((ymin + ymax) / 2, 2) if ymin is not None and ymax is not None else 40.0
    payload["target_expected_yield"] = slider_value

    # if the soil info is basically empty, skip
    if all(
        payload.get(key) in (None, 0.0)
        for key in ("ph", "organic_matter_pct", "p2o5_mgkg", "k2o_mgkg")
    ):
        return None

    return payload


# ---------------------------------------------------------------------
# PARSE RECOMMENDATION PAGE
# ---------------------------------------------------------------------
def parse_recommendation_html(html: str) -> Dict[str, Optional[str]]:
    """
    We now just grab the full text blocks for:
    - "Recommandations basées sur la formule régionale : ... "
    - "Recommandations basées sur les formules génériques : ... "

    and return them as plain text strings.

    We still extract N/P/K/cost because that's reliable and cheap.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    # capture big sections
    regional_match = re.search(
        r"(Recommandations basées sur la formule régionale\s*:.*?)(?=Recommandations basées sur les formules génériques\s*:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    generic_match = re.search(
        r"(Recommandations basées sur les formules génériques\s*:.*?)(?=Recommandations basées sur la formule régionale\s*:|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    regional_text = regional_match.group(1).strip() if regional_match else ""
    generic_text = generic_match.group(1).strip() if generic_match else ""

    # numeric fields
    def extract(pattern: str) -> Optional[float]:
        m = re.search(pattern, text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    rec_n = extract(r"N\s*\(kg\s*N/ha\)\s*[:\t\s]*(\d+\.?\d*)")
    rec_p = extract(r"P\s*\(kg\s*P/ha\)\s*[:\t\s]*(\d+\.?\d*)")
    rec_k = extract(r"K\s*\(kg\s*K/ha\)\s*[:\t\s]*(\d+\.?\d*)")
    rec_cost = extract(r"(\d+\.?\d*)\s*dh/ha")

    return {
        "rec_n_kg_ha": rec_n,
        "rec_p_kg_ha": rec_p,
        "rec_k_kg_ha": rec_k,
        "rec_cost_dh_ha": rec_cost,
        "regional_recommendations": regional_text,
        "generic_recommendations": generic_text,
    }


# ---------------------------------------------------------------------
# REQUESTS TO SERVER
# ---------------------------------------------------------------------
def fetch_recommendation(
    payload: Dict[str, Optional[float]],
    expected_yield: float,
    *,
    user_agent: str,
    timeout: float,
) -> Dict[str, Optional[str]]:
    required_keys = ("province_id", "ph", "organic_matter_pct", "p2o5_mgkg", "k2o_mgkg")
    if any(payload.get(key) is None for key in required_keys):
        return {
            "rec_n_kg_ha": None,
            "rec_p_kg_ha": None,
            "rec_k_kg_ha": None,
            "rec_cost_dh_ha": None,
            "regional_recommendations": "",
            "generic_recommendations": "",
        }

    params = {
        "id_province": payload["province_id"],
        "culture": TARGET_CROP_ID,
        "ph": payload["ph"],
        "mo": payload["organic_matter_pct"],
        "p": payload["p2o5_mgkg"],
        "k": payload["k2o_mgkg"],
        "rdt": expected_yield,
        "x_coord": round(payload["lon"], 4),
        "y_coord": round(payload["lat"], 4),
    }

    session = get_thread_session(user_agent)
    try:
        response = session.get(RECO_ENDPOINT, params=params, timeout=timeout)
        response.raise_for_status()
        return parse_recommendation_html(response.text)
    except requests.RequestException:
        return {
            "rec_n_kg_ha": None,
            "rec_p_kg_ha": None,
            "rec_k_kg_ha": None,
            "rec_cost_dh_ha": None,
            "regional_recommendations": "",
            "generic_recommendations": "",
        }


def sample_point(
    lon: float,
    lat: float,
    *,
    user_agent: str,
    timeout: float,
    yield_scenarios: Sequence[Optional[float]],
) -> List[Dict[str, Optional[str]]]:
    session = get_thread_session(user_agent)
    params = {"x": f"{lon:.6f}", "y": f"{lat:.6f}"}

    payload: Optional[Dict[str, Optional[float]]] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(INFO_ENDPOINT, params=params, timeout=timeout)
            response.raise_for_status()
            payload = parse_info_payload(response.text)
            if not payload:
                return []
            payload["lon"] = lon
            payload["lat"] = lat
            break
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                return []
            time.sleep(REQUEST_DELAY * attempt)

    if payload is None:
        return []

    default_yield = payload["target_expected_yield"]
    scenarios = yield_scenarios or [default_yield]
    rows: List[Dict[str, Optional[str]]] = []
    for val in scenarios:
        scenario_yield = default_yield if val is None else val
        rec = fetch_recommendation(payload, scenario_yield, user_agent=user_agent, timeout=timeout)
        row: Dict[str, Optional[str]] = {**payload, **rec}
        row["scenario_expected_yield"] = scenario_yield
        rows.append(row)

    return rows


# ---------------------------------------------------------------------
# RESOLUTION DETECTION (PARALLEL)
# ---------------------------------------------------------------------
def _probe_resolution_point(
    lon: float,
    lat: float,
    steps: Sequence[float],
    *,
    user_agent: str,
    timeout: float,
) -> Tuple[Optional[float], Optional[float]]:
    base_rows = sample_point(lon, lat, user_agent=user_agent, timeout=timeout, yield_scenarios=[None])
    if not base_rows:
        return None, None

    base = base_rows[0]
    base_vals = (
        base.get("ph"),
        base.get("organic_matter_pct"),
        base.get("p2o5_mgkg"),
        base.get("k2o_mgkg"),
    )

    lat_res = None
    lon_res = None

    for step in steps:
        other_rows = sample_point(lon, lat + step, user_agent=user_agent, timeout=timeout, yield_scenarios=[None])
        if other_rows:
            other = other_rows[0]
            other_vals = (
                other.get("ph"),
                other.get("organic_matter_pct"),
                other.get("p2o5_mgkg"),
                other.get("k2o_mgkg"),
            )
            if other_vals != base_vals:
                lat_res = step
                break

    for step in steps:
        other_rows = sample_point(lon + step, lat, user_agent=user_agent, timeout=timeout, yield_scenarios=[None])
        if other_rows:
            other = other_rows[0]
            other_vals = (
                other.get("ph"),
                other.get("organic_matter_pct"),
                other.get("p2o5_mgkg"),
                other.get("k2o_mgkg"),
            )
            if other_vals != base_vals:
                lon_res = step
                break

    return lat_res, lon_res


def detect_native_resolution(
    *,
    sample_size: int,
    user_agent: str,
    timeout: float,
    max_workers: int = 32,
) -> Tuple[float, float]:
    steps = sorted(
        {0.5, 0.25, 0.125, 0.0625, 0.03125, 1 / 24, 1 / 48, 1 / 72, 1 / 96, 1 / 120}
    )
    rng = random.Random(42)

    lat_res_global: Optional[float] = None
    lon_res_global: Optional[float] = None

    points: List[Tuple[float, float]] = []
    for _ in range(sample_size):
        lon = rng.uniform(MAP_BOUNDS["lon_min"], MAP_BOUNDS["lon_max"])
        lat = rng.uniform(MAP_BOUNDS["lat_min"], MAP_BOUNDS["lat_max"])
        points.append((lon, lat))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _probe_resolution_point,
                lon,
                lat,
                steps,
                user_agent=user_agent,
                timeout=timeout,
            )
            for (lon, lat) in points
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Detecting resolution"):
            lat_res, lon_res = fut.result()
            if lat_res:
                lat_res_global = lat_res if lat_res_global is None else min(lat_res_global, lat_res)
            if lon_res:
                lon_res_global = lon_res if lon_res_global is None else min(lon_res_global, lon_res)

            if (
                lat_res_global is not None
                and lon_res_global is not None
                and lat_res_global <= (1 / 120)
                and lon_res_global <= (1 / 120)
            ):
                break

    if lat_res_global is None or lon_res_global is None:
        raise RuntimeError("Resolution detection failed; increase sample size or check connectivity.")

    return lat_res_global, lon_res_global


# ---------------------------------------------------------------------
# GRID + HARVEST
# ---------------------------------------------------------------------
def build_axis(min_value: float, max_value: float, step: float) -> np.ndarray:
    count = int(math.floor((max_value - min_value) / step)) + 1
    values = min_value + step * np.arange(count)
    return np.round(values, 6)


def harvest_grid(
    lat_axis: np.ndarray,
    lon_axis: np.ndarray,
    *,
    output_path: Path,
    user_agent: str,
    timeout: float,
    yield_scenarios: Sequence[Optional[float]],
    max_workers: int,
    chunk_size: int,
    flush_rows: int,
    request_delay: float,
) -> None:
    total_points = len(lat_axis) * len(lon_axis)
    buffer: List[Dict[str, Optional[str]]] = []
    wrote_header = output_path.exists()

    def point_iter() -> Iterator[Tuple[float, float]]:
        for lat in lat_axis:
            for lon in lon_axis:
                yield lon, lat

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda pt: sample_point(
                pt[0],
                pt[1],
                user_agent=user_agent,
                timeout=timeout,
                yield_scenarios=yield_scenarios,
            ),
            point_iter(),
            chunksize=chunk_size,
        )

        for rows in tqdm(results, total=total_points, desc="Sampling Fertimap grid"):
            if request_delay > 0:
                time.sleep(request_delay)
            if not rows:
                continue
            for row in rows:
                buffer.append({col: row.get(col) for col in OUTPUT_COLUMNS})
            if len(buffer) >= flush_rows:
                df = pd.DataFrame(buffer, columns=OUTPUT_COLUMNS)
                df.to_csv(
                    output_path,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                )
                wrote_header = True
                buffer.clear()

    if buffer:
        df = pd.DataFrame(buffer, columns=OUTPUT_COLUMNS)
        df.to_csv(
            output_path,
            mode="a",
            header=not wrote_header,
            index=False,
        )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel Fertimap soil atlas exporter (text dump mode)")
    parser.add_argument("--output", default="data/fertimap_grid.csv", help="Path to output CSV")
    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Sampling stride multiplier relative to native resolution (1 = ~1 km)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=94,
        help="Number of worker threads",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.05,
        help="Sleep between samples (seconds)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Thread pool map chunk size",
    )
    parser.add_argument(
        "--flush-rows",
        type=int,
        default=5_000,
        help="Flush CSV after this many rows",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=80,
        help="Number of resolution probe attempts",
    )
    parser.add_argument(
        "--yields",
        type=float,
        nargs="*",
        default=[40.0, 70.0],
        help="Expected yields to request (use none to keep slider value)",
    )
    parser.add_argument(
        "--user-agent",
        default="fertimap-grid-export/3.3 (+github.com/othmaneechchabi)",
        help="Custom User-Agent header",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    global REQUEST_DELAY
    REQUEST_DELAY = args.request_delay

    yield_scenarios: List[Optional[float]] = args.yields if args.yields else [None]

    lat_res, lon_res = detect_native_resolution(
        sample_size=args.sample_size,
        user_agent=args.user_agent,
        timeout=args.timeout,
        max_workers=min(32, args.max_workers),
    )

    base_res = min(lat_res, lon_res)
    step = base_res * max(1, args.stride)

    lat_axis = build_axis(MAP_BOUNDS["lat_min"], MAP_BOUNDS["lat_max"], step)
    lon_axis = build_axis(MAP_BOUNDS["lon_min"], MAP_BOUNDS["lon_max"], step)

    print(f"Detected resolution: {lat_res:.6f}° lat x {lon_res:.6f}° lon (step={step:.6f})")
    print(
        f"Lat samples: {len(lat_axis)}, Lon samples: {len(lon_axis)}, total points: {len(lat_axis) * len(lon_axis):,}"
    )

    harvest_grid(
        lat_axis,
        lon_axis,
        output_path=output_path,
        user_agent=args.user_agent,
        timeout=args.timeout,
        yield_scenarios=yield_scenarios,
        max_workers=args.max_workers,
        chunk_size=args.chunk_size,
        flush_rows=args.flush_rows,
        request_delay=args.request_delay,
    )

    df = pd.read_csv(output_path)
    print(df.head())
    print(f"Rows collected: {len(df):,}")


if __name__ == "__main__":
    main()
