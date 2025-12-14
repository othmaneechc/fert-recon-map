"""Helper utilities to build Earth Engine map tiles."""

from __future__ import annotations

import calendar
import math
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import ee

from .datasets import (
    DATASETS,
    DEFAULT_COUNTRY,
    LEGENDS,
    TILE_CACHE_TTL_SECONDS,
    VIS_PARAMS,
    DatasetConfig,
)

COUNTRY_EXPANSIONS = {
    "Morocco": ["Morocco", "Western Sahara"],
}

WORLD_COVER_CLASSES = {
    0: "No data",
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


class TileServiceError(RuntimeError):
    """Raised when the tile service cannot fulfill a request."""


_EE_INITIALIZED = False
_COUNTRY_AOI_CACHE: Dict[str, ee.Geometry] = {}
_TILE_CACHE: Dict[Tuple[str, int, int, int, str], Dict[str, Any]] = {}
_ALLOWED_AGGREGATES = {"native", "monthly", "yearly"}
_CUSTOM_SERVICE_ACCOUNT: Optional[str] = None
_CUSTOM_PRIVATE_KEY_JSON: Optional[str] = None


@dataclass(frozen=True)
class TileRequest:
    dataset: str
    year: int
    month: int
    day: Optional[int]
    aggregate: str


def ensure_ee_initialized() -> None:
    """Initialise the Earth Engine API using environment-based credentials."""
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return

    if _CUSTOM_SERVICE_ACCOUNT and _CUSTOM_PRIVATE_KEY_JSON:
        try:
            creds = ee.ServiceAccountCredentials(
                _CUSTOM_SERVICE_ACCOUNT,
                key_data=_CUSTOM_PRIVATE_KEY_JSON,
            )
            ee.Initialize(creds)
            _EE_INITIALIZED = True
            return
        except Exception as exc:  # pragma: no cover - requires EE creds
            raise TileServiceError("Failed to initialise with uploaded credentials.") from exc
    raise TileServiceError(
        "Earth Engine is not authenticated. Upload a service account key JSON to continue."
    )


def list_datasets() -> Dict[str, Dict[str, Any]]:
    """Return dataset metadata for clients."""
    return {
        key: {
            "description": ds.get("description"),
            "cadence": ds.get("cadence"),
            "aggregation": ds.get("agg"),
            "scale_m": ds.get("scale_m"),
            "unit": ds.get("unit"),
            "visParams": VIS_PARAMS.get(key),
            "legend": LEGENDS.get(key, []),
        }
        for key, ds in DATASETS.items()
    }


def set_service_account_key(key_json: str) -> Dict[str, str]:
    """Initialise Earth Engine using an uploaded service account key."""
    global _CUSTOM_SERVICE_ACCOUNT, _CUSTOM_PRIVATE_KEY_JSON, _EE_INITIALIZED
    global _COUNTRY_AOI_CACHE, _TILE_CACHE

    try:
        parsed = json.loads(key_json)
    except json.JSONDecodeError as exc:  # pragma: no cover - user input
        raise TileServiceError("Uploaded key is not valid JSON.") from exc

    service_account = parsed.get("client_email")
    if not service_account:
        raise TileServiceError("Uploaded key is missing 'client_email'.")

    key_str = json.dumps(parsed)
    try:
        creds = ee.ServiceAccountCredentials(service_account, key_data=key_str)
        ee.Initialize(creds)
    except Exception as exc:  # pragma: no cover - EE credentials failure
        raise TileServiceError("Failed to initialise Earth Engine with uploaded key.") from exc

    _CUSTOM_SERVICE_ACCOUNT = service_account
    _CUSTOM_PRIVATE_KEY_JSON = key_str
    _EE_INITIALIZED = True
    _COUNTRY_AOI_CACHE = {}
    _TILE_CACHE = {}

    return {"serviceAccount": service_account}


def current_auth_status() -> Dict[str, Optional[str]]:
    return {
        "customServiceAccount": _CUSTOM_SERVICE_ACCOUNT,
        "usingUploadedKey": bool(_CUSTOM_SERVICE_ACCOUNT and _CUSTOM_PRIVATE_KEY_JSON),
        "initialized": _EE_INITIALIZED,
    }


def _country_aoi(country_name: str) -> ee.Geometry:
    cached = _COUNTRY_AOI_CACHE.get(country_name)
    if cached:
        return cached
    candidate_names = COUNTRY_EXPANSIONS.get(country_name, [country_name])
    fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(
        ee.Filter.inList("country_na", candidate_names)
    )
    geometry = fc.geometry()
    if geometry is None:
        raise TileServiceError(f"Could not find AOI for country '{country_name}'.")
    _COUNTRY_AOI_CACHE[country_name] = geometry
    return geometry


def _lookup_admin_names(point: ee.Geometry) -> Dict[str, Optional[str]]:
    admin1_name: Optional[str]
    admin2_name: Optional[str]
    try:
        admin1 = ee.FeatureCollection("FAO/GAUL/2015/level1").filterBounds(point).first()
        admin1_name = (
            admin1.get("ADM1_NAME").getInfo() if admin1 is not None else None
        )
    except Exception:  # pragma: no cover - EE lookups may fail offline
        admin1_name = None

    try:
        admin2 = ee.FeatureCollection("FAO/GAUL/2015/level2").filterBounds(point).first()
        admin2_name = (
            admin2.get("ADM2_NAME").getInfo() if admin2 is not None else None
        )
    except Exception:  # pragma: no cover
        admin2_name = None

    return {"admin1": admin1_name, "admin2": admin2_name}


def _build_collection_window(
    ds_cfg: DatasetConfig, start: ee.Date, end: ee.Date, aoi: ee.Geometry
) -> ee.Image:
    ic = (
        ee.ImageCollection(ds_cfg["ee_id"])
        .select(ds_cfg["band"])
        .filterDate(start, end)
    )
    agg = ds_cfg.get("agg", "mean").lower()
    if agg == "sum":
        img = ic.sum()
    elif agg == "median":
        img = ic.median()
    else:
        img = ic.mean()
    return img.toFloat().clip(aoi)


def _build_vpd_monthly(
    ds_cfg: DatasetConfig, start: ee.Date, end: ee.Date, aoi: ee.Geometry
) -> ee.Image:
    src = ds_cfg["source"]
    ic = ee.ImageCollection(src).filterDate(start, end)
    t2m = ic.select("temperature_2m").mean()
    td2m = ic.select("dewpoint_temperature_2m").mean()

    t_c = t2m.subtract(273.15)
    td_c = td2m.subtract(273.15)

    def _svp(temp_c: ee.Image) -> ee.Image:
        # Saturation vapour pressure (kPa) using Tetens formula.
        return (
            temp_c.multiply(17.27)
            .divide(temp_c.add(237.3))
            .exp()
            .multiply(0.6108)
        )

    es = _svp(t_c)
    e = _svp(td_c)
    vpd = es.subtract(e)
    vpd = vpd.where(vpd.lt(0), 0)
    return vpd.toFloat().clip(aoi)


def _transform_for_visualisation(dataset: str, image: ee.Image) -> ee.Image:
    """Apply dataset-specific scaling so styling thresholds match the legend."""
    if dataset in {"modis_ndvi", "modis_evi"}:
        return image.multiply(0.0001).rename(dataset)
    if dataset in {"modis_lst_day", "modis_lst_night"}:
        return image.multiply(0.02).subtract(273.15).rename(dataset)
    if dataset == "era5_temp2m":
        return image.subtract(273.15).rename(dataset)
    return image.rename(dataset)


def _time_window(
    dataset: str, year: int, month: int, day: Optional[int], aggregate: str
) -> Tuple[ee.Date, ee.Date]:
    config = DATASETS.get(dataset)
    if not config:
        raise TileServiceError(f"Unknown dataset '{dataset}'.")

    agg_mode = (aggregate or "native").lower()
    if agg_mode not in _ALLOWED_AGGREGATES:
        raise TileServiceError(
            f"Unknown aggregate '{aggregate}'. Expected one of {_ALLOWED_AGGREGATES}."
        )

    cadence = (config.get("cadence") or "MONTHLY").upper()

    if agg_mode == "yearly":
        start = ee.Date.fromYMD(year, 1, 1)
        end = start.advance(1, "year")
        return start, end

    if agg_mode == "monthly":
        start = ee.Date.fromYMD(year, month, 1)
        _, ndays = calendar.monthrange(year, month)
        end = start.advance(ndays, "day")
        return start, end

    # Native cadence handling
    if cadence in {"DAILY", "DAY"}:
        if day is None:
            raise TileServiceError(f"Dataset '{dataset}' requires a day selection.")
        _, ndays = calendar.monthrange(year, month)
        if day < 1 or day > ndays:
            raise TileServiceError(
                f"Day must be between 1 and {ndays} for {month:02d}/{year}."
            )
        start = ee.Date.fromYMD(year, month, day)
        end = start.advance(1, "day")
        return start, end

    if cadence in {"8D", "8DAY", "8-DAY", "EIGHTDAY", "EIGHT-DAY"}:
        if day is None:
            raise TileServiceError(f"Dataset '{dataset}' requires a day selection.")
        _, ndays = calendar.monthrange(year, month)
        if day < 1 or day > ndays:
            raise TileServiceError(
                f"Day must be between 1 and {ndays} for {month:02d}/{year}."
            )
        start = ee.Date.fromYMD(year, month, day)
        end = start.advance(8, "day")
        return start, end

    start = ee.Date.fromYMD(year, month, 1)
    _, ndays = calendar.monthrange(year, month)
    end = start.advance(ndays, "day")
    return start, end


def _build_dataset_image(
    dataset: str, year: int, month: int, day: Optional[int], aggregate: str
) -> ee.Image:
    config = DATASETS.get(dataset)
    if not config:
        raise TileServiceError(f"Unknown dataset '{dataset}'.")
    aoi = _country_aoi(DEFAULT_COUNTRY)
    start, end = _time_window(dataset, year, month, day, aggregate)
    if config["type"] == "collection":
        base = _build_collection_window(config, start, end, aoi)
    elif config["type"] == "derived_vpd":
        base = _build_vpd_monthly(config, start, end, aoi)
    else:
        raise TileServiceError(f"Dataset '{dataset}' has unsupported type '{config['type']}'.")

    # Convert to visualisation units and clip for safety.
    transformed = _transform_for_visualisation(dataset, base)
    return transformed.clip(aoi).toFloat(), start, end, config


def _cache_key(request: TileRequest) -> Tuple[str, int, int, int, str]:
    return (
        request.dataset,
        request.year,
        request.month,
        request.day or 0,
        request.aggregate,
    )


def get_tile_metadata(
    dataset: str,
    year: int,
    month: int,
    day: Optional[int],
    aggregate: str = "native",
) -> Dict[str, Any]:
    """Return (and cache) the EE tile metadata for a dataset selection."""
    ensure_ee_initialized()

    if month < 1 or month > 12:
        raise TileServiceError("Month must be in the range 1-12.")

    agg_mode = (aggregate or "native").lower()
    if agg_mode not in _ALLOWED_AGGREGATES:
        raise TileServiceError(
            f"Unknown aggregate '{aggregate}'. Expected one of {_ALLOWED_AGGREGATES}."
        )

    request = TileRequest(
        dataset=dataset,
        year=year,
        month=month,
        day=day,
        aggregate=agg_mode,
    )
    cache_key = _cache_key(request)
    now = time.time()
    cached = _TILE_CACHE.get(cache_key)
    if cached and cached["expires_at"] > now:
        return cached["payload"]

    image, start, end, config = _build_dataset_image(dataset, year, month, day, agg_mode)
    vis_params = VIS_PARAMS.get(dataset)
    if not vis_params:
        raise TileServiceError(f"Missing visualisation parameters for '{dataset}'.")

    map_dict = image.getMapId(vis_params)
    tile_url = map_dict["tile_fetcher"].url_format
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=TILE_CACHE_TTL_SECONDS)

    try:
        start_str = start.format("YYYY-MM-dd").getInfo()
    except Exception:  # pragma: no cover - EE formatting failure
        start_str = None
    try:
        end_inclusive = end.advance(-1, "day")
        end_str = end_inclusive.format("YYYY-MM-dd").getInfo()
    except Exception:  # pragma: no cover
        end_str = None

    payload = {
        "dataset": dataset,
        "year": year,
        "month": month,
        "mapId": map_dict["mapid"],
        "token": map_dict["token"],
        "tileUrl": tile_url,
        "expiresAt": expires_at.isoformat(),
        "visParams": vis_params,
        "legend": LEGENDS.get(dataset, []),
        "attribution": "Tiles © Google Earth Engine contributors",
        "scaleMeters": config.get("scale_m"),
        "country": DEFAULT_COUNTRY,
        "month": month,
        "day": day,
        "aggregate": agg_mode,
        "cadence": config.get("cadence"),
        "startDate": start_str,
        "endDate": end_str,
        "unit": config.get("unit"),
    }

    _TILE_CACHE[cache_key] = {
        "payload": payload,
        "expires_at": now + TILE_CACHE_TTL_SECONDS,
    }
    return payload


def sample_dataset_value(
    dataset: str,
    year: int,
    month: int,
    day: Optional[int],
    aggregate: str,
    lat: float,
    lng: float,
) -> Dict[str, Any]:
    """Sample the requested dataset at a geographic point."""
    ensure_ee_initialized()

    agg_mode = (aggregate or "native").lower()
    if agg_mode not in _ALLOWED_AGGREGATES:
        raise TileServiceError(
            f"Unknown aggregate '{aggregate}'. Expected one of {_ALLOWED_AGGREGATES}."
        )

    image, start, end, config = _build_dataset_image(dataset, year, month, day, agg_mode)
    try:
        band_name = image.bandNames().getInfo()[0]
    except Exception:  # pragma: no cover - fallback if bands missing
        band_name = dataset

    point = ee.Geometry.Point([lng, lat])
    try:
        scale_value = float(config.get("scale_m", 1000))
    except (TypeError, ValueError):
        scale_value = 1000.0
    scale_value = max(30.0, scale_value)

    ancillary_images = []
    try:
        landcover_img = ee.Image("ESA/WorldCover/v200/2021").select("Map").rename("landcover")
        ancillary_images.append(landcover_img)
    except Exception:  # pragma: no cover - dataset availability
        landcover_img = None

    try:
        elevation_img = ee.Image("USGS/SRTMGL1_003").select("elevation")
        ancillary_images.append(elevation_img)
    except Exception:  # pragma: no cover
        elevation_img = None

    if ancillary_images:
        sample_image = ee.Image.cat([image] + ancillary_images)
    else:
        sample_image = image

    feature = sample_image.sample(point, scale=scale_value, numPixels=1).first()
    if feature is None:
        raise TileServiceError("No data available at this location.")

    props = feature.getInfo().get("properties", {})
    value = props.get(band_name)

    # Normalise NaN/Inf to None for downstream consumers.
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):  # type: ignore[arg-type]
            value = None

    landcover_code = props.get("landcover") if "landcover" in props else None
    landcover_label = None
    if landcover_code is not None:
        try:
            landcover_int = int(round(float(landcover_code)))
            landcover_label = WORLD_COVER_CLASSES.get(landcover_int)
            landcover_code = landcover_int
        except (TypeError, ValueError):
            landcover_label = None
            landcover_code = None

    elevation_m = props.get("elevation") if "elevation" in props else None
    if isinstance(elevation_m, float) and (math.isnan(elevation_m) or math.isinf(elevation_m)):
        elevation_m = None

    admin_info = _lookup_admin_names(point)

    try:
        start_str = start.format("YYYY-MM-dd").getInfo()
    except Exception:  # pragma: no cover
        start_str = None
    try:
        end_inclusive = end.advance(-1, "day")
        end_str = end_inclusive.format("YYYY-MM-dd").getInfo()
    except Exception:  # pragma: no cover
        end_str = None

    return {
        "dataset": dataset,
        "band": band_name,
        "value": value,
        "unit": config.get("unit"),
        "aggregate": agg_mode,
        "cadence": config.get("cadence"),
        "lat": lat,
        "lng": lng,
        "scaleMeters": config.get("scale_m"),
        "startDate": start_str,
        "endDate": end_str,
        "landCoverCode": landcover_code,
        "landCoverLabel": landcover_label,
        "elevationM": elevation_m,
        "admin1Name": admin_info.get("admin1"),
        "admin2Name": admin_info.get("admin2"),
        "country": DEFAULT_COUNTRY,
    }


def get_download_url(
    dataset: str,
    year: int,
    month: int,
    day: Optional[int],
    aggregate: str,
) -> Dict[str, Any]:
    ensure_ee_initialized()
    agg_mode = (aggregate or "native").lower()
    if agg_mode not in _ALLOWED_AGGREGATES:
        raise TileServiceError(
            f"Unknown aggregate '{aggregate}'. Expected one of {_ALLOWED_AGGREGATES}."
        )

    image, start, end, config = _build_dataset_image(dataset, year, month, day, agg_mode)
    aoi = _country_aoi(DEFAULT_COUNTRY)
    try:
        region = aoi.bounds(1e-3).getInfo()["coordinates"]
    except Exception as exc:  # pragma: no cover - EE failure
        raise TileServiceError("Failed to derive download region bounds.") from exc

    try:
        scale_value = float(config.get("scale_m", 1000))
    except (TypeError, ValueError):
        scale_value = 1000.0
    scale_value = max(30.0, scale_value)

    file_name = f"{dataset}_{year:04d}{month:02d}"
    params = {
        "scale": scale_value,
        "crs": "EPSG:4326",
        "region": region,
        "fileFormat": "GEO_TIFF",
        "format": "GEO_TIFF",
        "description": file_name,
    }

    try:
        url = image.getDownloadURL(params)
    except Exception as exc:  # pragma: no cover
        raise TileServiceError("Failed to generate download URL from Earth Engine.") from exc

    try:
        start_str = start.format("YYYY-MM-dd").getInfo()
    except Exception:
        start_str = None
    try:
        end_inclusive = end.advance(-1, "day")
        end_str = end_inclusive.format("YYYY-MM-dd").getInfo()
    except Exception:
        end_str = None

    return {
        "url": url,
        "fileName": f"{file_name}.tif",
        "aggregate": agg_mode,
        "startDate": start_str,
        "endDate": end_str,
    }


__all__ = [
    "TileServiceError",
    "ensure_ee_initialized",
    "get_tile_metadata",
    "list_datasets",
    "current_auth_status",
    "sample_dataset_value",
    "set_service_account_key",
    "get_download_url",
]
