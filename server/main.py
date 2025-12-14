"""FastAPI application exposing GEE tile metadata."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from .gee_service import (
    TileServiceError,
    ensure_ee_initialized,
    get_tile_metadata,
    list_datasets,
    current_auth_status,
    get_download_url,
    set_service_account_key,
    sample_dataset_value,
)
from .fertimap_service import (
    FertimapServiceError,
    complete_recommendation as fertimap_complete_recommendation,
    get_crop_mapping,
    get_location_defaults,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fert Recon - GEE Tile Service",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Open up for local dev; production deployments can tighten this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ServiceAccountPayload(BaseModel):
    key: str
    model_config = ConfigDict(extra="forbid")


class FertimapRequest(BaseModel):
    lat: float
    lng: float
    ph: float
    mo: float
    p: float
    k: float
    crop_name: str
    expected_yield: float
    model_config = ConfigDict(extra="forbid")


@app.on_event("startup")
def _startup() -> None:  # pragma: no cover - depends on EE credentials
    try:
        ensure_ee_initialized()
        logger.info("Earth Engine client initialised successfully.")
    except TileServiceError as exc:
        logger.warning("Earth Engine initialisation deferred: %s", exc)


api = APIRouter(prefix="/api")


@api.get("/health", response_model=Dict[str, Any])
def api_health() -> Dict[str, Any]:
    """Simple health check endpoint."""
    return {"ok": True}


@api.get("/datasets", response_model=Dict[str, Any])
def api_datasets() -> Dict[str, Any]:
    """Return dataset metadata for clients."""
    return list_datasets()


@api.get("/tiles", response_model=Dict[str, Any])
def api_tiles(
    dataset: str = Query(..., description="Dataset key, e.g. 'modis_ndvi'."),
    year: int = Query(..., ge=1980, le=2100, description="4-digit year."),
    month: int = Query(..., ge=1, le=12, description="Month number (1-12)."),
    day: int | None = Query(
        None,
        ge=1,
        le=31,
        description="Optional day-of-month for sub-monthly datasets.",
    ),
    aggregate: str = Query(
        "native",
        description="Aggregation level: native, monthly, or yearly.",
    ),
) -> Dict[str, Any]:
    """Return (and cache) tile metadata for a dataset/year/month selection."""
    try:
        return get_tile_metadata(dataset, year, month, day, aggregate)
    except TileServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - pass-through for diagnostics
        logger.exception("Unexpected error while building tile metadata.")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@api.get("/sample", response_model=Dict[str, Any])
def api_sample(
    dataset: str = Query(..., description="Dataset key, e.g. 'modis_ndvi'."),
    year: int = Query(..., ge=1980, le=2100, description="4-digit year."),
    month: int = Query(..., ge=1, le=12, description="Month number (1-12)."),
    day: int | None = Query(
        None,
        ge=1,
        le=31,
        description="Optional day-of-month for sub-monthly datasets.",
    ),
    aggregate: str = Query(
        "native",
        description="Aggregation level: native, monthly, or yearly.",
    ),
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude in degrees."),
    lng: float = Query(..., ge=-180.0, le=180.0, description="Longitude in degrees."),
) -> Dict[str, Any]:
    """Sample the requested dataset at the provided lat/lng point."""
    try:
        return sample_dataset_value(dataset, year, month, day, aggregate, lat, lng)
    except TileServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - diagnostics
        logger.exception("Unexpected error while sampling dataset value.")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@api.get("/fertimap/crops", response_model=Dict[str, Any])
def api_fertimap_crops() -> Dict[str, Any]:
    try:
        crop_mapping = get_crop_mapping()
    except FertimapServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    crops = [{"name": name, "id": crop_id} for name, crop_id in crop_mapping.items()]
    return {"crops": crops}


@api.post("/fertimap/recommendation", response_model=Dict[str, Any])
def api_fertimap_recommendation(payload: FertimapRequest) -> Dict[str, Any]:
    try:
        return fertimap_complete_recommendation(
            lon=payload.lng,
            lat=payload.lat,
            ph=payload.ph,
            mo=payload.mo,
            p=payload.p,
            k=payload.k,
            crop_name=payload.crop_name,
            expected_yield=payload.expected_yield,
        )
    except FertimapServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - diagnostics
        logger.exception("Unexpected error while fetching Fertimap recommendation.")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@api.get("/fertimap/defaults", response_model=Dict[str, Any])
def api_fertimap_defaults(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude in degrees."),
    lng: float = Query(..., ge=-180.0, le=180.0, description="Longitude in degrees."),
) -> Dict[str, Any]:
    """Fetch default soil analysis values for a clicked location."""
    try:
        defaults = get_location_defaults(lon=lng, lat=lat)
        return {"defaults": defaults}
    except FertimapServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - diagnostics
        logger.exception("Unexpected error while fetching Fertimap defaults.")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@api.post("/auth/service-account", response_model=Dict[str, Any])
def api_upload_service_account(payload: ServiceAccountPayload) -> Dict[str, Any]:
    try:
        result = set_service_account_key(payload.key)
        return {"ok": True, **result}
    except TileServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - diagnostics
        logger.exception("Unexpected error while uploading service account key.")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@api.get("/auth/status", response_model=Dict[str, Any])
def api_auth_status() -> Dict[str, Any]:
    return current_auth_status()


@api.get("/download", response_model=Dict[str, Any])
def api_download(
    dataset: str = Query(..., description="Dataset key, e.g. 'modis_ndvi'."),
    year: int = Query(..., ge=1980, le=2100, description="4-digit year."),
    month: int = Query(..., ge=1, le=12, description="Month number (1-12)."),
    day: int | None = Query(
        None,
        ge=1,
        le=31,
        description="Optional day-of-month for sub-monthly datasets.",
    ),
    aggregate: str = Query(
        "native",
        description="Aggregation level: native, monthly, or yearly.",
    ),
) -> Dict[str, Any]:
    try:
        return get_download_url(dataset, year, month, day, aggregate)
    except TileServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error while preparing download URL.")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@app.get("/health", response_model=Dict[str, Any])
def root_health() -> Dict[str, Any]:
    """Backward-compatible health endpoint at root."""
    return api_health()


@app.get("/datasets", response_model=Dict[str, Any])
def root_datasets() -> Dict[str, Any]:
    """Backward-compatible dataset metadata endpoint."""
    return api_datasets()


@app.get("/tiles", response_model=Dict[str, Any])
def root_tiles(
    dataset: str = Query(..., description="Dataset key, e.g. 'modis_ndvi'."),
    year: int = Query(..., ge=1980, le=2100, description="4-digit year."),
    month: int = Query(..., ge=1, le=12, description="Month number (1-12)."),
    day: int | None = Query(
        None,
        ge=1,
        le=31,
        description="Optional day-of-month for sub-monthly datasets.",
    ),
    aggregate: str = Query(
        "native",
        description="Aggregation level: native, monthly, or yearly.",
    ),
) -> Dict[str, Any]:
    """Backward-compatible tile endpoint without prefix."""
    return api_tiles(
        dataset=dataset,
        year=year,
        month=month,
        day=day,
        aggregate=aggregate,
    )


@app.get("/sample", response_model=Dict[str, Any])
def root_sample(
    dataset: str = Query(..., description="Dataset key, e.g. 'modis_ndvi'."),
    year: int = Query(..., ge=1980, le=2100, description="4-digit year."),
    month: int = Query(..., ge=1, le=12, description="Month number (1-12)."),
    day: int | None = Query(
        None,
        ge=1,
        le=31,
        description="Optional day-of-month for sub-monthly datasets.",
    ),
    aggregate: str = Query(
        "native",
        description="Aggregation level: native, monthly, or yearly.",
    ),
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude in degrees."),
    lng: float = Query(..., ge=-180.0, le=180.0, description="Longitude in degrees."),
) -> Dict[str, Any]:
    """Backward-compatible sampling endpoint without prefix."""
    return api_sample(
        dataset=dataset,
        year=year,
        month=month,
        day=day,
        aggregate=aggregate,
        lat=lat,
        lng=lng,
    )


@app.post("/auth/service-account", response_model=Dict[str, Any])
def root_upload_service_account(payload: ServiceAccountPayload) -> Dict[str, Any]:
    return api_upload_service_account(payload)


@app.get("/auth/status", response_model=Dict[str, Any])
def root_auth_status() -> Dict[str, Any]:
    return api_auth_status()


@app.get("/download", response_model=Dict[str, Any])
def root_download(
    dataset: str = Query(..., description="Dataset key, e.g. 'modis_ndvi'."),
    year: int = Query(..., ge=1980, le=2100, description="4-digit year."),
    month: int = Query(..., ge=1, le=12, description="Month number (1-12)."),
    day: int | None = Query(
        None,
        ge=1,
        le=31,
        description="Optional day-of-month for sub-monthly datasets.",
    ),
    aggregate: str = Query(
        "native",
        description="Aggregation level: native, monthly, or yearly.",
    ),
) -> Dict[str, Any]:
    return api_download(dataset=dataset, year=year, month=month, day=day, aggregate=aggregate)


app.include_router(api)
