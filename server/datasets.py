"""Dataset configuration and visualization parameters for the GEE tile service."""

from __future__ import annotations

from typing import Dict, List, TypedDict


class DatasetConfig(TypedDict, total=False):
    type: str
    ee_id: str
    band: str
    cadence: str
    agg: str
    scale_m: int
    description: str
    source: str
    unit: str


DATASETS: Dict[str, DatasetConfig] = {
    "chirps_precip": {
        "type": "collection",
        "ee_id": "UCSB-CHG/CHIRPS/DAILY",
        "band": "precipitation",
        "cadence": "DAILY",
        "agg": "sum",
        "scale_m": 5000,
        "description": "CHIRPS daily precipitation aggregated over the selected period.",
        "unit": "mm",
    },
    "era5_temp2m": {
        "type": "collection",
        "ee_id": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "band": "temperature_2m",
        "cadence": "MONTHLY",
        "agg": "mean",
        "scale_m": 11132,
        "description": "ERA5-Land monthly aggregated 2m air temperature.",
        "unit": "°C",
    },
    "era5_soil_moisture": {
        "type": "collection",
        "ee_id": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "band": "volumetric_soil_water_layer_1",
        "cadence": "MONTHLY",
        "agg": "mean",
        "scale_m": 11132,
        "description": "ERA5-Land monthly aggregated soil moisture layer 1.",
        "unit": "m³/m³",
    },
    "modis_ndvi": {
        "type": "collection",
        "ee_id": "MODIS/061/MOD13A3",
        "band": "NDVI",
        "cadence": "MONTHLY",
        "agg": "mean",
        "scale_m": 1000,
        "description": "MODIS monthly NDVI (scale 1e-4).",
        "unit": "Index (unitless)",
    },
    "modis_evi": {
        "type": "collection",
        "ee_id": "MODIS/061/MOD13A3",
        "band": "EVI",
        "cadence": "MONTHLY",
        "agg": "mean",
        "scale_m": 1000,
        "description": "MODIS monthly EVI (scale 1e-4).",
        "unit": "Index (unitless)",
    },
    "modis_lst_day": {
        "type": "collection",
        "ee_id": "MODIS/061/MOD11A2",
        "band": "LST_Day_1km",
        "cadence": "8D",
        "agg": "mean",
        "scale_m": 1000,
        "description": "MODIS 8-day LST Day aggregated over the selected period (×0.02 K).",
        "unit": "°C",
    },
    "modis_lst_night": {
        "type": "collection",
        "ee_id": "MODIS/061/MOD11A2",
        "band": "LST_Night_1km",
        "cadence": "8D",
        "agg": "mean",
        "scale_m": 1000,
        "description": "MODIS 8-day LST Night aggregated over the selected period (×0.02 K).",
        "unit": "°C",
    },
    "era5_vpd": {
        "type": "derived_vpd",
        "source": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "cadence": "MONTHLY",
        "scale_m": 11132,
        "description": "Monthly VPD (kPa) from ERA5-Land 2m temperature and dew point.",
        "unit": "kPa",
    },
}


VIS_PARAMS: Dict[str, Dict[str, object]] = {
    "modis_ndvi": {
        "min": -0.2,
        "max": 0.9,
        "palette": ["#b4c8dc", "#d2b48c", "#c8c864", "#78c864", "#32b432", "#147814"],
    },
    "modis_evi": {
        "min": -0.2,
        "max": 0.9,
        "palette": ["#c8b4a0", "#dcc878", "#b4dc64", "#64c864", "#28a03c", "#00641e"],
    },
    "chirps_precip": {
        "min": 0,
        "max": 200,
        "palette": ["#fffac8", "#c8e6ff", "#78b4ff", "#3c78e6", "#1e3cb4", "#0a1464"],
    },
    "era5_temp2m": {
        "min": -5,
        "max": 45,
        "palette": ["#9696ff", "#64c8ff", "#64ffc8", "#c8ff64", "#ffc864", "#ff9650", "#c83232"],
    },
    "era5_soil_moisture": {
        "min": 0.0,
        "max": 0.5,
        "palette": ["#8b5a2b", "#d2b48c", "#c8dcb4", "#96c8dc", "#6496c8", "#3264b4"],
    },
    "era5_vpd": {
        "min": 0.0,
        "max": 3.0,
        "palette": ["#6496ff", "#96c8c8", "#c8dc96", "#ffc864", "#ff9650", "#c85032"],
    },
    "modis_lst_day": {
        "min": -5,
        "max": 55,
        "palette": ["#9696ff", "#64c8ff", "#96ff96", "#ffff64", "#ff9650", "#c83232"],
    },
    "modis_lst_night": {
        "min": -10,
        "max": 35,
        "palette": ["#9696ff", "#78b4ff", "#96c8c8", "#c8dcb4", "#ffdca0", "#ffb478"],
    },
}


LEGENDS: Dict[str, List[Dict[str, str]]] = {
    "modis_ndvi": [
        {"color": "rgb(20, 120, 20)", "label": "Dense vegetation (> 0.8)"},
        {"color": "rgb(50, 180, 50)", "label": "Moderate-dense (0.6-0.8)"},
        {"color": "rgb(120, 200, 100)", "label": "Moderate (0.4-0.6)"},
        {"color": "rgb(200, 200, 100)", "label": "Sparse (0.2-0.4)"},
        {"color": "rgb(210, 180, 140)", "label": "Very sparse (0-0.2)"},
        {"color": "rgb(180, 200, 220)", "label": "Water/Barren (< 0)"},
    ],
    "modis_evi": [
        {"color": "rgb(0, 100, 30)", "label": "Very dense (> 0.8)"},
        {"color": "rgb(40, 160, 60)", "label": "Dense (0.6-0.8)"},
        {"color": "rgb(100, 200, 100)", "label": "Moderate (0.4-0.6)"},
        {"color": "rgb(180, 220, 100)", "label": "Sparse (0.2-0.4)"},
        {"color": "rgb(220, 200, 120)", "label": "Very sparse (0-0.2)"},
        {"color": "rgb(200, 180, 160)", "label": "Barren (< 0)"},
    ],
    "chirps_precip": [
        {"color": "rgb(10, 20, 100)", "label": "Extreme (> 200mm)"},
        {"color": "rgb(30, 60, 180)", "label": "Very heavy (100-200mm)"},
        {"color": "rgb(60, 120, 230)", "label": "Heavy (60-100mm)"},
        {"color": "rgb(120, 180, 255)", "label": "Moderate (30-60mm)"},
        {"color": "rgb(200, 230, 255)", "label": "Light (10-30mm)"},
        {"color": "rgb(255, 250, 200)", "label": "Very dry (< 10mm)"},
    ],
    "era5_temp2m": [
        {"color": "rgb(200, 50, 50)", "label": "Very hot (> 35°C)"},
        {"color": "rgb(255, 150, 80)", "label": "Hot (30-35°C)"},
        {"color": "rgb(255, 200, 100)", "label": "Warm (25-30°C)"},
        {"color": "rgb(200, 255, 100)", "label": "Mild (20-25°C)"},
        {"color": "rgb(100, 255, 200)", "label": "Cool (10-20°C)"},
        {"color": "rgb(100, 200, 255)", "label": "Cold (0-10°C)"},
        {"color": "rgb(150, 150, 255)", "label": "Freezing (< 0°C)"},
    ],
    "era5_soil_moisture": [
        {"color": "rgb(50, 100, 180)", "label": "Saturated (> 0.5)"},
        {"color": "rgb(100, 150, 200)", "label": "Wet (0.4-0.5)"},
        {"color": "rgb(150, 200, 220)", "label": "Moist (0.3-0.4)"},
        {"color": "rgb(200, 220, 180)", "label": "Moderate (0.2-0.3)"},
        {"color": "rgb(210, 180, 140)", "label": "Dry (0.1-0.2)"},
        {"color": "rgb(139, 90, 43)", "label": "Very dry (< 0.1)"},
    ],
    "era5_vpd": [
        {"color": "rgb(200, 80, 50)", "label": "Extreme (> 3.0 kPa)"},
        {"color": "rgb(255, 150, 80)", "label": "Very high (2.0-3.0 kPa)"},
        {"color": "rgb(255, 200, 100)", "label": "High (1.5-2.0 kPa)"},
        {"color": "rgb(200, 220, 150)", "label": "Moderate (1.0-1.5 kPa)"},
        {"color": "rgb(150, 200, 200)", "label": "Slight (0.5-1.0 kPa)"},
        {"color": "rgb(100, 150, 255)", "label": "Low stress (< 0.5 kPa)"},
    ],
    "modis_lst_day": [
        {"color": "rgb(200, 50, 50)", "label": "Very hot (> 45°C)"},
        {"color": "rgb(255, 150, 80)", "label": "Hot (35-45°C)"},
        {"color": "rgb(255, 255, 100)", "label": "Warm (25-35°C)"},
        {"color": "rgb(150, 255, 150)", "label": "Mild (15-25°C)"},
        {"color": "rgb(100, 200, 255)", "label": "Cool (5-15°C)"},
        {"color": "rgb(150, 150, 255)", "label": "Cold (< 5°C)"},
    ],
    "modis_lst_night": [
        {"color": "rgb(255, 180, 120)", "label": "Warm (> 25°C)"},
        {"color": "rgb(255, 220, 150)", "label": "Mild (20-25°C)"},
        {"color": "rgb(200, 220, 180)", "label": "Cool (15-20°C)"},
        {"color": "rgb(150, 200, 200)", "label": "Chilly (10-15°C)"},
        {"color": "rgb(120, 180, 255)", "label": "Cold (0-10°C)"},
        {"color": "rgb(150, 150, 255)", "label": "Freezing (< 0°C)"},
    ],
}


DEFAULT_COUNTRY = "Morocco"
TILE_CACHE_TTL_SECONDS = 55 * 60  # cache entries for just under an hour
