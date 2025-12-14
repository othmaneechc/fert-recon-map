"""Local Fertimap surrogate models (no remote Fertimap calls)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
import json
import re
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

# Local training data and model artefacts.
DATA_PATH = Path(__file__).resolve().parents[1] / "public" / "data" / "fertimap_grid.csv"
MODEL_PATH = Path(__file__).resolve().parent / "fertimap_rf_models.joblib"
PROVINCE_MAP_PATH = Path(__file__).resolve().parent / "fertimap_provinces.json"

NUMERIC_FEATURES = [
    "lat",
    "lon",
    "ph",
    "organic_matter_pct",
    "p2o5_mgkg",
    "k2o_mgkg",
    "scenario_expected_yield",
]
CATEGORICAL_FEATURES = [
    "soil_type",
    "texture",
    "target_crop_name",
]
TARGET_COLUMNS = {
    "rec_n_kg_ha": "N",
    "rec_p_kg_ha": "P",
    "rec_k_kg_ha": "K",
    "rec_cost_dh_ha": "cost",
}
FERTILIZER_PRODUCTS = [
    {"name": "NPK(10.20.20)", "mode": "fond"},
    {"name": "Ammonitrates", "mode": "couverture"},
    {"name": "TSP", "mode": "fond"},
    {"name": "NPK(16.11.20)", "mode": "fond"},
]
FOND_PRODUCT_COL = "regional_fond_product"
FOND_QTY_COL = "regional_fond_qty_qx_ha"
COUV_PRODUCT_COL = "regional_couverture_product"
COUV_QTY_COL = "regional_couverture_qty_qx_ha"
TRAIN_SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

class FertimapServiceError(Exception):
    """Raised when the Fertimap service cannot fulfil a request."""


_MODEL_CACHE: Optional[Dict[str, Any]] = None
_PROVINCE_MAP: Dict[int, Dict[str, str]] = {}
_PARSED_CACHE: Optional[pd.DataFrame] = None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if pd.isna(value):  # type: ignore[arg-type]
            return None
        return int(value)
    except Exception:
        return None


def _load_province_map() -> Dict[int, Dict[str, str]]:
    global _PROVINCE_MAP
    if _PROVINCE_MAP:
        return _PROVINCE_MAP
    if PROVINCE_MAP_PATH.exists():
        try:
            data = json.loads(PROVINCE_MAP_PATH.read_text(encoding="utf-8"))
            _PROVINCE_MAP = {int(k): v for k, v in data.items()}
            return _PROVINCE_MAP
        except Exception as exc:
            logger.warning("Failed to load province map: %s", exc)
    _PROVINCE_MAP = {}
    return _PROVINCE_MAP


@lru_cache(maxsize=1)
def _grid_df() -> pd.DataFrame:
    global _PARSED_CACHE
    if _PARSED_CACHE is not None:
        return _PARSED_CACHE
    if not DATA_PATH.exists():
        raise FertimapServiceError(
            f"Local Fertimap grid data not found at {DATA_PATH}. Populate the CSV before serving."
        )
    df = pd.read_csv(DATA_PATH)
    df = _augment_parsed_columns(df)
    _PARSED_CACHE = df
    return df


@lru_cache(maxsize=1)
def _grid_coords() -> np.ndarray:
    df = _grid_df()
    return df[["lat", "lon"]].to_numpy()


def _nearest_index(lat: float, lon: float) -> int:
    coords = _grid_coords()
    target = np.array([lat, lon])
    distances = np.square(coords - target).sum(axis=1)
    return int(np.argmin(distances))


def _nearest_row(lat: float, lon: float) -> pd.Series:
    df = _grid_df()
    idx = _nearest_index(lat, lon)
    return df.iloc[idx]


PRODUCT_LINE_RE = re.compile(
    r"([0-9]+(?:\\.[0-9]+)?)\\s*qx/ha\\s+d['u]?\\s*([^\\n\\r]+?)\\s+comme\\s+engrais\\s+de\\s+(fond|couverture)",
    re.IGNORECASE,
)


def _parse_recommendation_lines(text: Optional[str]) -> list[dict[str, object]]:
    if not isinstance(text, str) or not text:
        return []
    lines: list[dict[str, object]] = []
    for match in PRODUCT_LINE_RE.findall(text):
        qty_str, product, mode = match
        try:
            qty = float(qty_str)
        except ValueError:
            continue
        product = product.strip().replace("\\xa0", " ")
        mode = mode.strip().lower()
        lines.append({"product": product, "mode": mode, "qty": qty})
    return lines


def _augment_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse regional recommendation text into fond/couverture targets if present."""
    df = df.copy()
    fond_products: list[Optional[str]] = []
    fond_qty: list[Optional[float]] = []
    couv_products: list[Optional[str]] = []
    couv_qty: list[Optional[float]] = []

    texts = df.get("regional_recommendations")
    if texts is None:
        df[FOND_PRODUCT_COL] = None
        df[FOND_QTY_COL] = None
        df[COUV_PRODUCT_COL] = None
        df[COUV_QTY_COL] = None
        return df

    for text in texts:
        parsed = _parse_recommendation_lines(text)
        fond = next((item for item in parsed if item["mode"] == "fond"), None)
        couverture = next((item for item in parsed if item["mode"] == "couverture"), None)
        fond_products.append(fond["product"] if fond else None)
        fond_qty.append(fond["qty"] if fond else None)
        couv_products.append(couverture["product"] if couverture else None)
        couv_qty.append(couverture["qty"] if couverture else None)

    df[FOND_PRODUCT_COL] = fond_products
    df[FOND_QTY_COL] = fond_qty
    df[COUV_PRODUCT_COL] = couv_products
    df[COUV_QTY_COL] = couv_qty
    return df


@lru_cache(maxsize=1)
def get_crop_mapping() -> Dict[str, int]:
    df = _grid_df()
    crops = {}
    crop_df = (
        df.loc[df["target_crop_name"].notna(), ["target_crop_name", "target_crop_id"]]
        .drop_duplicates()
        .sort_values("target_crop_id")
    )
    for _, row in crop_df.iterrows():
        name = _safe_str(row["target_crop_name"])
        crop_id = row.get("target_crop_id")
        crops[name] = int(crop_id) if pd.notna(crop_id) else len(crops) + 1
    if not crops:
        raise FertimapServiceError("No crop metadata available in the local grid.")
    return crops


def get_location_defaults(lon: float, lat: float) -> Dict[str, Any]:
    """Return the closest soil defaults from the precomputed Fertimap grid."""
    row = _nearest_row(lat, lon)
    province_map = _load_province_map()
    province_id = _safe_int(row.get("province_id"))
    province_info = province_map.get(province_id or -1, {})
    defaults: Dict[str, Any] = {
        "ph": _safe_float(row.get("ph")),
        "mo": _safe_float(row.get("organic_matter_pct")),
        "p": _safe_float(row.get("p2o5_mgkg")),
        "k": _safe_float(row.get("k2o_mgkg")),
        "crop_id": _safe_int(row.get("target_crop_id")),
        "crop_name": _safe_str(row.get("target_crop_name")) or None,
        "yield_min": _safe_float(row.get("target_yield_min")),
        "yield_max": _safe_float(row.get("target_yield_max")),
        "yield_step": _safe_float(row.get("target_yield_step")),
        "yield_unit": _safe_str(row.get("target_yield_unit")),
        "expected_yield": _safe_float(row.get("scenario_expected_yield")),
        "province_id": province_id,
        "province_name": province_info.get("province", ""),
    }
    return defaults


def _build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    reg = RandomForestRegressor(
        n_estimators=40,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("reg", reg)])


def _build_classifier() -> Pipeline:
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def _prepare_training_frame() -> pd.DataFrame:
    df = _grid_df().copy()
    train_df = df.dropna(subset=list(TARGET_COLUMNS.keys()))
    if len(train_df) > TRAIN_SAMPLE_SIZE:
        train_df = train_df.sample(TRAIN_SAMPLE_SIZE, random_state=RANDOM_STATE)
    return train_df


def train_local_models(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    """Train RandomForest regressors on the grid and persist them for serving."""
    train_df = _prepare_training_frame()
    if train_df.empty:
        raise FertimapServiceError("Training data is empty; cannot train Fertimap surrogate models.")

    feature_frame = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    models: Dict[str, Pipeline] = {}
    metrics: Dict[str, Any] = {
        "rows": len(train_df),
        "sampled": len(train_df) < len(_grid_df()),
        "sklearn_version": sklearn.__version__,
    }

    # Nutrient regressors
    for target in TARGET_COLUMNS.keys():
        pipeline = _build_pipeline()
        pipeline.fit(feature_frame, train_df[target])
        models[target] = pipeline

    # Fond/couverture classifiers & regressors from parsed text
    parsed_df = _grid_df()
    fond_df = parsed_df.dropna(subset=[FOND_PRODUCT_COL]).copy()
    if not fond_df.empty:
        fond_features = fond_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        fond_clf = _build_classifier()
        fond_clf.fit(fond_features, fond_df[FOND_PRODUCT_COL])
        models["fond_product"] = fond_clf
        fond_qty_df = fond_df.dropna(subset=[FOND_QTY_COL])
        if not fond_qty_df.empty:
            fond_reg = _build_pipeline()
            fond_reg.fit(fond_qty_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], fond_qty_df[FOND_QTY_COL])
            models["fond_qty"] = fond_reg

    couv_df = parsed_df.dropna(subset=[COUV_PRODUCT_COL]).copy()
    if not couv_df.empty:
        couv_features = couv_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        couv_clf = _build_classifier()
        couv_clf.fit(couv_features, couv_df[COUV_PRODUCT_COL])
        models["couv_product"] = couv_clf
        couv_qty_df = couv_df.dropna(subset=[COUV_QTY_COL])
        if not couv_qty_df.empty:
            couv_reg = _build_pipeline()
            couv_reg.fit(couv_qty_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], couv_qty_df[COUV_QTY_COL])
            models["couv_qty"] = couv_reg

    bundle = {
        "models": models,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "targets": TARGET_COLUMNS,
        "training_rows": len(train_df),
        "metadata": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    logger.info("Trained Fertimap surrogate models and saved to %s", model_path)
    global _MODEL_CACHE
    _MODEL_CACHE = bundle
    return bundle


def _load_model_bundle() -> Dict[str, Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if MODEL_PATH.exists():
        try:
            bundle = joblib.load(MODEL_PATH)
            metadata = bundle.get("metadata", {})
            saved_version = metadata.get("sklearn_version")
            if not saved_version or saved_version != sklearn.__version__:
                logger.info(
                    "Reloading Fertimap models because sklearn version changed (saved %s, current %s).",
                    saved_version,
                    sklearn.__version__,
                )
                bundle = train_local_models()
            _MODEL_CACHE = bundle
            return _MODEL_CACHE
        except Exception as exc:
            logger.warning("Failed to load Fertimap model bundle; retraining. Error: %s", exc)
            _MODEL_CACHE = train_local_models()
            return _MODEL_CACHE
    _MODEL_CACHE = train_local_models()
    return _MODEL_CACHE


def predict_recommendation(features: Dict[str, Any]) -> Dict[str, float]:
    bundle = _load_model_bundle()
    models: Dict[str, Pipeline] = bundle["models"]

    row = {**{col: np.nan for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES}, **features}
    input_df = pd.DataFrame([row])

    predictions: Dict[str, float] = {}
    for target, alias in TARGET_COLUMNS.items():
        model = models.get(target)
        if model is None:
            continue
        try:
            value = float(model.predict(input_df)[0])
        except AttributeError:
            logger.warning("Model predict failed due to version mismatch; retraining Fertimap models.")
            bundle = train_local_models()
            model = bundle["models"].get(target)
            if model is None:
                continue
            value = float(model.predict(input_df)[0])
        predictions[alias] = round(value, 3)
    return predictions


def _predict_product_and_qty(models: Dict[str, Pipeline], input_df: pd.DataFrame, kind: str) -> tuple[str | None, float | None]:
    product_model = models.get(f"{kind}_product")
    qty_model = models.get(f"{kind}_qty")
    product = None
    qty = None
    try:
        if product_model is not None:
            product = str(product_model.predict(input_df)[0])
    except Exception as exc:
        logger.debug("Failed to predict %s product: %s", kind, exc)
    try:
        if qty_model is not None:
            qty = float(qty_model.predict(input_df)[0])
    except Exception as exc:
        logger.debug("Failed to predict %s quantity: %s", kind, exc)
    return product, qty


def complete_recommendation(
    lon: float,
    lat: float,
    ph: float,
    mo: float,
    p: float,
    k: float,
    crop_name: str,
    expected_yield: float,
) -> Dict[str, Any]:
    """Predict N/P/K/cost using local RF models and nearest soil descriptors."""
    nearest = _nearest_row(lat, lon)
    crop_mapping = get_crop_mapping()
    if crop_mapping and crop_name not in crop_mapping:
        raise FertimapServiceError(f"Unsupported crop '{crop_name}'.")

    model_features = {
        "lat": lat,
        "lon": lon,
        "ph": ph,
        "organic_matter_pct": mo,
        "p2o5_mgkg": p,
        "k2o_mgkg": k,
        "scenario_expected_yield": expected_yield,
        "soil_type": _safe_str(nearest.get("soil_type")),
        "texture": _safe_str(nearest.get("texture")),
        "target_crop_name": crop_name or _safe_str(nearest.get("target_crop_name")),
    }

    bundle = _load_model_bundle()
    models: Dict[str, Pipeline] = bundle["models"]

    row = {**{col: np.nan for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES}, **model_features}
    input_df = pd.DataFrame([row])

    predictions = predict_recommendation(model_features)
    province_map = _load_province_map()
    province_id = _safe_int(nearest.get("province_id"))
    province_info = province_map.get(province_id or -1, {})

    location = {
        "province_id": province_id,
        "region": province_info.get("region") or _safe_str(nearest.get("region")),
        "province": province_info.get("province") or _safe_str(nearest.get("province")),
        "commune": _safe_str(nearest.get("commune")),
        "soil_type": _safe_str(nearest.get("soil_type")),
        "texture": _safe_str(nearest.get("texture")),
    }

    recommendation = {
        "N": predictions.get("N"),
        "P": predictions.get("P"),
        "K": predictions.get("K"),
        "cost": predictions.get("cost"),
        "recommendations": {"regional": [], "selected_yield": [], "generic": []},
        "source": "rf_model",
    }

    def format_line(qty: float, product: str, mode: str) -> Dict[str, object]:
        return {
            "quantity": max(round(qty, 3), 0.0),
            "name": product,
            "type": f"engrais de {mode}" if mode else "",
        }

    fond_product, fond_qty = _predict_product_and_qty(models, input_df, "fond")
    couv_product, couv_qty = _predict_product_and_qty(models, input_df, "couv")

    lines: Dict[str, Dict[str, object]] = {}
    if fond_product:
        lines["fond"] = format_line(fond_qty or 0.0, fond_product, "fond")
    if couv_product:
        lines["couverture"] = format_line(couv_qty or 0.0, couv_product, "couverture")

    # Fallback heuristic if models did not return any product
    if not lines:
        n = predictions.get("N") or 0.0
        p = predictions.get("P") or 0.0
        k = predictions.get("K") or 0.0
        if p > 0:
            lines["fond"] = format_line(max((p / 46.0) * 100.0, 0.0), "TSP", "fond")
        if n > 0:
            lines["couverture"] = format_line(max(n / 33.5, 0.0), "Ammonitrates", "couverture")

    # Generic section mirrors fond/couverture for now
    recommendation["recommendations"]["regional"] = [lines.get("fond")] if "fond" in lines else []
    recommendation["recommendations"]["selected_yield"] = [
        l for l in (lines.get("fond"), lines.get("couverture")) if l
    ]
    recommendation["recommendations"]["generic"] = [
        l for l in (lines.get("fond"), lines.get("couverture")) if l
    ]

    return {"location": location, "recommendation": recommendation}
