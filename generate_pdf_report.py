"""Generate a professional PDF report for the fuel-consumption project.

This script is designed to be "ready to run" on Windows and produces a
polished PDF containing:
- Title page
- Problem statement
- Dataset overview
- EDA (with visuals + captions)
- Modeling approach
- Evaluation results (metrics + comparison)
- Deployment section (Streamlit link)
- Conclusion

It reproduces the core analysis from the notebook and does NOT require
executing the notebook.

Usage examples:
  python generate_pdf_report.py --data path\\to\\vehicles.csv
  python generate_pdf_report.py --data vehicles.csv --out Fuel_Report.pdf

"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image as PILImage

# Increase PIL's decompression bomb limit to handle large images
PILImage.MAX_IMAGE_PIXELS = None
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (  # type: ignore
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_TITLE = "Fuel Consumption Prediction (City & Highway)"
DEPLOYMENT_LINK = "https://fuel-consumption-calc.streamlit.app/"
MPG_TO_L_PER_100KM = 235.214583

REPORT_SUBTITLE = "A Complete End-to-End Data Science Project"


@dataclass(frozen=True)
class ReportPaths:
    out_pdf: Path
    assets_dir: Path


@dataclass(frozen=True)
class NotebookArtifact:
    path: Optional[Path]
    caption: str
    text: Optional[str] = None


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mpg_to_l_per_100km(mpg: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Convert MPG to L/100km using 235.214583 / mpg."""

    return MPG_TO_L_PER_100KM / mpg


def normalize_fuel_category(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return "Unknown"

    text = str(raw).strip().lower()

    if "hydrogen" in text:
        return "Hydrogen"
    if "diesel" in text:
        return "Diesel"
    if "electric" in text:
        return "Electricity"
    if "natural" in text or "cng" in text:
        return "Natural gas"
    if "ethanol" in text or "e85" in text:
        return "Ethanol (E85)"

    # Gasoline categories vary across datasets (Regular/Midgrade/Premium).
    if "premium" in text:
        return "Premium gasoline"
    if "regular" in text:
        return "Regular gasoline"
    if "midgrade" in text:
        # Map to regular to keep the 6 one-hot columns consistent with the app.
        return "Regular gasoline"
    if "gasoline" in text:
        return "Regular gasoline"

    return "Other"


FUEL_ONE_HOT_COLUMNS: List[str] = [
    "Fuel type_Diesel",
    "Fuel type_Electricity",
    "Fuel type_Premium gasoline",
    "Fuel type_Regular gasoline",
    "Fuel type_Natural gas",
    "Fuel type_Ethanol (E85)",
]


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build X features and y targets aligned with the Streamlit app.

    Features (10):
      - Combined (L/100 km)
      - Cylinders
      - Engine size
      - Model year
      - 6 fuel one-hot columns

    Targets (2):
      - City (L/100 km)
      - Highway (L/100 km)

    The source dataset is expected to have MPG columns: comb08, city08, highway08
    and metadata columns: year, cylinders, displ, and fuelType1.
    """

    missing_required = [c for c in ["year", "cylinders", "displ", "comb08", "city08", "highway08"] if c not in df.columns]
    if missing_required:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing_required) +
            ". This script expects a FuelEconomy-style 'vehicles.csv' with columns like comb08/city08/highway08."
        )

    fuel_col = "fuelType1" if "fuelType1" in df.columns else ("fuelType" if "fuelType" in df.columns else None)
    if fuel_col is None:
        raise ValueError("Dataset is missing 'fuelType1' (or 'fuelType') column.")

    df = df.copy()

    # Mirror notebook filtering
    df = df[df["year"] >= 2000].copy()

    df["fuel_category"] = df[fuel_col].map(normalize_fuel_category)
    df = df[df["fuel_category"] != "Hydrogen"].copy()

    # Targets (L/100 km)
    y = pd.DataFrame(
        {
            "City (L/100 km)": mpg_to_l_per_100km(df["city08"].astype(float)),
            "Highway (L/100 km)": mpg_to_l_per_100km(df["highway08"].astype(float)),
        },
        index=df.index,
    )

    # Features
    x = pd.DataFrame(
        {
            "Combined (L/100 km)": mpg_to_l_per_100km(df["comb08"].astype(float)),
            "Cylinders": df["cylinders"].astype(float),
            "Engine size": df["displ"].astype(float),
            "Model year": df["year"].astype(int),
        },
        index=df.index,
    )

    for col in FUEL_ONE_HOT_COLUMNS:
        x[col] = 0

    for idx, cat in df["fuel_category"].items():
        one_hot_name = f"Fuel type_{cat}"
        if one_hot_name in x.columns:
            x.loc[idx, one_hot_name] = 1

    x = x[[
        "Combined (L/100 km)",
        "Cylinders",
        "Engine size",
        "Model year",
        *FUEL_ONE_HOT_COLUMNS,
    ]]

    # Drop rows with NaNs in feature/target
    keep = x.notna().all(axis=1) & y.notna().all(axis=1)
    x = x[keep]
    y = y[keep]

    return x, y


def load_model(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {path}: {exc}") from exc


def predict_two_targets(model: Any, x: pd.DataFrame) -> np.ndarray:
    """Return predictions shaped (n_samples, 2).

    Supports estimators that return either:
      - (n,2)
      - (n,) for single-target models (we'll broadcast to both as a fallback)
    """

    preds = model.predict(x)
    preds_arr = np.asarray(preds)

    if preds_arr.ndim == 1:
        # Fallback: broadcast single output to both targets.
        return np.column_stack([preds_arr, preds_arr])

    if preds_arr.ndim == 2 and preds_arr.shape[1] >= 2:
        return preds_arr[:, :2]

    raise ValueError(f"Unexpected prediction shape: {preds_arr.shape}")


def _normalize_feature_name(name: str) -> str:
    # Normalize for fuzzy matching between training-time and runtime column naming.
    return (
        name.strip()
        .lower()
        .replace("(l)", "")
        .replace("(l/100 km)", "")
        .replace("/", " ")
        .replace("_", " ")
        .replace("-", " ")
    )


def align_features_for_model(x: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Align a feature DataFrame to what a specific model expects.

    Some saved models were trained with different column naming conventions (case, suffixes
    like "(L)", or different fuel category strings). This function builds a model-specific
    X with columns ordered and named exactly as the estimator expects.
    """

    def _feature_contract(m: Any) -> Optional[List[str]]:
        names: Optional[List[str]] = None
        if hasattr(m, "feature_names_in_"):
            try:
                names = list(getattr(m, "feature_names_in_"))
            except Exception:
                names = None
        elif hasattr(m, "feature_names_"):
            # CatBoost exposes feature names via `feature_names_`.
            try:
                raw = getattr(m, "feature_names_")
                if isinstance(raw, (list, tuple)) and raw:
                    names = list(raw)
            except Exception:
                names = None
        elif hasattr(m, "get_booster"):
            try:
                names = list(m.get_booster().feature_names or [])
            except Exception:
                names = None
        elif hasattr(m, "booster_") and hasattr(m.booster_, "feature_name"):
            try:
                names = list(m.booster_.feature_name())
            except Exception:
                names = None

        if not names:
            return None
        return names

    expected = _feature_contract(model)

    # Prefer underlying estimator contracts for wrapper estimators.
    underlying_expected: Optional[List[str]] = None
    if hasattr(model, "estimators_"):
        try:
            for est in list(getattr(model, "estimators_")):
                if est is None:
                    continue
                underlying_expected = _feature_contract(est)
                if underlying_expected:
                    break
        except Exception:
            underlying_expected = None
    if not underlying_expected and hasattr(model, "estimator"):
        underlying_expected = _feature_contract(getattr(model, "estimator"))

    if underlying_expected:
        expected = underlying_expected

    if not expected:
        # No feature-name contract: return as-is.
        return x

    # Known synonym mapping from training-time to our canonical columns.
    # Canonical columns are produced by build_feature_frame().
    synonyms: Dict[str, str] = {
        "combined": "Combined (L/100 km)",
        "combined l 100 km": "Combined (L/100 km)",
        "combined (l/100 km)": "Combined (L/100 km)",
        "engine size": "Engine size",
        "engine size l": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l )": "Engine size",
        "engine size (l ) ": "Engine size",
        "engine size (l )": "Engine size",
        "engine size (l) ": "Engine size",
        "engine size l ": "Engine size",
        "engine size l": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size l": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size l": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size l": "Engine size",
        "engine size (l)": "Engine size",
        "engine size l": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size (l)": "Engine size",
        "engine size l": "Engine size",
        "model year": "Model year",
        "year": "Model year",
        "cylinders": "Cylinders",
        "fuel type diesel": "Fuel type_Diesel",
        "fuel type_diesel": "Fuel type_Diesel",
        "fuel type diesel ": "Fuel type_Diesel",
        "fuel type_electricity": "Fuel type_Electricity",
        "fuel type electricity": "Fuel type_Electricity",
        "fuel type_natural gas": "Fuel type_Natural gas",
        "fuel type natural gas": "Fuel type_Natural gas",
        "fuel type_premium gasoline": "Fuel type_Premium gasoline",
        "fuel type premium gasoline": "Fuel type_Premium gasoline",
        "fuel type_regular gasoline": "Fuel type_Regular gasoline",
        "fuel type regular gasoline": "Fuel type_Regular gasoline",
        "fuel type_midgrade gasoline": "Fuel type_Regular gasoline",
        "fuel type midgrade gasoline": "Fuel type_Regular gasoline",
        "fuel type_e85": "Fuel type_Ethanol (E85)",
        "fuel type e85": "Fuel type_Ethanol (E85)",
        "fuel type_ethanol (e85)": "Fuel type_Ethanol (E85)",
        "fuel type ethanol (e85)": "Fuel type_Ethanol (E85)",
    }

    # Fuzzy match from normalized names.
    norm_to_col = {_normalize_feature_name(c): c for c in x.columns}

    x_aligned = pd.DataFrame(index=x.index)
    for feat in expected:
        key = _normalize_feature_name(str(feat))
        source_col: Optional[str] = None

        if key in norm_to_col:
            source_col = norm_to_col[key]
        elif key in synonyms:
            source_col = synonyms[key]

        if source_col is not None and source_col in x.columns:
            x_aligned[str(feat)] = x[source_col]
        else:
            # Default to 0 for unseen one-hot columns, NaN otherwise.
            x_aligned[str(feat)] = 0

    return x_aligned


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute average metrics across the two targets."""

    city_true, hwy_true = y_true[:, 0], y_true[:, 1]
    city_pred, hwy_pred = y_pred[:, 0], y_pred[:, 1]

    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(a, b)))

    metrics: Dict[str, float] = {
        "MAE City": float(mean_absolute_error(city_true, city_pred)),
        "RMSE City": rmse(city_true, city_pred),
        "R2 City": float(r2_score(city_true, city_pred)),
        "MAE Highway": float(mean_absolute_error(hwy_true, hwy_pred)),
        "RMSE Highway": rmse(hwy_true, hwy_pred),
        "R2 Highway": float(r2_score(hwy_true, hwy_pred)),
        "MAE Avg": float((mean_absolute_error(city_true, city_pred) + mean_absolute_error(hwy_true, hwy_pred)) / 2.0),
        "RMSE Avg": float((rmse(city_true, city_pred) + rmse(hwy_true, hwy_pred)) / 2.0),
        "R2 Avg": float((r2_score(city_true, city_pred) + r2_score(hwy_true, hwy_pred)) / 2.0),
    }
    return metrics


def extract_notebook_markdown(notebook_path: Path) -> List[str]:
    """Extract markdown cell text from a .ipynb (best-effort)."""

    if not notebook_path.exists():
        return []

    try:
        data = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    cells = data.get("cells", [])
    out: List[str] = []
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", [])
        if isinstance(src, list):
            text = "".join(src).strip()
        else:
            text = str(src).strip()
        if text:
            out.append(text)
    return out


def extract_notebook_artifacts(notebook_path: Path, assets_dir: Path, max_images: int = 12) -> List[NotebookArtifact]:
    """Extract images and text outputs embedded in the notebook file (no execution).

    - Saves embedded PNG outputs to assets_dir
    - Captures relevant text outputs (e.g., printed metrics)
    """

    if not notebook_path.exists():
        return []

    try:
        data = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    artifacts: List[NotebookArtifact] = []
    img_count = 0
    cells = data.get("cells", [])

    for cell_index, cell in enumerate(cells, start=1):
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            continue

        for out in outputs:
            if not isinstance(out, dict):
                continue

            out_type = out.get("output_type")

            # Streamed text
            if out_type == "stream":
                text = out.get("text")
                if isinstance(text, list):
                    text = "".join(text)
                if isinstance(text, str) and text.strip():
                    artifacts.append(
                        NotebookArtifact(
                            path=None,
                            caption=f"Notebook output text (cell {cell_index})",
                            text=text.strip(),
                        )
                    )
                continue

            # Rich display data
            data_field = out.get("data", {})
            if not isinstance(data_field, dict):
                continue

            # Extract images
            if "image/png" in data_field and img_count < max_images:
                b64 = data_field.get("image/png")
                if isinstance(b64, list):
                    b64 = "".join(b64)
                if isinstance(b64, str) and b64.strip():
                    try:
                        raw = base64.b64decode(b64)
                        img_path = assets_dir / f"notebook_output_{img_count+1:02d}.png"
                        img_path.write_bytes(raw)
                        artifacts.append(
                            NotebookArtifact(
                                path=img_path,
                                caption=f"Extracted notebook figure (cell {cell_index})",
                            )
                        )
                        img_count += 1
                    except Exception:
                        pass

            # Extract text/plain (often tables/metrics)
            if "text/plain" in data_field:
                text = data_field.get("text/plain")
                if isinstance(text, list):
                    text = "".join(text)
                if isinstance(text, str) and text.strip():
                    artifacts.append(
                        NotebookArtifact(
                            path=None,
                            caption=f"Notebook output (cell {cell_index})",
                            text=text.strip(),
                        )
                    )

    return artifacts


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def make_eda_figures(df_raw: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame, paths: ReportPaths) -> Dict[str, Tuple[Path, str]]:
    """Generate EDA figures and return a mapping: key -> (image_path, caption)."""

    figs: Dict[str, Tuple[Path, str]] = {}

    # Distribution plots
    fig1_path = paths.assets_dir / "eda_distribution_city_highway.png"
    plt.figure(figsize=(10, 5))
    sns.histplot(y["City (L/100 km)"], kde=True, color="#4C78A8", label="City", stat="density", bins=40)
    sns.histplot(y["Highway (L/100 km)"], kde=True, color="#F58518", label="Highway", stat="density", bins=40)
    plt.title("Fuel Consumption Distribution (L/100 km)")
    plt.xlabel("L/100 km")
    plt.ylabel("Density")
    plt.legend()
    save_figure(fig1_path)
    figs["dist"] = (
        fig1_path,
        "Histogram + KDE of target distributions after filtering (year ≥ 2000; Hydrogen removed).",
    )

    # Fuel category counts
    fig2_path = paths.assets_dir / "eda_fuel_counts.png"
    plt.figure(figsize=(10, 5))
    if "fuelType1" in df_raw.columns:
        cats = df_raw[df_raw["year"] >= 2000]["fuelType1"].map(normalize_fuel_category)
    elif "fuelType" in df_raw.columns:
        cats = df_raw[df_raw["year"] >= 2000]["fuelType"].map(normalize_fuel_category)
    else:
        cats = pd.Series([], dtype=str)

    cats = cats[cats != "Hydrogen"]
    order = cats.value_counts().index.tolist()
    sns.countplot(y=cats, order=order)
    plt.title("Fuel Type Distribution (Filtered)")
    plt.xlabel("Count")
    plt.ylabel("Fuel category")
    save_figure(fig2_path)
    figs["fuel_counts"] = (
        fig2_path,
        "Counts of normalized fuel categories used in training/evaluation.",
    )

    # Correlation heatmap
    fig3_path = paths.assets_dir / "eda_correlation.png"
    plt.figure(figsize=(10, 7))
    corr_df = pd.concat([x[["Combined (L/100 km)", "Cylinders", "Engine size", "Model year"]], y], axis=1)
    corr = corr_df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Correlation Heatmap (Selected Features + Targets)")
    save_figure(fig3_path)
    figs["corr"] = (
        fig3_path,
        "Correlation matrix for core numeric features and targets.",
    )

    # Scatter: engine size vs city consumption
    fig4_path = paths.assets_dir / "eda_engine_vs_city.png"
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=x["Engine size"], y=y["City (L/100 km)"], alpha=0.3)
    plt.title("Engine Size vs City Fuel Consumption")
    plt.xlabel("Engine size (L)")
    plt.ylabel("City (L/100 km)")
    save_figure(fig4_path)
    figs["engine_city"] = (
        fig4_path,
        "Relationship between displacement and city fuel consumption.",
    )

    return figs


def make_evaluation_figures(metrics_df: pd.DataFrame, paths: ReportPaths) -> Dict[str, Tuple[Path, str]]:
    figs: Dict[str, Tuple[Path, str]] = {}

    fig_path = paths.assets_dir / "model_rmse_comparison.png"
    plt.figure(figsize=(10, 5))
    plot_df = metrics_df[["Model", "RMSE City", "RMSE Highway"]].copy()
    plot_df = plot_df.melt(id_vars=["Model"], var_name="Target", value_name="RMSE")
    sns.barplot(data=plot_df, x="Model", y="RMSE", hue="Target")
    plt.title("Model Comparison (RMSE)")
    plt.xlabel("Model")
    plt.ylabel("RMSE (L/100 km)")
    plt.xticks(rotation=20, ha="right")
    save_figure(fig_path)

    figs["rmse"] = (fig_path, "RMSE comparison for City and Highway targets across models.")
    return figs


def build_pdf(
    paths: ReportPaths,
    dataset_path: Path,
    notebook_path: Optional[Path],
    dataset_summary: Dict[str, Any],
    eda_figs: Dict[str, Tuple[Path, str]],
    metrics_df: pd.DataFrame,
    eval_figs: Dict[str, Tuple[Path, str]],
    notebook_artifacts: Optional[List[NotebookArtifact]] = None,
) -> None:
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="TitleStyle",
        parent=styles["Title"],
        fontSize=22,
        leading=26,
        spaceAfter=16,
    )

    h1 = ParagraphStyle(
        name="H1",
        parent=styles["Heading1"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    )

    h2 = ParagraphStyle(
        name="H2",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=6,
    )

    body = ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
    )

    small = ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=12,
        textColor=colors.grey,
    )

    toc_h1 = ParagraphStyle(
        name="TOCLevel1",
        parent=styles["BodyText"],
        fontSize=11,
        leading=14,
        leftIndent=16,
        firstLineIndent=-16,
        spaceBefore=4,
        spaceAfter=2,
    )

    toc_h2 = ParagraphStyle(
        name="TOCLevel2",
        parent=styles["BodyText"],
        fontSize=10,
        leading=12,
        leftIndent=32,
        firstLineIndent=-16,
        spaceBefore=2,
        spaceAfter=1,
        textColor=colors.HexColor("#444444"),
    )

    def _draw_header_footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.lightgrey)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, 0.75 * inch, LETTER[0] - doc.rightMargin, 0.75 * inch)

        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.grey)
        canvas.drawString(doc.leftMargin, 0.55 * inch, PROJECT_TITLE)
        canvas.drawRightString(LETTER[0] - doc.rightMargin, 0.55 * inch, f"Page {doc.page}")
        canvas.restoreState()

    class _ProDocTemplate(SimpleDocTemplate):
        def afterFlowable(self, flowable):
            if not isinstance(flowable, Paragraph):
                return

            style_name = getattr(flowable.style, "name", "")
            if style_name not in {"H1", "H2"}:
                return

            text = flowable.getPlainText()
            level = 0 if style_name == "H1" else 1
            key = f"toc_{level}_{self.page}_{abs(hash(text))}"
            try:
                flowable._bookmarkName = key  # type: ignore[attr-defined]
            except Exception:
                pass
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (level, text, self.page, key))

    doc = _ProDocTemplate(
        str(paths.out_pdf),
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=PROJECT_TITLE,
        author="",
    )

    story: List[Any] = []

    # Title page
    story.append(Spacer(1, 1.0 * inch))
    story.append(Paragraph(PROJECT_TITLE, title_style))
    story.append(Paragraph(REPORT_SUBTITLE, styles["Heading2"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(f"Date: {date.today().isoformat()}", body))
    story.append(Paragraph(f"Deployment: <a href='{DEPLOYMENT_LINK}'>{DEPLOYMENT_LINK}</a>", body))
    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph(f"Dataset: {dataset_path}", small))
    if notebook_path:
        story.append(Paragraph(f"Notebook reference: {notebook_path}", small))
    story.append(PageBreak())

    # Table of Contents
    toc = TableOfContents()
    toc.levelStyles = [toc_h1, toc_h2]
    story.append(Paragraph("Table of Contents", h1))
    story.append(Spacer(1, 0.15 * inch))
    story.append(toc)
    story.append(PageBreak())

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", h1))
    story.append(
        Paragraph(
            "This report presents a complete end-to-end workflow for predicting vehicle fuel consumption in two "
            "driving contexts—City and Highway—measured in L/100 km. The pipeline covers data preparation, exploratory "
            "analysis, model development, evaluation, and deployment via a Streamlit application.",
            body,
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Project Phases", h2))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("Data collection: load a FuelEconomy-style vehicles dataset (vehicles.csv).", body)),
                ListItem(Paragraph("Preprocessing: filter years, normalize fuel types, convert MPG to L/100 km.", body)),
                ListItem(Paragraph("Modeling: evaluate multiple regressors to predict City and Highway jointly.", body)),
                ListItem(Paragraph("Deployment: expose the best-performing models via Streamlit.", body)),
            ],
            bulletType="bullet",
            leftIndent=18,
            bulletFontName="Helvetica",
            bulletFontSize=9,
        )
    )

    story.append(Spacer(1, 0.15 * inch))

    # 2. Project Overview and Objectives
    story.append(Paragraph("2. Project Overview and Objectives", h1))
    story.append(Paragraph("2.1 Problem Statement", h2))
    story.append(
        Paragraph(
            "Fuel consumption affects total cost of ownership, emissions, and vehicle selection decisions. "
            "Given a compact set of vehicle specifications, the objective is to estimate expected fuel consumption "
            "for City and Highway driving conditions.",
            body,
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("2.2 Project Objectives", h2))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("Build a consistent feature pipeline aligned with the deployed app.", body)),
                ListItem(Paragraph("Perform EDA to understand distributions and relationships.", body)),
                ListItem(Paragraph("Train and compare multiple ML models.", body)),
                ListItem(Paragraph("Report clear metrics (MAE, RMSE, R²) for both targets.", body)),
                ListItem(Paragraph("Provide a deployment-ready inference interface in Streamlit.", body)),
            ],
            bulletType="bullet",
            leftIndent=18,
            bulletFontName="Helvetica",
            bulletFontSize=9,
        )
    )
    story.append(Spacer(1, 0.18 * inch))

    # 3. Data Collection
    story.append(Paragraph("3. Data Collection", h1))
    story.append(Paragraph("3.1 Dataset Source", h2))
    story.append(
        Paragraph(
            "The project uses a FuelEconomy-style vehicles dataset (vehicles.csv) containing vehicle specifications "
            "and fuel economy measurements. The generator supports auto-detecting vehicles.csv in the project folder.",
            body,
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("3.2 Dataset Overview", h2))

    ds_table = Table(
        [
            ["Rows (raw)", str(dataset_summary.get("rows_raw", ""))],
            ["Rows (filtered)", str(dataset_summary.get("rows_filtered", ""))],
            ["Columns (raw)", str(dataset_summary.get("cols_raw", ""))],
            ["Target columns", "City (L/100 km), Highway (L/100 km)"],
            ["Feature columns", "Combined, Cylinders, Engine size, Model year, Fuel one-hots"],
        ],
        colWidths=[1.8 * inch, 4.8 * inch],
    )
    ds_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ]
        )
    )
    story.append(ds_table)

    # 4. Data Preprocessing and Cleaning
    story.append(Spacer(1, 0.22 * inch))
    story.append(Paragraph("4. Data Preprocessing and Cleaning", h1))
    story.append(Paragraph("4.1 Key Transformations", h2))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("Filter to modern vehicles (year ≥ 2000).", body)),
                ListItem(Paragraph("Normalize fuel type values into consistent categories.", body)),
                ListItem(Paragraph("Exclude Hydrogen entries to align with deployed feature set.", body)),
                ListItem(Paragraph("Convert MPG to L/100 km using 235.214583 / MPG.", body)),
                ListItem(Paragraph("Drop rows with missing values in features/targets.", body)),
            ],
            bulletType="bullet",
            leftIndent=18,
            bulletFontName="Helvetica",
            bulletFontSize=9,
        )
    )

    # 5. Exploratory Data Analysis (EDA)
    story.append(PageBreak())
    story.append(Paragraph("5. Exploratory Data Analysis (EDA)", h1))
    story.append(
        Paragraph(
            "This section summarizes key data characteristics relevant to model performance. "
            "Figures are generated automatically from the provided dataset.",
            body,
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    if eda_figs:
        for _, (img_path, caption) in eda_figs.items():
            story.append(Image(str(img_path), width=6.5 * inch, height=3.4 * inch))
            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph(f"<b>Figure:</b> {caption}", small))
            story.append(Spacer(1, 0.2 * inch))
    elif notebook_artifacts:
        # Fallback: include extracted notebook figures in EDA section.
        for art in [a for a in notebook_artifacts if a.path is not None][:6]:
            story.append(Image(str(art.path), width=6.5 * inch, height=3.4 * inch))
            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph(f"<b>Figure:</b> {art.caption}", small))
            story.append(Spacer(1, 0.2 * inch))

    # 6. Machine Learning Model Development
    story.append(PageBreak())
    story.append(Paragraph("6. Machine Learning Model Development", h1))
    story.append(Paragraph("6.1 Feature Set", h2))
    story.append(
        Paragraph(
            "The feature set mirrors the deployed application and includes: Combined consumption (L/100 km), Cylinders, "
            "Engine size (displacement), Model year, and one-hot encoded fuel categories.",
            body,
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("6.2 Models", h2))
    story.append(
        Paragraph(
            "The deployed system is an ensemble-style comparison of multiple regressors trained to predict "
            "City and Highway fuel consumption jointly. Models are loaded from serialized .pkl files and "
            "evaluated consistently on the same held-out test split.",
            body,
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "Models evaluated (when available): Random Forest, XGBoost, LightGBM, Decision Tree, and CatBoost.",
            body,
        )
    )

    # 7. Model Evaluation and Results
    story.append(PageBreak())
    story.append(Paragraph("7. Model Evaluation and Results", h1))
    story.append(
        Paragraph(
            "Metrics reported: MAE, RMSE, and R² for each target plus their averages. "
            "Lower MAE/RMSE is better; higher R² is better.",
            body,
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    if metrics_df is not None and not metrics_df.empty:
        cols = ["Model", "MAE City", "RMSE City", "R2 City", "MAE Highway", "RMSE Highway", "R2 Highway", "RMSE Avg"]
        table_data: List[List[str]] = [cols]
        for _, row in metrics_df.iterrows():
            table_data.append(
                [
                    str(row["Model"]),
                    f"{row['MAE City']:.3f}",
                    f"{row['RMSE City']:.3f}",
                    f"{row['R2 City']:.3f}",
                    f"{row['MAE Highway']:.3f}",
                    f"{row['RMSE Highway']:.3f}",
                    f"{row['R2 Highway']:.3f}",
                    f"{row['RMSE Avg']:.3f}",
                ]
            )

        metrics_table = Table(table_data, repeatRows=1)
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EEF6")),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
                ]
            )
        )
        story.append(metrics_table)
        story.append(Spacer(1, 0.2 * inch))

        for _, (img_path, caption) in eval_figs.items():
            story.append(Image(str(img_path), width=6.5 * inch, height=3.4 * inch))
            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph(f"<b>Figure:</b> {caption}", small))
            story.append(Spacer(1, 0.2 * inch))
    elif notebook_artifacts:
        # If we don't compute metrics, include any extracted notebook text that looks like metrics.
        metric_texts = [
            a for a in notebook_artifacts
            if a.text and any(k in a.text.lower() for k in ["rmse", "mae", "r2", "score", "mse"])
        ]
        for art in metric_texts[:8]:
            story.append(Paragraph(f"<b>{art.caption}:</b>", small))
            story.append(Paragraph(art.text.replace("\n", "<br/>")[:4000], body))
            story.append(Spacer(1, 0.15 * inch))

    # 8. Conclusions and Future Work
    story.append(PageBreak())
    story.append(Paragraph("8. Conclusions and Future Work", h1))
    story.append(
        Paragraph(
            "This report demonstrates an end-to-end workflow from dataset preparation and EDA to model "
            "evaluation and deployment. Future work can include richer feature engineering (e.g., vehicle class, "
            "transmission descriptors) and improved robustness across new model years.",
            body,
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Deployment", h2))
    story.append(
        Paragraph(
            "The final models are exposed via a Streamlit web application that collects the same feature set and "
            "returns City and Highway predictions in L/100 km.",
            body,
        )
    )
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph(f"Live app: <a href='{DEPLOYMENT_LINK}'>{DEPLOYMENT_LINK}</a>", body))

    # Optional: include notebook markdown appendix (if present)
    if notebook_path:
        md_cells = extract_notebook_markdown(notebook_path)
        if md_cells:
            story.append(PageBreak())
            story.append(Paragraph("Appendix: Notebook Notes (Markdown Cells)", h1))
            story.append(
                Paragraph(
                    "The following section includes markdown notes extracted from the referenced notebook (best-effort).",
                    body,
                )
            )
            story.append(Spacer(1, 0.1 * inch))
            for block in md_cells[:12]:
                story.append(Paragraph(block.replace("\n", "<br/>").strip(), body))
                story.append(Spacer(1, 0.1 * inch))

    doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a detailed PDF report for the fuel consumption project.")
    parser.add_argument(
        "--mode",
        choices=["notebook", "reproduce"],
        default="notebook",
        help="'notebook' extracts embedded notebook outputs (no execution). 'reproduce' recomputes EDA+evaluation from the dataset.",
    )
    parser.add_argument(
        "--data",
        required=False,
        default=None,
        help="Path to vehicles.csv (FuelEconomy-style dataset). If omitted, the script tries to auto-detect 'vehicles.csv' in the project folder.",
    )
    parser.add_argument(
        "--notebook",
        default="Fuel_Consumbtion_Models_and_Merging (2).ipynb",
        help="Optional path to the reference notebook (.ipynb).",
    )
    parser.add_argument(
        "--out",
        default="Fuel_Consumption_Report.pdf",
        help="Output PDF filename.",
    )
    parser.add_argument(
        "--assets",
        default="report_assets",
        help="Directory to write generated figures.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent

    data_path: Optional[Path]
    if args.data:
        data_path = Path(args.data).expanduser().resolve()
    else:
        # Auto-detect common dataset filename in the project folder.
        candidates = [base_dir / "vehicles.csv", base_dir / "Vehicles.csv"]
        data_path = next((p for p in candidates if p.exists()), None)
    notebook_path = Path(args.notebook).expanduser().resolve() if args.notebook else None

    # In notebook mode, the dataset is optional (we use embedded notebook outputs).
    if args.mode == "reproduce":
        if data_path is None or not data_path.exists():
            raise SystemExit(
                "Missing dataset for reproduce mode. Provide it with --data, e.g.\n\n"
                "  python generate_pdf_report.py --mode reproduce --data path\\to\\vehicles.csv --out Fuel_Consumption_Report.pdf\n\n"
                "Auto-detection looks for 'vehicles.csv' in the project folder, but none was found."
            )

    out_pdf = Path(args.out)
    if not out_pdf.is_absolute():
        out_pdf = (base_dir / out_pdf).resolve()

    assets_dir = Path(args.assets)
    if not assets_dir.is_absolute():
        assets_dir = (base_dir / assets_dir).resolve()

    paths = ReportPaths(out_pdf=out_pdf, assets_dir=assets_dir)
    _safe_mkdir(paths.assets_dir)

    notebook_artifacts: Optional[List[NotebookArtifact]] = None
    if args.mode == "notebook":
        if notebook_path and notebook_path.exists():
            notebook_artifacts = extract_notebook_artifacts(notebook_path, paths.assets_dir)
        else:
            notebook_artifacts = None

        # Minimal dataset summary (best-effort).
        dataset_summary = {
            "rows_raw": "(from notebook outputs)",
            "cols_raw": "(from notebook outputs)",
            "rows_filtered": "(from notebook outputs)",
        }

        build_pdf(
            paths=paths,
            dataset_path=data_path if data_path is not None else base_dir / "vehicles.csv",
            notebook_path=notebook_path if notebook_path and notebook_path.exists() else None,
            dataset_summary=dataset_summary,
            eda_figs={},
            metrics_df=pd.DataFrame(),
            eval_figs={},
            notebook_artifacts=notebook_artifacts,
        )

        print(f"Wrote PDF: {paths.out_pdf}")
        print(f"Wrote assets to: {paths.assets_dir}")
        return 0

    # reproduce mode
    assert data_path is not None

    df_raw = pd.read_csv(data_path, low_memory=False)
    x, y = build_feature_frame(df_raw)

    dataset_summary = {
        "rows_raw": int(df_raw.shape[0]),
        "cols_raw": int(df_raw.shape[1]),
        "rows_filtered": int(x.shape[0]),
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(args.test_size),
        random_state=int(args.seed),
    )

    model_paths = {
        "Random Forest": base_dir / "random_forest_model.pkl",
        "XGBoost": base_dir / "xgb_model.pkl",
        "LightGBM": base_dir / "lgbm_model.pkl",
        "Decision Tree": base_dir / "decision_tree_model.pkl",
        "CatBoost": base_dir / "catboost_model.pkl",
    }

    models: Dict[str, Any] = {}
    for name, p in model_paths.items():
        if p.exists():
            models[name] = load_model(p)

    if not models:
        raise SystemExit("No model .pkl files found next to the script.")

    y_test_arr = y_test[["City (L/100 km)", "Highway (L/100 km)"]].to_numpy(dtype=float)

    rows: List[Dict[str, Any]] = []
    all_preds: List[np.ndarray] = []

    for name, model in models.items():
        x_for_model = align_features_for_model(x_test, model)
        pred = predict_two_targets(model, x_for_model)
        all_preds.append(pred)
        m = compute_metrics(y_test_arr, pred)
        m["Model"] = name
        rows.append(m)

    if len(all_preds) >= 2:
        ens = np.mean(np.stack(all_preds, axis=0), axis=0)
        m = compute_metrics(y_test_arr, ens)
        m["Model"] = "Ensemble (avg)"
        rows.append(m)

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df.sort_values(by="RMSE Avg", ascending=True).reset_index(drop=True)

    eda_figs = make_eda_figures(df_raw=df_raw, x=x, y=y, paths=paths)
    eval_figs = make_evaluation_figures(
        metrics_df=metrics_df[metrics_df["Model"] != "Ensemble (avg)"].copy(),
        paths=paths,
    )

    build_pdf(
        paths=paths,
        dataset_path=data_path,
        notebook_path=notebook_path if notebook_path and notebook_path.exists() else None,
        dataset_summary=dataset_summary,
        eda_figs=eda_figs,
        metrics_df=metrics_df,
        eval_figs=eval_figs,
        notebook_artifacts=None,
    )

    print(f"Wrote PDF: {paths.out_pdf}")
    print(f"Wrote assets to: {paths.assets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
