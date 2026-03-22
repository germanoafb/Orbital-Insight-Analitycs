"""
Visualization utilities for satellite imagery and analysis results.
Uses matplotlib and plotly for static and interactive charts.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64


# ── NDVI colour palette ────────────────────────────────────────────────────
NDVI_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ndvi",
    [
        (0.0, "#d73027"),   # Water / very low NDVI  → red
        (0.25, "#fee08b"),  # Bare soil              → yellow
        (0.5, "#ffffbf"),   # Sparse vegetation      → pale yellow
        (0.75, "#91cf60"),  # Moderate vegetation    → light green
        (1.0, "#1a9850"),   # Dense vegetation       → dark green
    ],
)


def ndvi_to_base64(ndvi: np.ndarray, title: str = "NDVI Map") -> str:
    """
    Render an NDVI array as a PNG image and return it as a base64 string.

    Args:
        ndvi: 2-D NDVI array with values in [-1, 1].
        title: Plot title.

    Returns:
        Base64-encoded PNG string (suitable for embedding in HTML or Streamlit).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(ndvi, cmap=NDVI_CMAP, vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="NDVI")
    ax.set_title(title)
    ax.axis("off")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def deforestation_map_to_base64(
    before: np.ndarray,
    after: np.ndarray,
    title: str = "Deforestation Detection",
) -> str:
    """
    Create a side-by-side comparison of NDVI images (before / after)
    with a difference map highlighting deforested areas.

    Args:
        before: NDVI array for the earlier period.
        after:  NDVI array for the later period.
        title:  Overall figure title.

    Returns:
        Base64-encoded PNG string.
    """
    diff = before - after  # Positive values = vegetation loss

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, label in zip(
        axes,
        [before, after, diff],
        ["Before", "After", "Vegetation Loss (Δ NDVI)"],
    ):
        if label.startswith("Vegetation"):
            cmap, vmin, vmax = "RdYlGn_r", -1, 1
        else:
            cmap, vmin, vmax = NDVI_CMAP, -1, 1
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def urban_growth_map_to_base64(
    before: np.ndarray,
    after: np.ndarray,
    title: str = "Urban Growth Map",
) -> str:
    """
    Highlight new urban areas that appeared between two classification maps.

    Args:
        before: Land-use classification array (0 = non-urban, 1 = urban).
        after:  Land-use classification array (0 = non-urban, 1 = urban).
        title:  Figure title.

    Returns:
        Base64-encoded PNG string.
    """
    new_urban = (after == 1) & (before == 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, label, cmap in zip(
        axes,
        [before, after, new_urban.astype(int)],
        ["Before (Urban)", "After (Urban)", "New Urban Areas"],
        ["Blues", "Blues", "Reds"],
    ):
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_title(label)
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def productivity_trend_chart(
    years: list,
    actual: list,
    predicted: list,
) -> go.Figure:
    """
    Build an interactive Plotly line chart comparing actual vs predicted
    agricultural productivity over time.

    Args:
        years:     List of year labels.
        actual:    Actual productivity values (tons/ha).
        predicted: Model-predicted productivity values (tons/ha).

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=years,
            y=actual,
            mode="lines+markers",
            name="Actual",
            line={"color": "#2196F3", "width": 2},
            marker={"size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=predicted,
            mode="lines+markers",
            name="Predicted",
            line={"color": "#FF9800", "dash": "dash", "width": 2},
            marker={"size": 8},
        )
    )
    fig.update_layout(
        title="Agricultural Productivity Trend",
        xaxis_title="Year",
        yaxis_title="Productivity (tons/ha)",
        legend={"orientation": "h"},
        template="plotly_white",
    )
    return fig


def ndvi_histogram(ndvi: np.ndarray) -> go.Figure:
    """
    Build an interactive Plotly histogram of NDVI values.

    Args:
        ndvi: NDVI array.

    Returns:
        Plotly Figure object.
    """
    flat = ndvi.flatten()
    fig = px.histogram(
        x=flat,
        nbins=50,
        labels={"x": "NDVI Value", "y": "Count"},
        title="NDVI Value Distribution",
        color_discrete_sequence=["#4CAF50"],
    )
    fig.update_layout(template="plotly_white")
    return fig


def feature_importance_chart(features: list, importances: list) -> go.Figure:
    """
    Build a horizontal bar chart for model feature importances.

    Args:
        features:    Feature names.
        importances: Importance scores.

    Returns:
        Plotly Figure object.
    """
    sorted_pairs = sorted(zip(importances, features))
    sorted_imp, sorted_feat = zip(*sorted_pairs)

    fig = go.Figure(
        go.Bar(
            x=list(sorted_imp),
            y=list(sorted_feat),
            orientation="h",
            marker_color="#9C27B0",
        )
    )
    fig.update_layout(
        title="Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
    )
    return fig
