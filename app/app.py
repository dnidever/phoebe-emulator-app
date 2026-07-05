from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make local src package importable when running from a fresh clone.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phoebe_emulator_app.emulator_utils import (  # noqa: E402
    DEFAULT_LABEL_RANGES,
    PHASE_GRID,
    ecc_per0_from_ecosw_esinw,
    in_training_range,
    labels_to_vector,
    lightcurve_metrics,
    load_emulator,
    predict_flux,
)

st.set_page_config(page_title="PHOEBE Emulator Explorer", layout="wide")

st.title("PHOEBE Eclipsing-Binary Emulator Explorer")
st.caption("Interactive viewer for a theborg emulator trained on PHOEBE light-curve-shape templates.")


@st.cache_resource(show_spinner=False)
def cached_load_emulator(path: str):
    return load_emulator(path)


def default_model_candidates() -> list[str]:
    """Find likely bundled or local model files."""
    candidates: list[str] = []

    preferred = [
        ROOT / "python" / "phoebe_emulator_app" / "data" / "phoebe_shape_annmodel.pkl",
        ROOT / "src" / "phoebe_emulator_app" / "data" / "phoebe_shape_annmodel.pkl",
        ROOT / "data" / "phoebe_shape_annmodel.pkl",
        ROOT / "phoebe_shape_annmodel.pkl",
    ]
    candidates.extend(str(p) for p in preferred if p.exists())

    search_dirs = [
        ROOT,
        ROOT / "data",
        ROOT / "models",
        ROOT / "shape_templates",
        ROOT / "python" / "phoebe_emulator_app" / "data",
        ROOT / "src" / "phoebe_emulator_app" / "data",
        Path.cwd(),
    ]
    for directory in search_dirs:
        if directory.exists():
            candidates.extend(str(p) for p in sorted(directory.glob("*.pkl")))

    # Preserve order while deduplicating.
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def slider_step(lo: float, hi: float, name: str) -> float:
    """Choose a useful slider step from the training range."""
    if name.lower() in ["incl", "inclination", "per0", "omega"]:
        return 0.05 if hi - lo <= 50 else 1.0
    width = hi - lo
    if width <= 0:
        return 0.001
    return max(width / 1000.0, 1e-5)


def range_for(bundle, name: str) -> tuple[float, float]:
    """Prefer emulator training ranges; fall back to known defaults."""
    if bundle.label_min is not None and bundle.label_max is not None and name in bundle.label_names:
        i = bundle.label_names.index(name)
        lo = float(bundle.label_min[i])
        hi = float(bundle.label_max[i])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            return lo, hi
    return DEFAULT_LABEL_RANGES.get(name, (0.0, 1.0))


def default_value(name: str, lo: float, hi: float) -> float:
    """Sensible starting value clipped to the available range."""
    defaults = {
        "r1_over_a": 0.15,
        "r2_over_a": 0.10,
        "incl": 85.0,
        "sbratio": 0.5,
        "q": 0.7,
        "ecosw": 0.05,
        "esinw": 0.05,
    }
    val = defaults.get(name, 0.5 * (lo + hi))
    return float(np.clip(val, lo, hi))

with st.sidebar:
    candidates = default_model_candidates()
    model_path = st.text_input(
        "Model .pkl",
        value=candidates[0] if candidates else "python/phoebe_emulator_app/data/phoebe_shape_annmodel.pkl",
        label_visibility="collapsed",
    )

    show_model_info = st.checkbox("Show model info", value=False)
    strict_range = True
    pad_fraction = 0.0
    show_residual = True

    fixed_ylimits = st.checkbox("Fixed y-axis", value=True)
    if fixed_ylimits:
        ymin, ymax = st.slider("Flux range", 0.0, 1.1, (0.4, 1.02), 0.01)

if not model_path:
    st.info("Enter the path to a trained emulator `.pkl` file in the sidebar.")
    st.stop()

try:
    bundle = cached_load_emulator(model_path)
except Exception as exc:
    st.error(f"Could not load emulator: {exc}")
    st.stop()

# One slider for every emulator label, in exactly the order expected by the model.
values: dict[str, float] = {}
with st.sidebar:
    #st.divider()
    st.subheader("Parameters")
    #st.caption("Sliders use the loaded emulator label names and training-label ranges when available.")

    st.markdown("""
    <style>
    section[data-testid="stSidebar"] .stSlider {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    section[data-testid="stSidebar"] .stSlider > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    section[data-testid="stSidebar"] label {
        margin-bottom: -0.35rem;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        margin-bottom: 0rem;
    }
    section[data-testid="stSidebar"] hr {
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    for name in bundle.label_names:
        lo, hi = range_for(bundle, name)
        step = slider_step(lo, hi, name)
        label = {
            "r1_over_a": "R1/a",
            "r2_over_a": "R2/a",
            "incl": "i",
            "sbratio": "SB",
            "q": "q",
            "ecosw": "e cosω",
            "esinw": "e sinω",
        }.get(name, name)
        values[name] = st.slider(
            label,
            min_value=float(lo),
            max_value=float(hi),
            value=default_value(name, lo, hi),
            step=float(step),
            key=f"slider_{name}",
        )

label_vector = labels_to_vector(values, bundle.label_names)

# Derived eccentricity/orientation display, if labels are present.
ecc = per0 = np.nan
if "ecosw" in values and "esinw" in values:
    ecc, per0 = ecc_per0_from_ecosw_esinw(values["ecosw"], values["esinw"])

inside, messages = in_training_range(label_vector, bundle, pad_fraction=pad_fraction)
if not inside:
    st.warning("Some labels are outside the training range:\n\n" + "\n".join(f"- {m}" for m in messages))
    if strict_range:
        st.stop()

try:
    flux = predict_flux(bundle, label_vector)
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

metrics = lightcurve_metrics(bundle.phase_grid, flux)

metric_cols = st.columns(5)
if np.isfinite(ecc):
    metric_cols[0].metric("ecc", f"{ecc:.4f}")
    metric_cols[1].metric("per0", f"{per0:.1f}°")
else:
    metric_cols[0].metric("labels", str(len(bundle.label_names)))
    metric_cols[1].metric("phase pixels", str(len(bundle.phase_grid)))
metric_cols[2].metric("depth", f"{metrics['depth']:.4f}")
metric_cols[3].metric("min flux", f"{metrics['min_flux']:.4f}")
metric_cols[4].metric("half-depth width", f"{metrics['eclipse_width_half_depth']:.3f}")

fig, ax = plt.subplots(figsize=(9, 3.0))
#fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(bundle.phase_grid, flux, lw=1.5, label="Emulator")
ax.axhline(1.0, ls="--", lw=0.8)
ax.set_xlabel("Phase")
ax.set_ylabel("Normalized flux")
ax.set_title("Emulated light curve")
ax.legend()
#if y_limits:
#    pad = max(0.02, 0.15 * metrics["depth"])
#    ax.set_ylim(max(0, metrics["min_flux"] - pad), 1.0 + pad)
if fixed_ylimits:
    ax.set_ylim(ymin, ymax)
st.pyplot(fig, clear_figure=True)

with st.expander("Parameter vector"):
    display = pd.DataFrame({"label": bundle.label_names, "value": label_vector})
    if bundle.label_min is not None and bundle.label_max is not None:
        display["training_min"] = bundle.label_min
        display["training_max"] = bundle.label_max
    st.dataframe(display, hide_index=True, use_container_width=True)

with st.expander("Compare two curves"):
    st.write("Make a second curve by perturbing one parameter from the current setting.")
    p_name = st.selectbox("Parameter to change", bundle.label_names, index=0)
    p_lo, p_hi = range_for(bundle, p_name)
    p2 = st.slider(
        f"Comparison {p_name}",
        min_value=float(p_lo),
        max_value=float(p_hi),
        value=float(values[p_name]),
        step=float(slider_step(p_lo, p_hi, p_name)),
        key="compare_slider",
    )

    values2 = dict(values)
    values2[p_name] = p2
    vector2 = labels_to_vector(values2, bundle.label_names)
    inside2, messages2 = in_training_range(vector2, bundle, pad_fraction=pad_fraction)

    if strict_range and not inside2:
        st.warning("Comparison model is outside range: " + "; ".join(messages2))
    else:
        try:
            flux2 = predict_flux(bundle, vector2)
            fig2, ax2 = plt.subplots(figsize=(9, 4.6))
            ax2.plot(bundle.phase_grid, flux, lw=2, label="Base")
            ax2.plot(bundle.phase_grid, flux2, lw=2, label=f"{p_name} = {p2:.4g}")
            ax2.axhline(1.0, ls="--", lw=0.8)
            ax2.set_xlabel("Phase")
            ax2.set_ylabel("Normalized flux")
            ax2.legend()
            st.pyplot(fig2, clear_figure=True)

            if show_residual:
                fig3, ax3 = plt.subplots(figsize=(9, 2.8))
                ax3.plot(bundle.phase_grid, flux2 - flux, lw=1.5)
                ax3.axhline(0.0, ls="--", lw=0.8)
                ax3.set_xlabel("Phase")
                ax3.set_ylabel("Comparison - base")
                st.pyplot(fig3, clear_figure=True)
        except Exception as exc:
            st.error(f"Comparison prediction failed: {exc}")
