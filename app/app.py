from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make local src package importable when running from a fresh clone.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "python"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phoebe_emulator_app.emulator_utils import (  # noqa: E402
    LABEL_NAMES,
    PHASE_GRID,
    ecc_per0_from_ecosw_esinw,
    ecosw_esinw_from_ecc_per0,
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
    candidates = []
    for directory in [ROOT, ROOT / "models", ROOT / "shape_templates", Path.cwd()]:
        if directory.exists():
            candidates.extend(str(p) for p in sorted(directory.glob("*.pkl")))
    return sorted(set(candidates))


with st.sidebar:
    st.header("Model")
    candidates = default_model_candidates()
    model_path = st.text_input(
        "Emulator .pkl path",
        value=candidates[0] if candidates else "",
        help="Path to a trained theborg emulator pickle.",
    )
    strict_range = st.checkbox("Require labels inside training range", value=True)
    pad_fraction = st.slider("Training-range padding", 0.0, 0.25, 0.0, 0.01)
    st.divider()
    st.header("Plot")
    y_limits = st.checkbox("Fix y-axis around eclipse", value=False)
    show_residual = st.checkbox("Show comparison residual", value=True)

if not model_path:
    st.info("Enter the path to a trained emulator `.pkl` file in the sidebar.")
    st.stop()

try:
    bundle = cached_load_emulator(model_path)
except Exception as exc:
    st.error(f"Could not load emulator: {exc}")
    st.stop()

if bundle.label_names != LABEL_NAMES:
    st.warning(f"Loaded label names {bundle.label_names}; app expects {LABEL_NAMES}. Proceeding with loaded order.")

# Slider ranges. Prefer actual training-label ranges when available.
def range_for(name: str, fallback: tuple[float, float]) -> tuple[float, float]:
    if bundle.label_min is None or bundle.label_max is None:
        return fallback
    i = bundle.label_names.index(name)
    return float(bundle.label_min[i]), float(bundle.label_max[i])

fallback_ranges = {
    "r1_over_a": (0.02, 0.45),
    "r2_over_a": (0.02, 0.45),
    "incl": (65.0, 90.0),
    "sbratio": (0.05, 2.0),
    "q": (0.1, 1.0),
    "ecosw": (-0.6, 0.6),
    "esinw": (-0.6, 0.6),
}

st.subheader("Parameters")
col1, col2, col3, col4 = st.columns(4)

values = {}
with col1:
    lo, hi = range_for("r1_over_a", fallback_ranges["r1_over_a"])
    values["r1_over_a"] = st.slider("R1/a", lo, hi, float(0.15 if lo <= 0.15 <= hi else 0.5 * (lo + hi)), step=(hi-lo)/500)
    lo, hi = range_for("r2_over_a", fallback_ranges["r2_over_a"])
    values["r2_over_a"] = st.slider("R2/a", lo, hi, float(0.10 if lo <= 0.10 <= hi else 0.5 * (lo + hi)), step=(hi-lo)/500)
with col2:
    lo, hi = range_for("incl", fallback_ranges["incl"])
    values["incl"] = st.slider("Inclination [deg]", lo, hi, float(85.0 if lo <= 85.0 <= hi else 0.5 * (lo + hi)), step=0.05)
    lo, hi = range_for("sbratio", fallback_ranges["sbratio"])
    values["sbratio"] = st.slider("Surface brightness ratio", lo, hi, float(0.5 if lo <= 0.5 <= hi else 0.5 * (lo + hi)), step=(hi-lo)/500)
with col3:
    lo, hi = range_for("q", fallback_ranges["q"])
    values["q"] = st.slider("Mass ratio q", lo, hi, float(0.7 if lo <= 0.7 <= hi else 0.5 * (lo + hi)), step=(hi-lo)/500)
    eccentricity_mode = st.radio("Eccentricity controls", ["e cosω / e sinω", "e / ω"], horizontal=False)
with col4:
    if eccentricity_mode == "e cosω / e sinω":
        lo, hi = range_for("ecosw", fallback_ranges["ecosw"])
        values["ecosw"] = st.slider("e cosω", lo, hi, float(0.05 if lo <= 0.05 <= hi else 0.0), step=(hi-lo)/500)
        lo, hi = range_for("esinw", fallback_ranges["esinw"])
        values["esinw"] = st.slider("e sinω", lo, hi, float(0.05 if lo <= 0.05 <= hi else 0.0), step=(hi-lo)/500)
    else:
        ecc = st.slider("ecc", 0.0, 0.6, 0.1, step=0.005)
        per0 = st.slider("ω / per0 [deg]", 0.0, 360.0, 45.0, step=1.0)
        values["ecosw"], values["esinw"] = ecosw_esinw_from_ecc_per0(ecc, per0)
        st.write(f"e cosω = `{values['ecosw']:.4f}`")
        st.write(f"e sinω = `{values['esinw']:.4f}`")

label_vector = labels_to_vector(values, bundle.label_names)
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

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("ecc", f"{ecc:.4f}")
m2.metric("per0", f"{per0:.1f}°")
m3.metric("depth", f"{metrics['depth']:.4f}")
m4.metric("min flux", f"{metrics['min_flux']:.4f}")
m5.metric("half-depth width", f"{metrics['eclipse_width_half_depth']:.3f}")

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(bundle.phase_grid, flux, lw=2, label="Emulator")
ax.axhline(1.0, ls="--", lw=0.8)
ax.set_xlabel("Phase")
ax.set_ylabel("Normalized flux")
ax.set_title("Emulated light curve")
ax.legend()
if y_limits:
    pad = max(0.02, 0.15 * metrics["depth"])
    ax.set_ylim(max(0, metrics["min_flux"] - pad), 1.0 + pad)
st.pyplot(fig, clear_figure=True)

with st.expander("Parameter vector"):
    display = pd.DataFrame({"label": bundle.label_names, "value": label_vector})
    st.dataframe(display, hide_index=True, use_container_width=True)

with st.expander("Compare two models"):
    st.write("Make a second curve by perturbing one parameter from the current setting.")
    p_name = st.selectbox("Parameter to change", bundle.label_names, index=2)
    p_lo, p_hi = range_for(p_name, fallback_ranges[p_name])
    p2 = st.slider(f"Comparison {p_name}", p_lo, p_hi, float(values[p_name]), step=(p_hi-p_lo)/500, key="compare_slider")

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
