"""Utilities for loading and evaluating a theborg PHOEBE light-curve emulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

LABEL_NAMES = ["r1_over_a", "r2_over_a", "incl", "sbratio", "q", "ecosw", "esinw"]
DEFAULT_LABEL_RANGES = {
    "r1_over_a": (0.02, 0.45),
    "r2_over_a": (0.02, 0.45),
    "incl": (65.0, 90.0),
    "sbratio": (0.05, 2.0),
    "q": (0.1, 1.0),
    "ecosw": (-0.6, 0.6),
    "esinw": (-0.6, 0.6),
}
NPHASE = 201
PHASE_GRID = np.linspace(-0.5, 0.5, NPHASE, endpoint=False)


@dataclass
class EmulatorBundle:
    emulator: object
    label_names: list[str]
    training_labels: np.ndarray | None
    label_min: np.ndarray | None
    label_max: np.ndarray | None
    phase_grid: np.ndarray


def _coerce_label_names(em: object) -> list[str]:
    names = getattr(em, "label_names", None)
    if names is None:
        names = getattr(em, "labels", None)
    if names is None:
        return LABEL_NAMES.copy()
    return [str(x) for x in names]


def load_emulator(path: str | Path) -> EmulatorBundle:
    """Load a theborg emulator pickle and infer label ranges from its training labels."""
    from theborg import emulator as theborg_emulator

    em = theborg_emulator.Emulator.read(str(path))
    label_names = _coerce_label_names(em)

    training_labels = getattr(em, "training_labels", None)
    if training_labels is not None:
        training_labels = np.asarray(training_labels, dtype=float)
        label_min = np.nanmin(training_labels, axis=0)
        label_max = np.nanmax(training_labels, axis=0)
    else:
        label_min = None
        label_max = None

    return EmulatorBundle(
        emulator=em,
        label_names=label_names,
        training_labels=training_labels,
        label_min=label_min,
        label_max=label_max,
        phase_grid=PHASE_GRID,
    )


def labels_to_vector(values: dict[str, float], label_names: Iterable[str] = LABEL_NAMES) -> np.ndarray:
    """Convert a dictionary of slider values into the emulator label vector."""
    return np.array([float(values[name]) for name in label_names], dtype=float)


def vector_to_dict(vector: np.ndarray, label_names: Iterable[str] = LABEL_NAMES) -> dict[str, float]:
    return {name: float(value) for name, value in zip(label_names, vector)}


def ecc_per0_from_ecosw_esinw(ecosw: float, esinw: float) -> tuple[float, float]:
    """Return eccentricity and argument of periastron in degrees."""
    ecc = float(np.hypot(ecosw, esinw))
    per0 = float(np.degrees(np.arctan2(esinw, ecosw)) % 360.0)
    return ecc, per0


def ecosw_esinw_from_ecc_per0(ecc: float, per0_deg: float) -> tuple[float, float]:
    """Return ecosw and esinw from eccentricity and per0 in degrees."""
    omega = np.deg2rad(per0_deg)
    return float(ecc * np.cos(omega)), float(ecc * np.sin(omega))


def in_training_range(vector: np.ndarray, bundle: EmulatorBundle, pad_fraction: float = 0.0) -> tuple[bool, list[str]]:
    """Check whether a label vector lies inside the training min/max label box."""
    if bundle.label_min is None or bundle.label_max is None:
        return True, []

    width = bundle.label_max - bundle.label_min
    lo = bundle.label_min - pad_fraction * width
    hi = bundle.label_max + pad_fraction * width

    bad = []
    for i, name in enumerate(bundle.label_names):
        if vector[i] < lo[i] or vector[i] > hi[i]:
            bad.append(f"{name}: {vector[i]:.5g} outside [{lo[i]:.5g}, {hi[i]:.5g}]")
    return len(bad) == 0, bad


def predict_flux(bundle: EmulatorBundle, vector: np.ndarray) -> np.ndarray:
    """Evaluate the emulator and return a 1-D flux array."""
    flux = np.asarray(bundle.emulator(vector), dtype=float).ravel()
    if flux.size != bundle.phase_grid.size:
        raise ValueError(f"Emulator returned {flux.size} points; expected {bundle.phase_grid.size}.")
    return flux


def lightcurve_metrics(phase: np.ndarray, flux: np.ndarray) -> dict[str, float]:
    """Simple morphology metrics for one emulated light curve."""
    flux = np.asarray(flux, dtype=float)
    depth = 1.0 - np.nanmin(flux)
    primary_phase = float(phase[np.nanargmin(flux)])
    half_level = 1.0 - 0.5 * depth
    in_eclipse = flux < half_level
    dphase = float(np.nanmedian(np.diff(phase))) if len(phase) > 1 else np.nan
    width = float(np.sum(in_eclipse) * dphase) if np.isfinite(dphase) else np.nan
    return {
        "min_flux": float(np.nanmin(flux)),
        "max_flux": float(np.nanmax(flux)),
        "depth": float(depth),
        "primary_phase": primary_phase,
        "eclipse_width_half_depth": width,
    }
