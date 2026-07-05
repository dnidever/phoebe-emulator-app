# PHOEBE Emulator Explorer

A small Streamlit app for interactively exploring a PHOEBE eclipsing-binary light-curve emulator trained with `theborg`.

The emulator is assumed to take these seven labels:

```text
r1_over_a, r2_over_a, incl, sbratio, q, ecosw, esinw
```

and return a normalized light curve on the phase grid `[-0.5, 0.5)` with 201 points.

## Install

```bash
cd phoebe-emulator-app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/dnidever/theborg.git
```

If `theborg` is already installed in your science environment, you can skip the last line.

## Run

```bash
streamlit run app.py
```

Then select your trained emulator `.pkl` file in the sidebar.

## Notes

- The app refuses to evaluate labels outside the emulator's training-label min/max range, unless you disable that checkbox.
- Eccentricity and argument of periastron are controlled through `ecosw` and `esinw`, but the app displays the corresponding `ecc` and `per0`.
- Use the "Compare two models" section to explore how a parameter change affects the light curve.
