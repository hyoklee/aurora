"""Compare an agent's Figure 10 grid with the reference, without editing its script.

usage (with clio_env.sh sourced, so cte_ice can reach the ice runtime):
  python verify_fig10.py <copy_of_run_dir> <ln_M_ref.npy>

Runs <dir>/fig10.py with matplotlib's contour/contourf patched to keep the
first Z array they draw (the ln M grid), then prints max |agent - reference|.
Run it on a COPY of the run directory: the script rewrites fig10.png there.
"""
import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

run_dir, ref_path = sys.argv[1:3]
grids = []


def keep(original):
    def wrapper(*args, **kwargs):
        z = kwargs.get("Z")
        arrays = [a for a in args if isinstance(a, np.ndarray) and a.ndim == 2]
        if z is None and arrays:
            z = arrays[2] if len(arrays) >= 3 else arrays[-1]
        if z is not None:
            grids.append(np.array(z, dtype=float))
        return original(*args, **kwargs)

    return wrapper


for owner in (matplotlib.axes.Axes, plt):
    for name in ("contour", "contourf"):
        setattr(owner, name, keep(getattr(owner, name)))

os.chdir(run_dir)
sys.path.insert(0, run_dir)
sys.argv = ["fig10.py"]
try:
    runpy.run_path("fig10.py", run_name="__main__")
except SystemExit:
    pass
except Exception as exc:  # e.g. a hard-coded output path; the grid may already be in hand
    print(f"fig10.py raised after {len(grids)} contour call(s): {type(exc).__name__}: {exc}")
ref = np.load(ref_path)
if not grids:
    print("no contour grid captured")
    sys.exit(1)
z = grids[0]
if z.shape != ref.shape:
    print(f"shape {z.shape} vs reference {ref.shape}")
    sys.exit(1)
print(f"max|ln M - ref| = {np.nanmax(np.abs(z - ref)):.3e}  (transpose: {np.nanmax(np.abs(z.T - ref)):.3e})")
