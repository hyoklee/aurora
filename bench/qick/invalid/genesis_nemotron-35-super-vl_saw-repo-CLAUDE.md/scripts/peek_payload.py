"""Peek at the byte-string payloads (I_e, I_g, Q_e, Q_g) to determine their encoding."""
import h5py
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ss_ge.h5"

with h5py.File(path, "r") as f:
    for q in ["Q1", "Q2"]:
        grp = f[q]
        print(f"===== {q} =====")
        for k in ["Angle", "Fidelity", "Batch Num", "Round Num", "Dates"]:
            print(f"  {k} = {grp[k][()]}")
        print(f"  Exp Config (first 300): {grp['Exp Config'][()][:300]!r}")
        print(f"  Syst Config (first 300): {grp['Syst Config'][()][:300]!r}")
        for k in ["I_e", "I_g", "Q_e", "Q_g"]:
            raw = grp[k][()]
            print(f"  {k}: len={len(raw)}, first 120 bytes: {raw[:120]!r}")
