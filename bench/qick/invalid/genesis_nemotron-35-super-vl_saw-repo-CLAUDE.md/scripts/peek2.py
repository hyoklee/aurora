"""Compact peek: fidelities per qubit and payload magic bytes."""
import h5py
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ss_ge.h5"

with h5py.File(path, "r") as f:
    for q in sorted(f.keys(), key=lambda k: int(k[1:])):
        grp = f[q]
        def get(dname):
            return float(grp[dname][()].ravel()[0])

        print(f"{q}: Fidelity={get('Fidelity'):.6f} "
              f"Angle={get('Angle'):+.6f} "
              f"Round={get('Round Num')} Batch={get('Batch Num')}")
        for k in ["I_e", "I_g", "Q_e", "Q_g"]:
            raw = bytes(grp[k][()])
            print(f"   {k}: len={len(raw)} magic={raw[:8]!r}")
