"""Diagnose how the stored 'Fidelity' for Q1 might have been computed.

Tries several simple definitions and compares with stored 0.8365.
"""
import sys

import h5py
import numpy as np

sys.path.insert(0, "scripts")
from analyze import decode_payload, rotate, fidelity_from_contrast

H5_PATH = "ss_ge.h5"


def simple_fidelity(xg, xe, threshold):
    pe_g = np.mean(xg > threshold)
    pg_e = np.mean(xe <= threshold)
    return 1.0 - 0.5 * (pe_g + pg_e), pe_g, pg_e


def main():
    with h5py.File(H5_PATH, "r") as f:
        grp = f["Q1"]
        angle = float(grp["Angle"][()].ravel()[0])
        stored = float(grp["Fidelity"][()].ravel()[0])
        Ie = decode_payload(grp["I_e"][()])
        Ig = decode_payload(grp["I_g"][()])
        Qe = decode_payload(grp["Q_e"][()])
        Qg = decode_payload(grp["Q_g"][()])

    Ie_r, Qe_r = rotate(Ie, Qe, angle)
    Ig_r, Qg_r = rotate(Ig, Qg, angle)

    print(f"stored fidelity Q1 = {stored:.4f}  ({stored*100:.2f}%)")
    print(f"rotation angle = {angle:+.4f} rad\n")

    variants = {
        "raw I, thr=0": simple_fidelity(Ig, Ie, 0.0),
        "raw I, optimal": fidelity_from_contrast(Ig, Ie),
        "rotated I, thr=0": simple_fidelity(Ig_r, Ie_r, 0.0),
        "rotated I, optimal": fidelity_from_contrast(Ig_r, Ie_r),
        "raw Q, thr=0": simple_fidelity(Qg, Qe, 0.0),
        "raw Q, optimal": fidelity_from_contrast(Qg, Qe),
        "rotated Q, thr=0": simple_fidelity(Qg_r, Qe_r, 0.0),
        "rotated Q, optimal": fidelity_from_contrast(Qg_r, Qe_r),
    }
    for name, val in variants.items():
        if len(val) == 3:
            F, pe_g, pg_e = val
            print(f"{name:22} F={F:.4f}  (P(e|g)={pe_g:.4f}, P(g|e)={pg_e:.4f})")
        else:
            F, t = val
            print(f"{name:22} F={F:.4f}  (thr={t:.2f})")

    # Check: maybe stored fidelity uses threshold at midpoint of medians on raw I
    med_g, med_e = np.median(Ig), np.median(Ie)
    thr_mid = 0.5 * (med_g + med_e)
    F, pe_g, pg_e = simple_fidelity(Ig, Ie, thr_mid)
    print(f"raw I, thr=midpoint of medians ({thr_mid:.2f}): F={F:.4f}")

    # Maybe stored is accuracy with threshold at 0 and no rotation on (I,Q) combined?
    # Print medians for context
    print(f"\nraw medians: g I={np.median(Ig):.2f} Q={np.median(Qg):.2f} | e I={np.median(Ie):.2f} Q={np.median(Qe):.2f}")
    print(f"rot medians: g I={np.median(Ig_r):.2f} Q={np.median(Qg_r):.2f} | e I={np.median(Ie_r):.2f} Q={np.median(Qe_r):.2f}")


if __name__ == "__main__":
    main()
