"""Ground truth: which MultiSenseGolf sensor values separate Beginner from
Professional golfers, read either from clio-core (after `clio-cae golf_omni.yaml`)
or straight from the staged HDF5 files (no clio-core), with read timings.

usage: python golf_truth.py --source cte|h5 [--h5-root DIR] [--out DIR] [--complete-only]
  cte: needs `source clio_env.sh` (runtime up, data ingested)
  h5 : reads <h5-root>/<zip>/<SwingNN>/<file>.hdf5 (the CAE staging cache)
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
from scipy import stats

JOINTS = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg",
          "LeftFoot", "Spine", "Spine1", "Spine2", "Neck", "Neck1", "Head",
          "RightShoulder", "RightArm", "RightForeArm", "RightHand",
          "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]
FS_PNS = 96.0
STREAMS = {
    "angvel": "pns-joint-angular-velocity/velocity-values/data",
    "accel": "pns-joint-synthetic-accel/acceleration-values/data",
    "quat": "pns-joint-quaternion/angle-values/data",
    "rootpos": "pns-joint-root-relative-position/cm-values/data",
    "pL": "wireless-insole-left/pressure-data/data",
    "pR": "wireless-insole-right/pressure-data/data",
    "copL": "wireless-insole-left/cop/data",
    "copR": "wireless-insole-right/cop/data",
    "hgyro": "pupil-imu-gyro/velocity-values/data",
    "hacc": "pupil-imu-accel/acceleration-values/data",
    "gaze": "pupil-gaze-xy/gaze-values/data",
}
GROUP = {**{f"P{i:02d}": "Beginner" for i in range(1, 9)},
         **{f"P{i:02d}": "Intermediate" for i in range(9, 17)},
         **{f"P{i:02d}": "Professional" for i in range(17, 25)}}


# ---------------------------------------------------------------- sources
class CteSource:
    name = "clio-core (CTE)"

    def __init__(self):
        import cte_golf as cg
        self.cg = cg
        cg.connect()

    def swings(self):
        rows = [r for r in self.cg.manifest() if r["path"].endswith(".hdf5")]
        return [(r["path"], r["tag"]) for r in rows]

    def read(self, handle, ds):
        return self.cg.read_array(handle + "/" + ds)

    def csv(self, name):
        return self.cg.read_csv("golf/Documentation.zip/" + name)


class H5Source:
    name = "HDF5 files (h5py)"

    def __init__(self, root):
        import h5py
        self.h5py, self.root = h5py, root

    def swings(self):
        out = []
        for d, _, fs in os.walk(self.root):
            for f in fs:
                if f.endswith("_stream_data.hdf5"):
                    p = os.path.join(d, f)
                    out.append((os.path.relpath(p, self.root), p))
        return sorted(out)

    def read(self, handle, ds):
        with self.h5py.File(handle, "r") as f:
            return f[ds][()]

    def read_many(self, handle, dss):  # one open per swing file
        with self.h5py.File(handle, "r") as f:
            return {k: f[ds][()] for k, ds in dss.items()}

    def csv(self, name):
        return pd.read_csv(os.path.join(self.root, "Documentation.zip", name),
                           encoding="utf-8-sig")


# ---------------------------------------------------------------- features
def joints3(a):
    return a.reshape(a.shape[0], 21, 3)


def rot_range_deg(q):
    """Largest rotation (deg) of each joint away from its first-frame pose."""
    q = q.reshape(q.shape[0], 21, 4).astype(np.float64)
    q /= np.linalg.norm(q, axis=2, keepdims=True)
    dot = np.abs(np.sum(q * q[:1], axis=2)).clip(0, 1)
    return np.degrees(2 * np.arccos(dot)).max(axis=0)


ADDRESS_FRAMES = 24  # first 0.25 s at 96 Hz: the golfer stands still at address
# A recording reaches the finish if it runs this long after the strike and the body
# has settled by the end. Some golfers' recordings stop ~0.6-0.8 s after the strike
# with the body still moving (P01, P07, P08, P10, P18), which cuts every
# whole-swing range (rotation, head rise) short.
MIN_AFTER_STRIKE_S = 1.0
MAX_END_MOTION = 0.10


def facing_frame(p):
    """Positions (T, 21, 3) in the golfer's own frame at address, in cm.

    The capture frame is y-up, but each golfer stood at a different spot and
    faced a slightly different way, and the inertial suit's global yaw drifts
    between swings (hip-line heading varies 20-30 deg across swings for P07,
    P16, P23). So the heading is taken per swing: the horizontal hip line
    (LeftUpLeg - RightUpLeg) averaged over the address frames. Every golfer in
    the dataset swings right-handed, so that line points toward the target.
    Returns positions relative to the hips' address position, with axes
    0 = toward the target, 1 = up, 2 = toward the ball (target x up).
    """
    lu, ru = JOINTS.index("LeftUpLeg"), JOINTS.index("RightUpLeg")
    line = (p[:ADDRESS_FRAMES, lu] - p[:ADDRESS_FRAMES, ru]).mean(axis=0)
    t = np.array([line[0], 0.0, line[2]])
    t /= np.linalg.norm(t)
    up = np.array([0.0, 1.0, 0.0])
    axes = np.stack([t, up, np.cross(t, up)])  # rows: target, up, toward ball
    origin = p[:ADDRESS_FRAMES, 0].mean(axis=0)
    return (p - origin) @ axes.T


def path_len(xy):
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))) if len(xy) > 1 else np.nan


def swing_features(d):
    f = {}
    av, ac, q, rp = d["angvel"], d["accel"], d["quat"], d["rootpos"]
    if len(av) > 2:
        f["swing_duration_s"] = len(av) / FS_PNS
        w = np.linalg.norm(joints3(av), axis=2).max(axis=0)
        a = np.linalg.norm(joints3(ac), axis=2).max(axis=0) if len(ac) else np.full(21, np.nan)
        r = rot_range_deg(q)
        for j, n in enumerate(JOINTS):
            f[f"imc_peak_angvel/{n}"] = w[j]
            f[f"imc_peak_accel/{n}"] = a[j]
            f[f"imc_rot_range_deg/{n}"] = r[j]
        p = joints3(rp)
        for n in ("LeftHand", "RightHand", "Head"):
            j = JOINTS.index(n)
            v = np.linalg.norm(np.diff(p[:, j], axis=0), axis=1) * FS_PNS
            f[f"imc_peak_speed_cm_s/{n}"] = v.max()
        # "root-relative" positions are global capture-frame positions (the
        # Hips row is the hips' own global position), so subtract the hips.
        rel = p - p[:, :1]
        f["imc_hip_to_head_cm"] = np.median(np.linalg.norm(rel[:, JOINTS.index("Head")], axis=1))
        f["imc_hip_to_rfoot_cm"] = np.median(np.linalg.norm(rel[:, JOINTS.index("RightFoot")], axis=1))
        f["imc_hips_travel_cm"] = np.ptp(p[:, 0], axis=0).max()
        if len(p) > ADDRESS_FRAMES:
            gf = facing_frame(p)  # target, up, toward ball; 0 = hips at address
            hips, head = gf[:, 0], gf[:, JOINTS.index("Head")]
            for k, axis in enumerate(("target", "up", "toward_ball")):
                f[f"imc_hips_range_{axis}_cm"] = np.ptp(hips[:, k])
                f[f"imc_head_range_{axis}_cm"] = np.ptp(head[:, k])
            f["imc_hips_max_toward_target_cm"] = hips[:, 0].max()  # slide toward the target
            f["imc_hips_max_away_target_cm"] = -hips[:, 0].min()  # sway away from it
            f["imc_head_dip_cm"] = head[:ADDRESS_FRAMES, 1].mean() - head[:, 1].min()
        # Recording window (not ranked): seconds after the strike (peak trail-hand
        # acceleration) and how much the body still moves in the last 0.25 s,
        # as a share of the swing's peak mean joint speed. The derived acceleration
        # is 8 frames shorter than the positions; that 0.08 s is ignored here.
        if len(ac) > ADDRESS_FRAMES and len(p) > ADDRESS_FRAMES:
            strike = int(np.argmax(np.linalg.norm(joints3(ac)[:, JOINTS.index("RightHand")], axis=1)))
            f["rec_after_strike_s"] = (len(p) - strike) / FS_PNS
            speed = np.linalg.norm(np.diff(p, axis=0), axis=2).mean(axis=1)
            f["rec_end_motion_frac"] = speed[-ADDRESS_FRAMES:].mean() / speed.max()
    pL, pR = d["pL"], d["pR"]
    if len(pL) > 2 and len(pR) > 2 and len(pL) == len(pR):
        L, R = pL.reshape(len(pL), -1).sum(1), pR.reshape(len(pR), -1).sum(1)
        tot = L + R
        frac = np.where(tot > 0, L / np.where(tot > 0, tot, 1), np.nan)  # lead (left) share
        f["insole_peak_total_load"] = tot.max()
        f["insole_load_cv"] = tot.std() / tot.mean() if tot.mean() > 0 else np.nan
        f["insole_lead_share_range"] = np.nanmax(frac) - np.nanmin(frac)
        f["insole_lead_share_end_minus_start"] = frac[-1] - frac[0]
        thr = 0.05 * max(pL.max(), pR.max(), 1)
        f["insole_peak_contact_cells"] = ((pL > thr).reshape(len(pL), -1).sum(1)
                                          + (pR > thr).reshape(len(pR), -1).sum(1)).max()
        for side, c in (("L", d["copL"]), ("R", d["copR"])):
            c = c[np.all(np.isfinite(c), axis=1)] if len(c) else c
            if len(c) > 2:
                f[f"insole_cop_path_{side}"] = path_len(c)
                f[f"insole_cop_range_x_{side}"] = np.ptp(c[:, 0])
                f[f"insole_cop_range_y_{side}"] = np.ptp(c[:, 1])
    g, h = d["hgyro"], d["hacc"]
    if len(g) > 5:
        f["head_peak_gyro"] = np.linalg.norm(g, axis=1).max()
    if len(h) > 5:
        f["head_accel_std"] = np.linalg.norm(h, axis=1).std()
    z = d["gaze"]
    if len(z) > 5:
        f["gaze_std_x"], f["gaze_std_y"] = z[:, 0].std(), z[:, 1].std()
    return f


def participant(path):
    return re.search(r"(P\d\d)_Swing", path).group(1)


# ---------------------------------------------------------------- stats
def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (b.mean() - a.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)) if sp > 0 else np.nan


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    out = np.empty_like(q)
    out[o] = q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cte", "h5"], required=True)
    ap.add_argument("--h5-root")
    ap.add_argument("--out", default=".")
    ap.add_argument("--skip-stats", action="store_true")
    ap.add_argument("--complete-only", action="store_true",
                    help="rank only recordings that reach a settled finish "
                         f"(>= {MIN_AFTER_STRIKE_S} s after the strike, last 0.25 s "
                         f"<= {MAX_END_MOTION:.0%} of peak motion)")
    a = ap.parse_args()
    src = CteSource() if a.source == "cte" else H5Source(a.h5_root)

    t0 = time.perf_counter()
    swings = src.swings()
    t_list = time.perf_counter() - t0
    t_read = t_feat = 0.0
    nbytes = 0
    rows = []
    for path, handle in swings:
        t1 = time.perf_counter()
        d = (src.read_many(handle, STREAMS) if hasattr(src, "read_many")
             else {k: src.read(handle, ds) for k, ds in STREAMS.items()})
        t2 = time.perf_counter()
        nbytes += sum(v.nbytes for v in d.values())
        f = swing_features(d)
        t_read += t2 - t1
        t_feat += time.perf_counter() - t2
        f["participant"] = participant(path)
        rows.append(f)
    t1 = time.perf_counter()
    meta = src.csv("Participant Metadata.csv")
    t_read += time.perf_counter() - t1
    timing = dict(source=src.name, swings=len(swings), datasets=len(swings) * len(STREAMS),
                  bytes_read=nbytes, list_s=round(t_list, 3), read_s=round(t_read, 3),
                  features_s=round(t_feat, 3))
    print(json.dumps(timing))
    os.makedirs(a.out, exist_ok=True)
    json.dump(timing, open(os.path.join(a.out, f"timing_{a.source}.json"), "w"), indent=1)
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(a.out, f"swing_features_{a.source}.csv"), index=False)
    if a.skip_stats:
        return

    name = "ranking.csv"
    if a.complete_only:
        ok = (sw.rec_after_strike_s >= MIN_AFTER_STRIKE_S) & (sw.rec_end_motion_frac <= MAX_END_MOTION)
        kept = ok.groupby(sw.participant.map(GROUP)).agg(["sum", "count"])
        print("complete recordings kept:", kept.to_dict("index"))
        sw, name = sw[ok], "ranking_complete.csv"
    sw = sw.drop(columns=[c for c in sw.columns if c.startswith("rec_")])
    per = sw.groupby("participant").median(numeric_only=True)
    nsw = sw.groupby("participant").count()
    per = per.where(nsw >= 5)  # a participant needs >= 5 swings with the feature
    per["group"] = per.index.map(GROUP)
    height = meta.set_index("Participant Number")["Height (cm)"]
    res = []
    for c in per.columns.drop("group"):
        b = per.loc[per.group == "Beginner", c].dropna()
        p = per.loc[per.group == "Professional", c].dropna()
        i = per.loc[per.group == "Intermediate", c].dropna()
        if len(b) < 5 or len(p) < 5:
            continue
        u, pv = stats.mannwhitneyu(p, b, alternative="two-sided")
        both = pd.concat([b, p])
        hh = height.reindex(both.index)
        rh = stats.spearmanr(both, hh).statistic if both.nunique() > 1 else np.nan
        res.append(dict(feature=c, hedges_g=hedges_g(b.values, p.values),
                        auc=u / (len(b) * len(p)), mw_p=pv,
                        beginner_median=b.median(), intermediate_median=i.median(),
                        professional_median=p.median(), n_beg=len(b), n_pro=len(p),
                        spearman_with_height=rh))
    r = pd.DataFrame(res)
    ok = r.mw_p.notna()
    r["mw_q_bh"] = np.nan
    r.loc[ok, "mw_q_bh"] = bh(r.mw_p[ok])
    r["abs_g"] = r.hedges_g.abs()
    r = r.sort_values("abs_g", ascending=False).reset_index(drop=True)
    r.index += 1
    r.to_csv(os.path.join(a.out, name), index_label="rank")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(r.head(25).round(3).to_string())


if __name__ == "__main__":
    main()
