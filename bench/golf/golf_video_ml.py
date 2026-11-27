"""Video patterns that separate Beginner from Professional golfers (MultiSenseGolf).

Reads the per-swing video HDF5 files that `clio-cae golf_video_omni.yaml` made
(/frames: uint8 luma [T, H, W] compressed with the FFMPEG HDF5 filter, /time_s)
either from clio-core (--source cte) or from the CAE staging cache (--source dir),
and runs four analyses:

  1. optical-flow motion profile (front camera): swing speed, tempo, smoothness
  2. head-camera ego-motion (FPV camera): how much the head moves
  3. pretrained video embeddings (R(2+1)D-18, Kinetics-400) + logistic regression,
     leave-one-golfer-out
  4. confound control: the same classifier on one still frame per swing
     (ResNet-18, ImageNet). A still frame carries no swing, so if it separates
     the groups as well, the model is reading the scene or the person.

Hand-crafted features (1, 2) are ranked like golf_truth.py: per-golfer medians,
8 vs 8, Hedges' g, Mann-Whitney p, BH q. Classifiers (3, 4) are scored at the
golfer level (mean probability over that golfer's swings) with a label-permutation
p-value.

usage: python golf_video_ml.py --source cte|dir [--root DIR] [--out DIR]
         [--device cpu|xpu|cuda|auto] [--flow cpu|ocl] [--clip-frames N] [--clip-size N]
  cte: needs clio_env.sh (runtime up, golf_video_omni.yaml ingested)
  dir: reads <root>/**/*.mp4.hdf5 (the CAE staging cache)
  --device: where the embedding networks run (auto: xpu, then cuda, then cpu)
  --flow ocl: Farneback optical flow through OpenCV's OpenCL path (a GPU if the
    OpenCL runtime has one); cpu is the default. Workers are spawned, not forked
    (an OpenCL context does not survive a fork), and worker i uses OpenCL GPU
    i mod --ocl-gpus.
Per-stage wall times go to video_ml.json under "timing".
"""
import argparse
import io
import json
import os
import re
import socket
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import cv2
import h5ffmpeg  # noqa: F401  registers the FFMPEG HDF5 filter with h5py
import h5py
import numpy as np
import pandas as pd
from scipy import stats

GROUP = {**{f"P{i:02d}": "Beginner" for i in range(1, 9)},
         **{f"P{i:02d}": "Professional" for i in range(17, 25)}}
FLOW_W = 160  # optical flow at 160 px wide (half the stored width)
FLOW_OCL = False  # set in each worker by ocl_worker_init()


def ocl_worker_init(counter, n_gpus):
    """Pool initializer for --flow ocl: pick this worker's GPU, then open OpenCL."""
    global FLOW_OCL
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    os.environ["OPENCV_OPENCL_DEVICE"] = f":GPU:{i % n_gpus}"
    cv2.ocl.setUseOpenCL(True)
    FLOW_OCL = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()


def ocl_device(_):
    dev = cv2.ocl.Device.getDefault().name()
    return f"{os.getpid()} {dev} {os.environ['OPENCV_OPENCL_DEVICE']}"


# ---------------------------------------------------------------- sources
def list_videos(a):
    """[(participant, swing, view, handle)] for every stored video HDF5."""
    out = []
    if a.source == "cte":
        import cte_golf as cg
        cg.connect()
        # Read every blob here, in the one connected process: a client forked
        # after connect() hangs on the runtime's shared memory, so the worker
        # processes get bytes, never a tag.
        for r in cg.manifest("golfvideo"):
            if r["path"].endswith(".mp4.hdf5"):
                out.append((r["path"], cg.read_bytes(r["tag"])))
    else:
        for d, _, fs in os.walk(a.root):
            for f in fs:
                if f.endswith(".mp4.hdf5"):
                    p = os.path.join(d, f)
                    out.append((os.path.relpath(p, a.root), p))
    rows = []
    for path, handle in sorted(out, key=lambda o: o[0]):
        m = re.search(r"(P\d\d)(?:_\d)?\.zip/(Swing\d+)/", path)
        if m and m.group(1) in GROUP:
            view = "front" if "Front_View" in path else "fpv"
            rows.append((m.group(1), m.group(2), view, handle))
    return rows


def open_video(source, handle):
    if source == "cte":  # handle holds the HDF5 file's bytes (list_videos)
        f = h5py.File(io.BytesIO(handle), "r")
    else:
        f = h5py.File(handle, "r")
    with f:
        return f["frames"][()], f["time_s"][()], float(f["frames"].attrs["fps"])


# ---------------------------------------------------------------- 1. front motion
def flow_series(frames):
    """Per frame pair: motion energy and moving area."""
    h = int(frames.shape[1] * FLOW_W / frames.shape[2])
    small = [cv2.resize(f, (FLOW_W, h), interpolation=cv2.INTER_AREA) for f in frames]
    e, area = [], []
    if FLOW_OCL:
        small = [cv2.UMat(f) for f in small]
    for a, b in zip(small[:-1], small[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        if FLOW_OCL:
            fl = fl.get()
        mag = np.linalg.norm(fl, axis=2)
        moving = mag > 1.0
        e.append(mag[moving].sum() / mag.size)
        area.append(moving.mean())
    return np.array(e), np.array(area)


def front_features(frames, fps):
    e, area = flow_series(frames)
    f = {}
    if len(e) < 10 or e.max() <= 0:
        return f
    e_s = np.convolve(e, np.ones(3) / 3, mode="same")
    peak = int(np.argmax(e_s))
    thr = 0.1 * e_s[peak]
    onset = int(np.argmax(e_s > thr))
    f["vid_peak_motion"] = e_s[peak] * fps  # px/s-weighted motion at the strike
    f["vid_mean_motion"] = e.mean() * fps
    f["vid_peak_moving_area"] = area.max()
    # Top of the backswing: the slowest moment between onset and the strike.
    if peak - onset >= 4:
        top_i = onset + 1 + int(np.argmin(e_s[onset + 1:peak]))
        back, down = (top_i - onset) / fps, (peak - top_i) / fps
        if down > 0:
            f["vid_backswing_s"] = back
            f["vid_downswing_s"] = down
            f["vid_tempo_ratio"] = back / down
    # Smoothness: local maxima of motion energy before the strike (fewer = smoother).
    pre = e_s[onset:peak + 1]
    if len(pre) > 3:
        f["vid_motion_peaks_before_strike"] = int(
            np.sum((pre[1:-1] > pre[:-2]) & (pre[1:-1] > pre[2:]) & (pre[1:-1] > thr)))
    jerk = np.diff(e_s, 2)
    f["vid_motion_jerk"] = np.abs(jerk).mean() / max(e_s.mean(), 1e-9)
    return f


# ---------------------------------------------------------------- 2. FPV ego-motion
def fpv_features(frames, fps):
    """Camera (head) motion from tracked corners: shift and rotation per frame."""
    f = {}
    shift, rot = [], []
    for a, b in zip(frames[:-1], frames[1:]):
        p0 = cv2.goodFeaturesToTrack(a, 200, 0.01, 8)
        if p0 is None or len(p0) < 8:
            shift.append(np.nan), rot.append(np.nan)
            continue
        p1, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None)
        ok = ok.ravel() == 1
        if ok.sum() < 8:
            shift.append(np.nan), rot.append(np.nan)
            continue
        m, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok])
        if m is None:
            shift.append(np.nan), rot.append(np.nan)
            continue
        shift.append(np.hypot(m[0, 2], m[1, 2]) / frames.shape[2])  # frame widths
        rot.append(abs(np.degrees(np.arctan2(m[1, 0], m[0, 0]))))
    shift, rot = np.array(shift) * fps, np.array(rot) * fps
    if np.isfinite(shift).sum() < 5:
        return f
    f["fpv_peak_shift_per_s"] = np.nanmax(shift)
    f["fpv_mean_shift_per_s"] = np.nanmean(shift)
    f["fpv_peak_roll_deg_s"] = np.nanmax(rot)
    f["fpv_total_path"] = np.nansum(shift) / fps
    return f


def one_swing(args):
    source, p, swing, view, handle = args
    frames, t, fps = open_video(source, handle)
    f = front_features(frames, fps) if view == "front" else fpv_features(frames, fps)
    return dict(participant=p, swing=swing, view=view, frames=len(frames), **f)


# ---------------------------------------------------------------- 3, 4. embeddings
def pick_device(name):
    import torch
    if name == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def sync(dev):
    import torch
    if dev == "xpu":
        torch.xpu.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def embed(rows, source, dev, n_frames=16, size=112, batch=16):
    """R(2+1)D-18 clip embedding and ResNet-18 first-frame embedding per front swing.

    Returns (embeddings, timing): clips are decoded and resized on the CPU
    ("prep"), then run through both networks on `dev` in batches ("model",
    after one warm-up batch that is timed apart as "warmup").
    """
    import torch
    import torchvision.models as tvm
    import torchvision.models.video as tvv

    torch.set_num_threads(min(32, os.cpu_count()))
    tm = {}
    t = time.perf_counter()
    vid = tvv.r2plus1d_18(weights=tvv.R2Plus1D_18_Weights.KINETICS400_V1).eval()
    vid.fc = torch.nn.Identity()
    img = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1).eval()
    img.fc = torch.nn.Identity()
    vid, img = vid.to(dev), img.to(dev)
    tm["load"] = time.perf_counter() - t

    t = time.perf_counter()
    clips, stills = [], []
    for p, swing, view, handle in rows:
        frames, _, _ = open_video(source, handle)
        idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
        clips.append(np.stack([cv2.resize(frames[i], (size, size),
                                          interpolation=cv2.INTER_AREA) for i in idx]))
        stills.append(cv2.resize(frames[0], (224, 224), interpolation=cv2.INTER_AREA))
    clips = torch.from_numpy(np.stack(clips))   # [N, T, S, S] uint8
    stills = torch.from_numpy(np.stack(stills))  # [N, 224, 224] uint8
    tm["prep"] = time.perf_counter() - t

    v_mean = torch.tensor([0.43216, 0.394666, 0.37645], device=dev).view(1, 3, 1, 1, 1)
    v_std = torch.tensor([0.22803, 0.22145, 0.216989], device=dev).view(1, 3, 1, 1, 1)
    i_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    i_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def run(lo, hi):
        x = clips[lo:hi].to(dev).float().div(255)[:, None].expand(-1, 3, -1, -1, -1)
        s = stills[lo:hi].to(dev).float().div(255)[:, None].expand(-1, 3, -1, -1)
        return (vid((x - v_mean) / v_std).cpu().numpy(),
                img((s - i_mean) / i_std).cpu().numpy())

    ev, ei = [], []
    with torch.no_grad():
        t = time.perf_counter()
        run(0, min(batch, len(rows)))  # warm-up: kernel compilation, allocator
        sync(dev)
        tm["warmup"] = time.perf_counter() - t
        t = time.perf_counter()
        for lo in range(0, len(rows), batch):
            a, b = run(lo, lo + batch)
            ev.append(a)
            ei.append(b)
        sync(dev)
        tm["model"] = time.perf_counter() - t
    ev, ei = np.concatenate(ev), np.concatenate(ei)
    out = [dict(participant=r[0], swing=r[1], video=ev[i], still=ei[i])
           for i, r in enumerate(rows)]
    return out, {k: round(v, 2) for k, v in tm.items()}


def logo_auc(X, golfers, y_golfer, n_perm=1000, seed=0):
    """Leave-one-golfer-out logistic regression; golfer-level AUC and perm p."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ids = sorted(set(golfers))
    golfers = np.asarray(golfers)

    def golfer_scores(labels):
        y = np.array([labels[g] for g in golfers])
        sc = {}
        for g in ids:
            tr, te = golfers != g, golfers == g
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
            clf.fit(X[tr], y[tr])
            sc[g] = clf.predict_proba(X[te])[:, 1].mean()
        return sc

    true = {g: y_golfer[g] for g in ids}
    sc = golfer_scores(true)
    auc = roc_auc_score([true[g] for g in ids], [sc[g] for g in ids])
    rng = np.random.default_rng(seed)
    perm = []
    lab = np.array([true[g] for g in ids])
    for _ in range(n_perm):
        sh = dict(zip(ids, rng.permutation(lab)))
        s2 = golfer_scores(sh)
        perm.append(roc_auc_score([sh[g] for g in ids], [s2[g] for g in ids]))
    p = (1 + np.sum(np.array(perm) >= auc)) / (1 + n_perm)
    return auc, p, sc


# ---------------------------------------------------------------- stats
def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (b.mean() - a.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)) if sp > 0 else np.nan


def rank_features(sw):
    per = sw.groupby("participant").median(numeric_only=True)
    n = sw.groupby("participant").count()
    per = per.where(n >= 3)
    per["group"] = per.index.map(GROUP)
    res = []
    for c in per.columns.drop(["group", "frames"], errors="ignore"):
        b = per.loc[per.group == "Beginner", c].dropna()
        p = per.loc[per.group == "Professional", c].dropna()
        if len(b) < 4 or len(p) < 4:
            continue
        u, pv = stats.mannwhitneyu(p, b, alternative="two-sided")
        res.append(dict(feature=c, hedges_g=hedges_g(b.values, p.values),
                        auc=u / (len(b) * len(p)), mw_p=pv, beginner_median=b.median(),
                        professional_median=p.median(), n_beg=len(b), n_pro=len(p)))
    return pd.DataFrame(res)


def bh_q(p):
    """Benjamini-Hochberg q-values for the p-values in p (NaN stays NaN)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    o = ok[np.argsort(p[ok])]
    v = p[o] * len(o) / (np.arange(len(o)) + 1)
    q[o] = np.minimum.accumulate(v[::-1])[::-1].clip(0, 1)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cte", "dir"], required=True)
    ap.add_argument("--root")
    ap.add_argument("--out", default=".")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--device", default="cpu", choices=["cpu", "xpu", "cuda", "auto"])
    ap.add_argument("--flow", default="cpu", choices=["cpu", "ocl"])
    ap.add_argument("--ocl-gpus", type=int, default=6)
    ap.add_argument("--clip-frames", type=int, default=16)
    ap.add_argument("--clip-size", type=int, default=112)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    timing = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    vids = list_videos(a)
    timing["read"] = time.perf_counter() - t0
    print(f"{len(vids)} videos ({sum(v[2] == 'front' for v in vids)} front, "
          f"{sum(v[2] == 'fpv' for v in vids)} fpv) read in {timing['read']:.1f} s")
    t0 = time.perf_counter()
    # One BLAS/OpenMP thread per worker: 32 spawned workers each starting 64
    # OpenBLAS threads exhaust the job's thread limit (pthread_create EAGAIN).
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    pool = dict(max_workers=a.workers)
    if a.flow == "ocl":  # the parent never opens OpenCL
        pool.update(mp_context=mp.get_context("spawn"), initializer=ocl_worker_init,
                    initargs=(mp.get_context("spawn").Value("i", 0), a.ocl_gpus))
    with ProcessPoolExecutor(**pool) as ex:
        if a.flow == "ocl":
            devs = sorted(set(ex.map(ocl_device, range(a.workers * 4))))
            print(f"OpenCL workers: {len(devs)}; e.g. {devs[:2]}")
            timing["flow_device"] = devs[0].split(" ", 1)[1] if devs else ""
        rows = list(ex.map(one_swing, [(a.source, *v) for v in vids], chunksize=2))
    timing["features"] = time.perf_counter() - t0
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(a.out, "video_swing_features.csv"), index=False)
    front = sw[sw.view == "front"].drop(columns=["view", "swing"])
    fpv = sw[sw.view == "fpv"].drop(columns=["view", "swing"])
    rank = pd.concat([rank_features(front.dropna(axis=1, how="all")),
                      rank_features(fpv.dropna(axis=1, how="all"))], ignore_index=True)
    rank["mw_q_bh"] = bh_q(rank.mw_p)  # one family: every front and FPV feature
    order = rank.hedges_g.abs().sort_values(ascending=False).index
    rank = rank.reindex(order).reset_index(drop=True)
    rank.index += 1
    rank.to_csv(os.path.join(a.out, "video_ranking.csv"), index_label="rank")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(rank.round(3).to_string())

    dev = pick_device(a.device)
    t1 = time.perf_counter()
    emb, tm = embed([v for v in vids if v[2] == "front"], a.source, dev,
                    a.clip_frames, a.clip_size, a.batch)
    timing["embedding"] = time.perf_counter() - t1
    timing["embedding_stages"] = tm
    print(f"embeddings on {dev}: {tm}")
    golfers = [e["participant"] for e in emb]
    y = {g: int(GROUP[g] == "Professional") for g in set(golfers)}
    summary = dict(videos=len(vids), host=socket.gethostname(), device=dev,
                   flow=a.flow, workers=a.workers, clip_frames=a.clip_frames,
                   clip_size=a.clip_size, timing=timing, classifiers={})
    t1 = time.perf_counter()
    for name in ("video", "still"):
        X = np.stack([e[name] for e in emb])
        auc, p, sc = logo_auc(X, golfers, y, n_perm=a.perms)
        summary["classifiers"][name] = dict(
            golfer_auc=round(auc, 3), perm_p=round(p, 4),
            scores={g: round(float(s), 3) for g, s in sorted(sc.items())})
        print(f"{name}: leave-one-golfer-out golfer AUC {auc:.3f}, permutation p {p:.4f}")
    timing["classifiers"] = time.perf_counter() - t1
    timing["total"] = time.perf_counter() - t_all
    for k, v in timing.items():
        if isinstance(v, float):
            timing[k] = round(v, 2)
    print("timing (s):", timing)
    json.dump(summary, open(os.path.join(a.out, "video_ml.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
