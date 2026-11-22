#!/usr/bin/env python
"""MultiSenseGolf: which sensor values best separate Beginner vs Professional golfers.

All data are read from the running clio-core runtime via cte_golf (no downloads).
"""
import os, re, sys, time
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cte_golf as cg

# ---- skill groups per paper Table 3 (P01-P08 Beginner, P09-P16 Intermediate,
# P17-P24 Professional). Cross-checked against Participant Metadata license field.
def group_of(pid):
    n = int(pid[1:])
    if 1 <= n <= 8:
        return 'Beginner'
    if 9 <= n <= 16:
        return 'Intermediate'
    if 17 <= n <= 24:
        return 'Professional'
    raise ValueError(pid)

JOINTS = ['Hips','RightUpLeg','RightLeg','RightFoot','LeftUpLeg','LeftLeg',
          'LeftFoot','Spine','Spine1','Spine2','Neck','Neck1','Head',
          'RightShoulder','RightArm','RightForeArm','RightHand',
          'LeftShoulder','LeftArm','LeftForeArm','LeftHand']
AXES3 = ['x','y','z']
AXES4 = ['w','x','y','z']

PNS_STREAMS = {
    'pns-joint-euler/angle-values': 3,
    'pns-joint-quaternion/angle-values': 4,
    'pns-joint-parent-relative-position/cm-values': 3,
    'pns-joint-root-relative-position/cm-values': 3,
    'pns-joint-angular-velocity/velocity-values': 3,
    'pns-joint-synthetic-accel/acceleration-values': 3,
}


def _channel_stats(arr, prefix):
    feats = {}
    if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
        return feats
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return feats
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    rng = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
    rms = np.sqrt(np.nanmean(arr * arr, axis=0))
    C = arr.shape[1]
    for ci in range(C):
        feats['%s/ch%03d/mean' % (prefix, ci)] = float(mean[ci])
        feats['%s/ch%03d/std' % (prefix, ci)] = float(std[ci])
        feats['%s/ch%03d/range' % (prefix, ci)] = float(rng[ci])
        feats['%s/ch%03d/rms' % (prefix, ci)] = float(rms[ci])
    return feats


def _joint_mag_stats(arr, prefix, n_axis=3):
    feats = {}
    if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
        return feats
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return feats
    T = arr.shape[0]
    njoint = arr.shape[1] // n_axis
    if njoint == 0:
        return feats
    a = arr.reshape(T, njoint, n_axis)
    mag = np.sqrt(np.nansum(a * a, axis=2))
    mean = np.nanmean(mag, axis=0)
    std = np.nanstd(mag, axis=0, ddof=1)
    rng = np.nanmax(mag, axis=0) - np.nanmin(mag, axis=0)
    rms = np.sqrt(np.nanmean(mag * mag, axis=0))
    for j in range(njoint):
        jn = JOINTS[j] if j < len(JOINTS) else 'joint%d' % j
        feats['%s/mag/%s/mean' % (prefix, jn)] = float(mean[j])
        feats['%s/mag/%s/std' % (prefix, jn)] = float(std[j])
        feats['%s/mag/%s/range' % (prefix, jn)] = float(rng[j])
        feats['%s/mag/%s/rms' % (prefix, jn)] = float(rms[j])
    return feats


def _pressure_features(arr, side):
    feats = {}
    if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
        return feats
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 3 or arr.shape[0] < 2:
        return feats
    T = arr.shape[0]
    flat = arr.reshape(T, -1)
    cmean = np.nanmean(flat, axis=0)
    cstd = np.nanstd(flat, axis=0, ddof=1)
    cmax = np.nanmax(flat, axis=0)
    crms = np.sqrt(np.nanmean(flat * flat, axis=0))
    for i in range(flat.shape[1]):
        feats['pressure_%s/cell%03d/mean' % (side, i)] = float(cmean[i])
        feats['pressure_%s/cell%03d/std' % (side, i)] = float(cstd[i])
        feats['pressure_%s/cell%03d/max' % (side, i)] = float(cmax[i])
        feats['pressure_%s/cell%03d/rms' % (side, i)] = float(crms[i])
    total = np.nansum(flat, axis=1)
    feats['pressure_%s/total/mean' % side] = float(np.nanmean(total))
    feats['pressure_%s/total/std' % side] = float(np.nanstd(total, ddof=1))
    feats['pressure_%s/total/max' % side] = float(np.nanmax(total))
    feats['pressure_%s/total/range' % side] = float(np.nanmax(total) - np.nanmin(total))
    feats['pressure_%s/total/rms' % side] = float(np.sqrt(np.nanmean(total * total)))
    rows = np.arange(arr.shape[1])
    cols = np.arange(arr.shape[2])
    rr = np.repeat(rows, arr.shape[2])
    cc = np.tile(cols, arr.shape[1])
    wsum = np.nansum(flat, axis=1)
    safe = np.where(wsum == 0, 1.0, wsum)
    cop_r = np.nansum(flat * rr[None, :], axis=1) / safe
    cop_c = np.nansum(flat * cc[None, :], axis=1) / safe
    feats['pressure_%s/cop_row/mean' % side] = float(np.nanmean(cop_r))
    feats['pressure_%s/cop_row/std' % side] = float(np.nanstd(cop_r, ddof=1))
    feats['pressure_%s/cop_row/range' % side] = float(np.nanmax(cop_r) - np.nanmin(cop_r))
    feats['pressure_%s/cop_col/mean' % side] = float(np.nanmean(cop_c))
    feats['pressure_%s/cop_col/std' % side] = float(np.nanstd(cop_c, ddof=1))
    feats['pressure_%s/cop_col/range' % side] = float(np.nanmax(cop_c) - np.nanmin(cop_c))
    dr = np.diff(cop_r); dc = np.diff(cop_c)
    feats['pressure_%s/cop_pathlen' % side] = float(np.nansum(np.sqrt(dr*dr + dc*dc)))
    return feats


def _cop_features(arr, side):
    feats = {}
    if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
        return feats
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return feats
    for k in range(2):
        ax = 'x' if k == 0 else 'y'
        v = arr[:, k]
        feats['cop_%s/%s/mean' % (side, ax)] = float(np.nanmean(v))
        feats['cop_%s/%s/std' % (side, ax)] = float(np.nanstd(v, ddof=1))
        feats['cop_%s/%s/min' % (side, ax)] = float(np.nanmin(v))
        feats['cop_%s/%s/max' % (side, ax)] = float(np.nanmax(v))
        feats['cop_%s/%s/range' % (side, ax)] = float(np.nanmax(v) - np.nanmin(v))
    dr = np.diff(arr[:, 0]); dc = np.diff(arr[:, 1])
    feats['cop_%s/pathlen' % side] = float(np.nansum(np.sqrt(dr*dr + dc*dc)))
    return feats


def _gaze_features(arr):
    feats = {}
    if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
        return feats
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return feats
    for k, ax in enumerate(['x','y']):
        v = arr[:, k]
        feats['gaze/%s/mean' % ax] = float(np.nanmean(v))
        feats['gaze/%s/std' % ax] = float(np.nanstd(v, ddof=1))
        feats['gaze/%s/min' % ax] = float(np.nanmin(v))
        feats['gaze/%s/max' % ax] = float(np.nanmax(v))
        feats['gaze/%s/range' % ax] = float(np.nanmax(v) - np.nanmin(v))
    mag = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2)
    feats['gaze/mag/mean'] = float(np.nanmean(mag))
    feats['gaze/mag/std'] = float(np.nanstd(mag, ddof=1))
    feats['gaze/mag/max'] = float(np.nanmax(mag))
    feats['gaze/mag/range'] = float(np.nanmax(mag) - np.nanmin(mag))
    return feats


def _imu_features(arr, name):
    feats = {}
    if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
        return feats
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
        return feats
    for k, ax in enumerate(['x','y','z']):
        v = arr[:, k]
        feats['%s/%s/mean' % (name, ax)] = float(np.nanmean(v))
        feats['%s/%s/std' % (name, ax)] = float(np.nanstd(v, ddof=1))
        feats['%s/%s/min' % (name, ax)] = float(np.nanmin(v))
        feats['%s/%s/max' % (name, ax)] = float(np.nanmax(v))
        feats['%s/%s/range' % (name, ax)] = float(np.nanmax(v) - np.nanmin(v))
        feats['%s/%s/rms' % (name, ax)] = float(np.sqrt(np.nanmean(v*v)))
    mag = np.sqrt(np.nansum(arr*arr, axis=1))
    feats['%s/mag/mean' % name] = float(np.nanmean(mag))
    feats['%s/mag/std' % name] = float(np.nanstd(mag, ddof=1))
    feats['%s/mag/max' % name] = float(np.nanmax(mag))
    feats['%s/mag/range' % name] = float(np.nanmax(mag) - np.nanmin(mag))
    feats['%s/mag/rms' % name] = float(np.sqrt(np.nanmean(mag*mag)))
    return feats


def extract_swing(strs):
    feats = {}
    for stream_name, n_axis in PNS_STREAMS.items():
        arr = strs.get(stream_name)
        if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
            continue
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[0] < 2:
            continue
        feats.update(_channel_stats(arr, stream_name))
        feats.update(_joint_mag_stats(arr, stream_name, n_axis))
    for side in ['left','right']:
        key = 'wireless-insole-%s/pressure-data' % side
        arr = strs.get(key)
        if arr is not None and isinstance(arr, np.ndarray) and arr.size > 0:
            feats.update(_pressure_features(arr, side))
        ckey = 'wireless-insole-%s/cop' % side
        carr = strs.get(ckey)
        if carr is not None and isinstance(carr, np.ndarray) and carr.size > 0:
            feats.update(_cop_features(carr, side))
    g = strs.get('pupil-gaze-xy/gaze-values')
    if g is not None and isinstance(g, np.ndarray) and g.size > 0:
        feats.update(_gaze_features(g))
    a = strs.get('pupil-imu-accel/acceleration-values')
    if a is not None and isinstance(a, np.ndarray) and a.size > 0:
        feats.update(_imu_features(a, 'head_accel'))
    gv = strs.get('pupil-imu-gyro/velocity-values')
    if gv is not None and isinstance(gv, np.ndarray) and gv.size > 0:
        feats.update(_imu_features(gv, 'head_gyro'))
    return feats


def main():
    t_all0 = time.perf_counter()
    t_read0 = time.perf_counter()
    manifest = cg.manifest()
    zips = sorted(set(r['path'].split('/')[0] for r in manifest))
    data_tags = []
    time_tags = []
    for z in zips:
        pat = 'golf/' + z.replace('.', r'\.') + '/.*'
        ts = cg.tags(pat)
        data_tags.extend(t for t in ts if t.endswith('/data'))
        time_tags.extend(t for t in ts if t.endswith('/time_s'))
    arrays = {}
    for t in data_tags:
        arrays[t] = cg.read_array(t)
    time_arrays = {}
    for t in time_tags:
        time_arrays[t] = cg.read_array(t)
    # documentation CSVs (participant metadata + per-swing annotations)
    csv_tags = {r['path']: r['tag'] for r in manifest if r['path'].endswith('.csv')}
    csvs = {name: cg.read_csv(tag) for name, tag in csv_tags.items()}
    t_read1 = time.perf_counter()
    read_time_s = t_read1 - t_read0

    records = []
    for t, arr in arrays.items():
        parts = t.split('/')
        zipname = parts[1]
        hdf5 = parts[3]
        m = re.match(r'(P\d+)_Swing(\d+)_stream_data\.hdf5', hdf5)
        if not m:
            continue
        pid = m.group(1)
        swn = int(m.group(2))
        stream = '/'.join(parts[4:-1])
        records.append((pid, swn, stream, arr))

    swings = {}
    for pid, swn, stream, arr in records:
        swings.setdefault((pid, swn), {})[stream] = arr

    # durations from time_s (already read in timed window)
    duration = {}
    for t, arr in time_arrays.items():
        parts = t.split('/')
        hdf5 = parts[3]
        m = re.match(r'(P\d+)_Swing(\d+)_stream_data\.hdf5', hdf5)
        if not m:
            continue
        pid = m.group(1); swn = int(m.group(2))
        arr = np.asarray(arr).ravel()
        if arr.size >= 2:
            d = float(arr[-1] - arr[0])
            cur = duration.get((pid, swn))
            if cur is None or d > cur:
                duration[(pid, swn)] = d

    rows = []
    for (pid, swn), strs in swings.items():
        f = extract_swing(strs)
        f['pid'] = pid
        f['swing'] = swn
        f['group'] = group_of(pid)
        f['swing_duration_s'] = duration.get((pid, swn), np.nan)
        for sn in ['pns-joint-euler/angle-values']:
            if sn in strs:
                f['n_frames_pns'] = int(strs[sn].shape[0])
        rows.append(f)

    feat_df = pd.DataFrame(rows)
    out_parquet = os.path.join(HERE, 'golf_features.parquet')
    feat_df.to_csv(os.path.join(HERE, 'golf_features.csv'), index=False)

    meta = {'pid','swing','group'}
    feature_cols = [c for c in feat_df.columns if c not in meta]
    for c in feature_cols:
        feat_df[c] = pd.to_numeric(feat_df[c], errors='coerce')

    part = feat_df.groupby('pid')[feature_cols].mean(numeric_only=True)
    part['group'] = feat_df.groupby('pid')['group'].first()
    part = part.reset_index()

    beg = part[part['group'] == 'Beginner']
    pro = part[part['group'] == 'Professional']

    results = []
    for c in feature_cols:
        x = beg[c].dropna().to_numpy(dtype=float)
        y = pro[c].dropna().to_numpy(dtype=float)
        if x.size < 3 or y.size < 3:
            continue
        nx, ny = x.size, y.size
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        pooled = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
        if pooled == 0 or not np.isfinite(pooled):
            continue
        d = (x.mean() - y.mean()) / pooled
        try:
            _, pval = stats.mannwhitneyu(x, y, alternative='two-sided')
        except Exception:
            pval = np.nan
        try:
            _, tpval = stats.ttest_ind(x, y, equal_var=False)
        except Exception:
            tpval = np.nan
        results.append({
            'feature': c,
            'cohens_d': float(d),
            'abs_d': float(abs(d)),
            'mean_beginner': float(x.mean()),
            'mean_professional': float(y.mean()),
            'mannwhitney_p': float(pval),
            'welch_t_p': float(tpval),
            'n_beginner': int(nx),
            'n_professional': int(ny),
        })
    res_df = pd.DataFrame(results).sort_values('abs_d', ascending=False).reset_index(drop=True)
    res_df['rank'] = np.arange(1, len(res_df)+1)
    # Benjamini-Hochberg FDR on Mann-Whitney p-values
    pvals = res_df['mannwhitney_p'].to_numpy(dtype=float)
    qvals = np.full(len(pvals), np.nan)
    finite = np.isfinite(pvals)
    if finite.any():
        idx = np.where(finite)[0]
        order = idx[np.argsort(pvals[idx])]
        m = len(order)
        prev = 1.0
        for i in range(m-1, -1, -1):
            j = order[i]
            prev = min(prev, pvals[j] * m / (i + 1))
            qvals[j] = prev
    res_df['mannwhitney_q'] = qvals
    res_df['label'] = res_df['feature'].map(label)
    res_path = os.path.join(HERE, 'golf_separation_ranking.csv')
    res_df.to_csv(res_path, index=False)

    def label(feat):
        for sn, na in PNS_STREAMS.items():
            if feat.startswith(sn + '/'):
                mch = re.search(r'/ch(\d+)/', feat)
                if mch:
                    ci = int(mch.group(1))
                    joint = JOINTS[ci // na] if (ci // na) < len(JOINTS) else 'joint%d' % (ci // na)
                    axis = (AXES3 if na == 3 else AXES4)[ci % na]
                    stat = feat.split('/')[-1]
                    stream_short = sn.split('/')[0].replace('pns-joint-','')
                    return '%s: %s %s (%s)' % (stream_short, joint, axis, stat)
                if '/mag/' in feat:
                    return feat
        return feat
    res_df['label'] = res_df['feature'].map(label)

    print('\n=== TOP 40 SENSOR FEATURES (Beginner vs Professional) ===')
    print(res_df[['rank','label','cohens_d','abs_d','mannwhitney_p']].head(40).to_string(index=False))
    print('\nTotal features:', len(res_df))
    print('Read time (s):', round(read_time_s, 3))
    print('Total runtime (s):', round(time.perf_counter()-t_all0, 2))

    res_df.head(60).to_csv(os.path.join(HERE, 'golf_top60.csv'), index=False)
    with open(os.path.join(HERE, 'read_time.txt'), 'w') as fh:
        fh.write('%.3f\n' % read_time_s)


if __name__ == '__main__':
    main()
