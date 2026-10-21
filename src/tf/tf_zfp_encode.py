#!/usr/bin/env python3
"""Put ZFP-coded values into a copy of a granule, to see what lossy ZFP does to
congruence (before anything is built to decode ZFP on the PVC).

    python tf_zfp_encode.py <src.h5> <dst.h5> <mode>     -> one JSON line
    mode: lossless (ZFP reversible) | rate=<bits per value> (fixed rate, the
          mode GPU decoders implement)

dst.h5 is an existing copy of src.h5. Every chunked dataset the pipeline reads
(tf_inflate_bench.READS: ASTER TIR and geolocation, MODIS geolocation and the
EV_1KM_Emissive band, MISR geolocation and Red_Radiance) is ZFP-coded one HDF5
chunk at a time, as an H5Z-ZFP filter would store it, decoded again, and the
decoded values written into dst.h5 with the dataset's own deflate. So the
pipeline runs unchanged and reads exactly what a ZFP reader would hand it. The
contiguous ones (ASTER 11x11 geolocation, CERES, MOPITT; under 1 MB) stay as
they are.

ZFP has no notion of fill values and NaN, and one fill value in a 4x4 block
ruins the block's other values. So, as any real deployment would, values that
are NaN/Inf, equal _FillValue or outside valid_min..valid_max (latitude
outside +-90, longitude outside +-360) are masked: replaced by the chunk's
mean valid value before coding and restored exactly afterwards. The mask
(packbits + zlib) is counted in the ZFP size.

Reports per field (sensor + dataset name): values, raw bytes, the deflate
bytes stored now, ZFP bytes (+ mask), max and RMS error over valid values,
the valid values' standard deviation, and ZFP encode/decode seconds on one CPU
thread next to zlib inflate of the same chunks.
"""
import json, os, sys, time, zlib
import h5py, numpy as np, zfpy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tf_inflate_bench import selected


def valid_mask(name, ds, a):
    ok = np.isfinite(a)
    fv = ds.attrs.get('_FillValue')
    if fv is not None:
        ok &= a != np.asarray(fv).ravel()[0]
    lo, hi = ds.attrs.get('valid_min'), ds.attrs.get('valid_max')
    leaf = name.rsplit('/', 1)[-1].lower()
    if lo is None and 'latitude' in leaf: lo, hi = -90, 90
    if lo is None and 'longitude' in leaf: lo, hi = -360, 360
    with np.errstate(invalid='ignore'):
        if lo is not None: ok &= a >= np.asarray(lo).ravel()[0]
        if hi is not None: ok &= a <= np.asarray(hi).ravel()[0]
    return ok


def code(a, mode):
    """(decoded array, zfp bytes, encode s, decode s). Length-1 axes are dropped
    first: ZFP codes 4^d blocks, and in fixed-rate mode a padded block costs as
    much as a full one, so MISR's (1,512,2048) chunk coded as 3-D would cost 4x."""
    kw = {} if mode == 'lossless' else dict(rate=float(mode.split('=')[1]))
    t = time.time(); b = zfpy.compress_numpy(np.ascontiguousarray(a.squeeze()), **kw); te = time.time() - t
    t = time.time(); d = zfpy.decompress_numpy(b); td = time.time() - t
    return d.reshape(a.shape), len(b), te, td


def main():
    src_path, dst_path, mode = sys.argv[1:4]
    src, dst = h5py.File(src_path, 'r'), h5py.File(dst_path, 'r+')
    F, t0 = {}, time.time()
    for sensor, name, ds, sel in selected(src):
        if not ds.chunks or ds.compression != 'gzip':
            continue
        field = f"{sensor} {name.rsplit('/', 1)[-1]}"
        r = F.setdefault(field, dict(datasets=0, chunks=0, values=0, masked=0, raw_MB=0.0, deflate_MB=0.0,
                                     zfp_MB=0.0, mask_MB=0.0, max_err=0.0, _sq=0.0, _n=0, _s=0.0, _s2=0.0,
                                     inflate_s=0.0, zfp_encode_s=0.0, zfp_decode_s=0.0))
        r['datasets'] += 1
        out = dst[name]
        for k in range(ds.id.get_num_chunks()):
            info = ds.id.get_chunk_info(k)
            o = info.chunk_offset
            if sel is not None and any(not (o[a] <= v < o[a] + ds.chunks[a]) for a, v in enumerate(sel)):
                continue
            region = tuple(slice(o[a], min(o[a] + ds.chunks[a], ds.shape[a])) for a in range(ds.ndim))
            mask_, buf = ds.id.read_direct_chunk(o)
            t = time.time(); zlib.decompress(buf); r['inflate_s'] += time.time() - t
            a = ds[region]
            ok = valid_mask(name, ds, a)
            fill = a[ok].mean(dtype='f8') if ok.any() else 0.0
            d, nb, te, td = code(np.where(ok, a, fill).astype(a.dtype), mode)
            d = np.where(ok, d, a).astype(a.dtype)
            out[region] = d
            e = np.abs(d[ok].astype('f8') - a[ok])
            v = a[ok].astype('f8')
            r['chunks'] += 1; r['values'] += a.size; r['masked'] += int(a.size - ok.sum())
            r['raw_MB'] += a.nbytes / 1e6; r['deflate_MB'] += info.size / 1e6; r['zfp_MB'] += nb / 1e6
            if not ok.all():
                r['mask_MB'] += len(zlib.compress(np.packbits(~ok).tobytes())) / 1e6
            if e.size:
                r['max_err'] = max(r['max_err'], float(e.max())); r['_sq'] += float((e ** 2).sum())
                r['_n'] += e.size; r['_s'] += float(v.sum()); r['_s2'] += float((v ** 2).sum())
            r['zfp_encode_s'] += te; r['zfp_decode_s'] += td
    dst.close()
    for r in F.values():
        n = r.pop('_n'); sq, s, s2 = r.pop('_sq'), r.pop('_s'), r.pop('_s2')
        r['rms_err'] = (sq / n) ** 0.5 if n else 0.0
        r['std'] = max(s2 / n - (s / n) ** 2, 0) ** 0.5 if n else 0.0
        r['bits_per_value'] = (r['zfp_MB'] + r['mask_MB']) * 8e6 / r['values']
        for k, v in r.items():
            if isinstance(v, float): r[k] = float(f'{v:.4g}')
    tot = {k: round(sum(r[k] for r in F.values()), 3) for k in
           ('values', 'raw_MB', 'deflate_MB', 'zfp_MB', 'mask_MB', 'inflate_s', 'zfp_encode_s', 'zfp_decode_s')}
    print(json.dumps(dict(mode=mode, fields=F, total=tot, encode_wall_s=round(time.time() - t0, 1))))


if __name__ == '__main__':
    main()
