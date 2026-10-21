#!/usr/bin/env python3
"""How much of the pipeline's read time is deflate, and how much of that parallel
inflate on the CPU recovers (the first step before GPU decompression, option D).

    python tf_inflate_bench.py <granule.h5>              -> one JSON line
    python tf_inflate_bench.py --oversized <granule.h5>  -> comma-separated paths of
                     the datasets below whose chunk exceeds THRESH_MB (for
                     tf_rechunk.py's RECHUNK_INCLUDE)

    THREADS    thread counts for the pooled inflate (default 8,32,104)
    THRESH_MB  --oversized threshold, uncompressed (32, as tf_rechunk.py)

The datasets and selections are the ones tf_aster_blocks / tf_sources /
tf_contrast / tf_congruence read (READS below). MODIS and CERES groups outside
the strip are included, which the pipeline skips, so compare ratios within this
benchmark, not its absolute seconds with the pipeline's.

Per sensor, over the chunks each selection touches:
  h5       h5py read of the selection (HDF5: read + inflate, one thread)
  raw      read_direct_chunk of the same chunks (I/O, no inflate)
  inflate  zlib.decompress of those raw chunks, one thread
  pool_<n> the same chunks in one pool of n threads (zlib releases the GIL)
  max_chunk  the slowest single chunk: no amount of chunk parallelism beats it
The first chunk of every dataset that is not on an edge is checked against the
h5py read, and every inflated chunk must have the chunk's full size.
"""
import json, math, os, re, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
import h5py, numpy as np

THREADS = [int(t) for t in os.environ.get('THREADS', '8,32,104').split(',')]
THRESH = float(os.environ.get('THRESH_MB', 32)) * 2**20
BAND_IDX = int(os.environ.get('MODIS_BAND_IDX', 10))
CERES_FIELD = os.environ.get('CERES_FIELD', 'WN_Radiance')

# (sensor, path regex, selection or None for the whole dataset)
READS = [
    ('ASTER', r'ASTER/[^/]+/Geolocation/(Latitude|Longitude)', None),
    ('ASTER', r'ASTER/[^/]+/TIR/(Geolocation/(Latitude|Longitude)|ImageData10)', None),
    ('MODIS', r'MODIS/[^/]+/_1KM/Geolocation/(Latitude|Longitude)', None),
    ('MODIS', r'MODIS/[^/]+/_1KM/Data_Fields/EV_1KM_Emissive', (BAND_IDX,)),
    ('MISR', r'MISR/Geolocation/(GeoLatitude|GeoLongitude)', None),
    ('MISR', r'MISR/AN/Data_Fields/Red_Radiance', None),
    ('CERES', r'CERES/[^/]+/FM[12]/Time_and_Position/(Latitude|Longitude)', None),
    ('CERES', rf'CERES/[^/]+/FM[12]/Radiances/{CERES_FIELD}', None),
    ('MOPITT', r'MOPITT/[^/]+/(Geolocation/(Latitude|Longitude)|Data_Fields/MOPITTRadiances)', None),
]


def selected(f):
    out = []
    def visit(name, o):
        if isinstance(o, h5py.Dataset):
            for sensor, rx, sel in READS:
                if re.fullmatch(rx, name):
                    out.append((sensor, name, o, sel)); break
    f.visititems(visit)
    # tf_sources reads only the first MOPITT granule group
    mop = sorted({n.split('/')[1] for s, n, _, _ in out if s == 'MOPITT'})[:1]
    return [r for r in out if r[0] != 'MOPITT' or r[1].split('/')[1] in mop]


def chunks_touched(ds, sel):
    """(offset, filter_mask, raw bytes) of every allocated chunk the selection hits."""
    sid = ds.id
    want = None
    if sel is not None:   # leading integer indices, as EV[BAND_IDX, :, :]
        want = [(i, sel[i]) for i in range(len(sel))]
    out = []
    for k in range(sid.get_num_chunks()):
        info = sid.get_chunk_info(k)
        if want and any(not (info.chunk_offset[a] <= v < info.chunk_offset[a] + ds.chunks[a]) for a, v in want):
            continue
        out.append(info.chunk_offset)
    return out


def main():
    if sys.argv[1] == '--oversized':
        f = h5py.File(sys.argv[2], 'r')
        print(','.join(n for _, n, o, _ in selected(f)
                       if o.chunks and math.prod(o.chunks) * o.dtype.itemsize > THRESH))
        return
    path = sys.argv[1]
    f = h5py.File(path, 'r')
    S, bad = {}, []
    t_all = time.time()
    for sensor, name, ds, sel in selected(f):
        s = S.setdefault(sensor, dict(datasets=0, contiguous=0, chunks=0, raw_MB=0.0, out_MB=0.0,
                                      h5_s=0.0, raw_s=0.0, inflate_s=0.0, max_chunk_s=0.0, _raw=[]))
        s['datasets'] += 1
        t = time.time(); ref = ds[sel] if sel is not None else ds[()]; s['h5_s'] += time.time() - t
        if not ds.chunks:
            s['contiguous'] += 1; continue
        if ds.compression not in ('gzip', None) or ds.shuffle or ds.fletcher32 or ds.scaleoffset:
            bad.append(f'{name}: filters {ds._filters}'); continue
        offs = chunks_touched(ds, sel)
        t = time.time()
        raws = [ds.id.read_direct_chunk(o) for o in offs]
        s['raw_s'] += time.time() - t
        nbytes = math.prod(ds.chunks) * ds.dtype.itemsize
        for o, (mask, buf) in zip(offs, raws):
            t = time.time()
            out = buf if (mask & 1 or ds.compression is None) else zlib.decompress(buf)
            dt = time.time() - t
            s['inflate_s'] += dt; s['max_chunk_s'] = max(s['max_chunk_s'], dt)
            if len(out) != nbytes:
                bad.append(f'{name} chunk {o}: {len(out)} bytes, want {nbytes}')
            s['_raw'].append((buf, mask & 1 or ds.compression is None))
        s['chunks'] += len(offs)
        s['raw_MB'] += sum(len(b) for _, b in raws) / 1e6
        s['out_MB'] += len(offs) * nbytes / 1e6
        # spot check: first interior chunk == the h5py read of that region
        for o, (mask, buf) in zip(offs, raws):
            if all(o[a] + ds.chunks[a] <= ds.shape[a] for a in range(ds.ndim)):
                region = tuple(slice(o[a], o[a] + ds.chunks[a]) for a in range(ds.ndim))
                got = np.frombuffer(buf if (mask & 1 or ds.compression is None) else zlib.decompress(buf),
                                    ds.dtype).reshape(ds.chunks)
                if not np.array_equal(got, ds[region], equal_nan=ds.dtype.kind == 'f'):
                    bad.append(f'{name} chunk {o}: differs from h5py')
                break
        del ref
    for s in S.values():
        raw = s.pop('_raw')
        for n in THREADS:
            t = time.time()
            with ThreadPoolExecutor(n) as ex:
                list(ex.map(lambda r: r[0] if r[1] else zlib.decompress(r[0]), raw))
            s[f'pool_{n}_s'] = time.time() - t
        for k, v in s.items():
            if isinstance(v, float): s[k] = round(v, 3)
    tot = {k: round(sum(s[k] for s in S.values()), 3) for k in next(iter(S.values())) if k != 'max_chunk_s'}
    tot['max_chunk_s'] = max(s['max_chunk_s'] for s in S.values())
    print(json.dumps(dict(file=os.path.basename(path), cpus=os.cpu_count(), sensors=S, total=tot,
                          bench_s=round(time.time() - t_all, 1), problems=bad)))


if __name__ == '__main__':
    main()
