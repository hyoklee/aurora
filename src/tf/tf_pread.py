"""Parallel reads of the pipeline's deflate-compressed datasets.

HDF5 inflates the chunks of a read one at a time, and deflate is 95% of the
pipeline's read time (job 8949105). Almost every dataset the pipeline reads is
ONE chunk (ASTER TIR and geolocation, MODIS geolocation, one band of
EV_1KM_Emissive), so the parallelism has to come from reading many datasets at
once: read_many() fetches the raw chunks of a whole batch with h5py's
read_direct_chunk, inflates them all in one thread pool (zlib releases the
GIL) and places each into its output array.

    TF_PREAD=0       netCDF4 reads, as before (default)
    TF_PREAD=1       parallel reads where possible
    TF_PREAD=check   both; raise if any value differs (NaN == NaN)
    TF_PREAD_THREADS pool size (default 32; 104 gained little more in 8949105)

Anything this cannot reproduce exactly is read with netCDF4 instead: data
netCDF4 would transform (scale_factor, add_offset, _Unsigned), filters other
than deflate, and contiguous datasets. The result is the stored values as an
ndarray; callers take np.asarray(..., 'f8') of it, which is also what they did
with netCDF4's masked arrays, whose data is unchanged.

    PR = tf_pread.Reader(F)
    la, lo = PR.read_many([(G['Latitude'], ()), (G['Longitude'], ())])
    v = PR.read(EV, (BAND_IDX,))                    # == EV[BAND_IDX, :, :]
"""
import os, time, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np

MODE = os.environ.get('TF_PREAD', '0')
NTHREADS = int(os.environ.get('TF_PREAD_THREADS', '32'))
TRANSFORMS = ('scale_factor', 'add_offset', '_Unsigned')


def _nc(var, idx):
    return var[idx] if idx else var[:]


class Reader:
    def __init__(self, path):
        self.f = None
        self.stats = dict(mode=MODE, threads=NTHREADS, parallel=0, fallback=0, chunks=0, read_s=0.0)
        if MODE in ('1', 'check'):
            import h5py
            self.f = h5py.File(path, 'r')
            self.pool = ThreadPoolExecutor(NTHREADS)

    def read(self, var, idx=()):
        """var[idx] for leading integer indices idx (the other axes whole)."""
        return self.read_many([(var, idx)])[0]

    def read_many(self, reqs):
        """[var[idx] for (var, idx) in reqs], all chunks inflated in one pool."""
        t = time.time()
        plans = [self._plan(v, i) if self.f is not None else None for v, i in reqs]
        outs = [None] * len(reqs)
        jobs = []
        for r, p in enumerate(plans):
            if p is None:
                outs[r] = _nc(*reqs[r]); self.stats['fallback'] += 1
                continue
            ds, lo, hi, offs = p
            outs[r] = np.full([h - l for l, h in zip(lo, hi)], ds.fillvalue, dtype=ds.dtype)
            jobs += [(r, o, ds.id.read_direct_chunk(o)) for o in offs]   # h5py serialises HDF5 calls

        def place(job):
            r, o, (mask, buf) = job
            ds, lo, hi, _ = plans[r]
            ck = ds.chunks
            raw = buf if ds.compression is None or mask & 1 else zlib.decompress(buf)
            c = np.frombuffer(raw, ds.dtype).reshape(ck)
            src = tuple(slice(max(lo[a] - o[a], 0), min(o[a] + ck[a], hi[a]) - o[a]) for a in range(ds.ndim))
            dst = tuple(slice(o[a] + s.start - lo[a], o[a] + s.stop - lo[a]) for a, s in enumerate(src))
            outs[r][dst] = c[src]

        if jobs:
            list(self.pool.map(place, jobs))
        self.stats['chunks'] += len(jobs)
        for r, p in enumerate(plans):
            if p is None:
                continue
            k = len(reqs[r][1])
            outs[r] = (outs[r][(0,) * k] if k else outs[r]).astype(p[0].dtype.newbyteorder('='), copy=False)
            self.stats['parallel'] += 1
            if MODE == 'check':
                ref = np.ma.getdata(_nc(*reqs[r]))
                if ref.shape != outs[r].shape or not np.array_equal(
                        ref, outs[r], equal_nan=outs[r].dtype.kind == 'f'):
                    v = reqs[r][0]
                    raise SystemExit(f'tf_pread: {v.group().path}/{v.name}{list(reqs[r][1])} differs from netCDF4')
        self.stats['read_s'] = round(self.stats['read_s'] + time.time() - t, 3)
        return outs

    def _plan(self, var, idx):
        """(h5py dataset, lo, hi, chunk offsets touched), or None for a netCDF4 read."""
        if any(a in var.ncattrs() for a in TRANSFORMS):
            return None
        ds = self.f[var.group().path.rstrip('/') + '/' + var.name]
        if (not ds.chunks or ds.compression not in ('gzip', None) or ds.shuffle
                or ds.fletcher32 or ds.scaleoffset is not None):
            return None
        ck, k, nd = ds.chunks, len(idx), ds.ndim
        lo = list(idx) + [0] * (nd - k)
        hi = [i + 1 for i in idx] + list(ds.shape[k:])
        offs = []
        for n in range(ds.id.get_num_chunks()):
            o = ds.id.get_chunk_info(n).chunk_offset
            if all(o[a] < hi[a] and o[a] + ck[a] > lo[a] for a in range(nd)):
                offs.append(o)
        return ds, lo, hi, offs
