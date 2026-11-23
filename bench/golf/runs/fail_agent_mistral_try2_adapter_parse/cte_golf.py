"""Read MultiSenseGolf back out of clio-core (CTE) after `clio-cae golf_omni.yaml`.

The CAE Dataverse source stores, under the tag prefix "golf":
  golf                                   blob "manifest": one line per object
  golf/<zip>/<SwingNN>/<file>.hdf5/<stream>/<data|time_s>
                                         blobs "description" ("tensor<float32, 245, 63>")
                                         and chunk_0, chunk_1, ... (C-order bytes)
  golf/Documentation.zip/<name>.csv      blobs "description", chunk_0, ... (raw file)

Environment: the clio-core build's bin/ on PYTHONPATH and LD_LIBRARY_PATH, and
CLIO_SERVER_CONF pointing at the runtime's config (see clio_env.sh).
"""
import io
import os
import re

import numpy as np

import clio_cte_core_ext as cte

_DTYPES = {"float32": np.float32, "float64": np.float64, "int8": np.int8,
           "int16": np.int16, "int32": np.int32, "int64": np.int64,
           "uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32,
           "uint64": np.uint64}
_client = None


def connect():
    """Attach to the running clio-core runtime (once per process)."""
    global _client
    if _client is None:
        if not cte.clio_init(cte.RuntimeMode.kClient, False):
            raise RuntimeError("clio_init failed: is the runtime up?")
        if not cte.initialize_cte(os.environ["CLIO_SERVER_CONF"], cte.PoolQuery.Local()):
            raise RuntimeError("initialize_cte failed")
        _client = cte.get_cte_client()
    return _client


def tags(regex):
    """Tag names matching a full-string regex, e.g. r'golf/P01\\.zip/.*'."""
    return connect().TagQuery(regex, 0, cte.PoolQuery.Local())


def _blob(tag, name):
    return tag.GetBlob(name, tag.GetBlobSize(name))


def _chunks(tag):
    names = [b for b in tag.GetContainedBlobs() if b.startswith("chunk_")]
    names.sort(key=lambda b: int(b[6:]))
    return b"".join(_blob(tag, b) for b in names)


def manifest(prefix="golf"):
    """Rows of the manifest: dicts with path, size, file_id, member, tag."""
    connect()
    text = _blob(cte.Tag(prefix), "manifest").decode()
    rows = []
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        path, size, file_id, member, tag = line.split("\t")
        rows.append(dict(path=path, size=int(size), file_id=int(file_id),
                         member=member, tag=tag))
    return rows


def read_array(tag_name):
    """A dataset ingested by the HDF5 assimilator, as a numpy array."""
    connect()
    t = cte.Tag(tag_name)
    desc = _blob(t, "description").decode()
    m = re.fullmatch(r"tensor<(\w+)((?:, *\d+)*)>", desc.strip())
    if not m:
        raise ValueError(f"{tag_name}: unexpected description {desc!r}")
    dtype = _DTYPES[m.group(1)]
    shape = tuple(int(x) for x in re.findall(r"\d+", m.group(2)))
    if 0 in shape:
        return np.zeros(shape, dtype)
    return np.frombuffer(_chunks(t), dtype=dtype).reshape(shape)


def read_bytes(tag_name):
    """A file ingested as binary (e.g. a CSV), as bytes."""
    connect()
    return _chunks(cte.Tag(tag_name))


def read_csv(tag_name, **kw):
    import pandas as pd
    return pd.read_csv(io.BytesIO(read_bytes(tag_name)), encoding="utf-8-sig", **kw)
