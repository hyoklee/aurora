"""Read the IceCube 7-yr posterior release back out of clio-core (CTE) after
`clio-cae ice_omni.yaml`.

Tags under the prefix "ice" (Z = the release zip's file name):
  ice                         blob "manifest": one line per ingested object
  ice/Z/posterior.h5          blob "attributes": JSON of the HDF5 root attributes
  ice/Z/posterior.h5/<table>  one PyTables table per spatial template:
                              blob "description"  compound<40, N>{ln_A:float64@0, ...}
                              blob "attributes"   JSON of the table's attributes
                              chunk_0, chunk_1, ... the packed records
  ice/Z/example.py            the release's code   (chunk_0.. = the raw file)
  ice/Z/readme.txt            the release's readme (chunk_0.. = the raw file)

Environment: the clio-core build's bin/ on PYTHONPATH and LD_LIBRARY_PATH and
CLIO_SERVER_CONF pointing at the runtime's config (source ./clio_env.sh).

  import cte_ice as ci
  ci.manifest()                 # rows: path, size, file_id, member, tag
  ci.read_table(tag)            # numpy structured array (fields by name)
  ci.read_attributes(tag)       # dict
  ci.read_text(tag)             # str of a binary-ingested file
"""
import json
import os
import re

import numpy as np

import clio_cte_core_ext as cte

_DTYPES = {"float32": "<f4", "float64": "<f8", "int8": "<i1", "int16": "<i2",
           "int32": "<i4", "int64": "<i8", "uint8": "<u1", "uint16": "<u2",
           "uint32": "<u4", "uint64": "<u8"}
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
    """Tag names matching a full-string regex, e.g. r'ice/.*'."""
    return connect().TagQuery(regex, 0, cte.PoolQuery.Local())


def _blob(tag, name):
    return tag.GetBlob(name, tag.GetBlobSize(name))


def _chunks(tag):
    names = [b for b in tag.GetContainedBlobs() if b.startswith("chunk_")]
    names.sort(key=lambda b: int(b[6:]))
    return b"".join(_blob(tag, b) for b in names)


def manifest(prefix="ice"):
    """Rows of the manifest: dicts with path, size, file_id, member, tag."""
    connect()
    rows = []
    for line in _blob(cte.Tag(prefix), "manifest").decode().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        path, size, file_id, member, tag = line.split("\t")
        rows.append(dict(path=path, size=int(size), file_id=int(file_id),
                         member=member, tag=tag))
    return rows


def read_attributes(tag_name):
    """The HDF5 attributes stored with a file or dataset tag, as a dict."""
    connect()
    t = cte.Tag(tag_name)
    if "attributes" not in t.GetContainedBlobs():
        return {}
    return json.loads(_blob(t, "attributes").decode())


def read_table(tag_name):
    """A dataset ingested by the HDF5 assimilator, as a numpy array
    (a structured array for a compound/PyTables table)."""
    connect()
    t = cte.Tag(tag_name)
    desc = _blob(t, "description").decode().strip()
    m = re.fullmatch(r"compound<(\d+)((?:, *\d+)*)>\{(.*)\}", desc)
    if m:
        fields = [re.fullmatch(r"\s*(.+):(\w+)@(\d+)\s*", f).groups()
                  for f in m.group(3).split(",")]
        dtype = np.dtype({"names": [f[0] for f in fields],
                          "formats": [_DTYPES[f[1]] for f in fields],
                          "offsets": [int(f[2]) for f in fields],
                          "itemsize": int(m.group(1))})
        shape = tuple(int(x) for x in re.findall(r"\d+", m.group(2)))
    else:
        m = re.fullmatch(r"tensor<(\w+)((?:, *\d+)*)>", desc)
        if not m:
            raise ValueError(f"{tag_name}: unexpected description {desc!r}")
        dtype = np.dtype(_DTYPES[m.group(1)])
        shape = tuple(int(x) for x in re.findall(r"\d+", m.group(2)))
    return np.frombuffer(_chunks(t), dtype=dtype).reshape(shape)


def read_bytes(tag_name):
    """A file ingested as binary, as bytes."""
    connect()
    return _chunks(cte.Tag(tag_name))


def read_text(tag_name):
    return read_bytes(tag_name).decode("utf-8", "replace")
