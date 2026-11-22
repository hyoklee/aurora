"""Inspect the structure of ss_ge.h5: groups, datasets, attributes, shapes, dtypes."""
import h5py
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "ss_ge.h5"


def visit(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}DATASET {name}: shape={obj.shape} dtype={obj.dtype}")
        for k, v in obj.attrs.items():
            print(f"{indent}   attr {k} = {v!r}")
    else:
        print(f"{indent}GROUP   {name}")
        for k, v in obj.attrs.items():
            print(f"{indent}   attr {k} = {v!r}")


with h5py.File(path, "r") as f:
    print("ROOT attributes:")
    for k, v in f.attrs.items():
        print(f"  {k} = {v!r}")
    print("\nStructure:")
    f.visititems(visit)
