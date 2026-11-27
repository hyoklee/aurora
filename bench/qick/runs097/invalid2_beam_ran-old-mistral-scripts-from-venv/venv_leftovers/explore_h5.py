import h5py, numpy as np
p = "/home/hyoklee/aurora/bench/qick/runs/agent_mistral-large-4/ss_ge.h5"
def walk(name, obj):
    pad = "  " * name.count("/")
    if isinstance(obj, h5py.Dataset):
        print(f"{pad}DS {name} shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{pad}GR {name}")
    for k, v in obj.attrs.items():
        print(f"{pad}   attr {k} = {v!r}")
with h5py.File(p, "r") as f:
    print("ROOT attrs:")
    for k, v in f.attrs.items():
        print(f"  attr {k} = {v!r}")
    f.visititems(walk)
