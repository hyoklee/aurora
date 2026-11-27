import h5py, numpy as np, json
p = "/home/hyoklee/aurora/bench/qick/runs/agent_mistral-large-4/ss_ge.h5"
with h5py.File(p, "r") as f:
    for q in ["Q1","Q2","Q3","Q4","Q5","Q6"]:
        g = f[q]
        fid = g["Fidelity"][0]
        ang = g["Angle"][0]
        print(f"{q}: Fidelity={fid!r} Angle={ang!r}")
        for key in ["I_g","Q_g","I_e","Q_e"]:
            raw = g[key][0]
            if isinstance(raw, bytes):
                s = raw.decode("utf-8", errors="replace")
                print(f"  {key}: bytes len={len(raw)} head={s[:120]!r}")
                try:
                    arr = json.loads(s)
                    a = np.array(arr, dtype=float)
                    print(f"     -> json array shape={a.shape} dtype={a.dtype} min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g}")
                except Exception as e:
                    print(f"     -> json failed: {e}; trying np.frombuffer float64: {np.frombuffer(raw, dtype=np.float64)[:5]}")
            else:
                print(f"  {key}: type={type(raw)} val={raw!r}")
