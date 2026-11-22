import h5py, numpy as np, math

def parse_arr(ds):
    s = ds[0]
    if isinstance(s, bytes):
        s = s.decode()
    s = s.strip()
    return np.fromstring(s[1:-1], sep=' ')

def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

with h5py.File("ss_ge.h5", "r") as f:
    for i in range(1, 7):
        q = "Q%d" % i
        g = f[q]
        fid = float(g["Fidelity"][0])
        # scan N such that fid*N is (near) integer
        hits = []
        for N in range(2, 30001):
            v = fid * N
            if abs(v - round(v)) < 1e-7:
                hits.append((N, round(v)))
        print("%s stored=%.16f  integer-N hits (N,count) up to 30000: %s" % (q, fid, hits[:8]))

print()
# Gaussian z-values of stored fidelities
for q, fid in [("Q1", 0.8364727454575762), ("Q3", 0.7424570761793632), ("Q5", 0.7491248541423571)]:
    # invert Phi numerically
    lo, hi = -10, 10
    for _ in range(200):
        mid = (lo + hi) / 2
        if Phi(mid) < fid:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2
    print("%s stored=%.16f  -> Phi^-1 = %.6f ; if F=Phi(dmu/(2*sig)) then dmu/sig=%.6f" % (q, fid, z, 2 * z))
