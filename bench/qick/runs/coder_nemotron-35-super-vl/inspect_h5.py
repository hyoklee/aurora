import h5py
import numpy as np

f = h5py.File('ss_ge.h5', 'r')

def decode(s):
    """Decode a whitespace-separated numeric string dataset into a float array."""
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    s = s.strip().strip('[]')
    toks = s.split()
    return np.array(toks, dtype=float)

for q in sorted(f.keys(), key=lambda k: int(k[1:])):
    g = f[q]
    fid = float(np.ravel(g['Fidelity'][()])[0])
    angle = float(np.ravel(g['Angle'][()])[0])
    print('===', q, '===')
    print('  stored Fidelity =', fid)
    print('  Angle =', angle)
    print('  Dates =', g['Dates'][()], 'Batch Num =', g['Batch Num'][()], 'Round Num =', g['Round Num'][()])
    for k in ['I_g', 'I_e', 'Q_g', 'Q_e']:
        raw = g[k][()]
        if isinstance(raw, np.ndarray):
            raw = raw.ravel()[0]
        arr = decode(raw)
        print('   %-6s n=%d  min=% .3f  max=% .3f  mean=% .3f  std=% .3f' % (
            k, arr.size, arr.min(), arr.max(), arr.mean(), arr.std()))
    print()

f.close()
