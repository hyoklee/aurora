import h5py, numpy as np
f = h5py.File('ss_ge.h5','r')
g=f['Q1']
for name in ['I_g','Q_g','I_e','Q_e']:
    raw = g[name][()][0]
    text = raw.decode() if isinstance(raw, (bytes,bytearray)) else str(raw)
    import re
    nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', text)
    arr = np.array([float(x) for x in nums])
    print(name, arr.shape, arr[:5], '... len', len(arr))