import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
q='Q1'
g=f[q]
for name in ['I_g','Q_g','I_e','Q_e']:
    raw=g[name][0]
    print(name, type(raw), len(raw))
    print(repr(raw[:300]))
    print('---')
print('Angle',g['Angle'][0])
print('Exp Config head:', g['Exp Config'][0][:500])
print('Syst Config:', g['Syst Config'][0].decode('utf-8', 'replace'))
