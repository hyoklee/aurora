import h5py, numpy as np, re
f=h5py.File('ss_ge.h5','r')
def arr(raw):
    s=raw.decode('ascii') if isinstance(raw,(bytes,np.bytes_)) else str(raw)
    s=s.strip().strip('[]')
    return np.fromstring(s, sep=' ')
for q in f.keys():
    g=f[q]
    print('===',q,'Angle',g['Angle'][0],'Fidelity stored',g['Fidelity'][0])
    for name in ['I_g','Q_g','I_e','Q_e']:
        a=arr(g[name][0])
        print('   ',name,'len',a.size,'mean',a.mean(),'std',a.std())
