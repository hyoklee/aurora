import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
for q in f.keys():
    g=f[q]
    print('===',q,'Angle',g['Angle'][0],'Fidelity stored',g['Fidelity'][0])
    for name in ['I_g','Q_g','I_e','Q_e']:
        raw=g[name][0]
        s=raw.decode('ascii') if isinstance(raw,(bytes,np.bytes_)) else str(raw)
        arr=np.fromstring(s, sep=' ')
        print('   ',name,'len',arr.size,'min',arr.min(),'max',arr.max(),'mean',arr.mean(),'std',arr.std())
