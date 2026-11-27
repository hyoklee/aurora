import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
def parse(s):
    s=s.strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    g=f[q]
    print('==',q)
    print('Angle',float(np.asarray(g['Angle'][()]).ravel()[0]))
    print('Fidelity',float(np.asarray(g['Fidelity'][()]).ravel()[0]))
    print('Batch/Round',float(np.asarray(g['Batch Num'][()]).ravel()[0]),float(np.asarray(g['Round Num'][()]).ravel()[0]))
    for k in ['I_g','I_e','Q_g','Q_e']:
        s=g[k][()][0]
        if isinstance(s,(bytes,bytearray)): s=s.decode()
        arr=parse(s)
        print('   ',k,'n=',arr.size,'min',round(float(arr.min()),1),'max',round(float(arr.max()),1),'mean',round(float(arr.mean()),1))
