import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
g=f['Q1']
for k in ['Exp Config','Syst Config']:
    s=g[k][()][0]
    if isinstance(s,(bytes,bytearray)): s=s.decode()
    print('====',k,'====')
    print(s)
