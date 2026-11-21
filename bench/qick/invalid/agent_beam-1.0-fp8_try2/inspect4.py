import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
g=f['Q1']
raw=g['I_g'][0]
s=raw.decode('ascii')
print(repr(s[:200]))
print('last 100:', repr(s[-100:]))
# try np.loadtxt via StringIO
from io import StringIO
a=np.loadtxt(StringIO(s))
print('parsed', a.shape, a[:5], a[-3:])
