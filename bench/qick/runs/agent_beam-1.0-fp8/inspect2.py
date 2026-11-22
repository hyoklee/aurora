import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
for q in f.keys():
    g=f[q]
    print('===',q,'attrs',dict(g.attrs))
    for k,v in g.items():
        if isinstance(v,h5py.Dataset):
            print('  ',k,'shape',v.shape,'dtype',v.dtype)
            if v.size<=10 and v.dtype!=object and 'S' not in str(v.dtype):
                print('       data',v[...])
