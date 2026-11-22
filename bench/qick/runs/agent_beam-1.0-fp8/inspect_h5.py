import h5py, numpy as np
p='ss_ge.h5'
f=h5py.File(p,'r')
print('ROOT ATTRS:', dict(f.attrs))
for k,v in f.attrs.items():
    print('ATTR',repr(k),repr(v))

def walk(g,prefix=''):
    for k,v in g.items():
        path=prefix+'/'+k
        if isinstance(v,h5py.Group):
            print('GROUP',path,'attrs=',dict(v.attrs))
            walk(v,path)
        else:
            print('DATASET',path,'shape=',v.shape,'dtype=',v.dtype,'attrs=',dict(v.attrs))
            # print small scalar dataset values
            if v.size<=20:
                print('  data=',v[...])
walk(f)
print('keys top', list(f.keys()))
