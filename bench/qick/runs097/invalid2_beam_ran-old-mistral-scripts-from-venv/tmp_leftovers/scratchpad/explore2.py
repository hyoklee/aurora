import h5py, numpy as np

def load(q,name):
    f=h5py.File('ss_ge.h5','r')
    raw=f[q][name][()]
    s=raw[0].decode() if isinstance(raw,np.ndarray) else raw.decode()
    s=s.replace('[','').replace(']','')
    return np.fromstring(s, sep=' ')

f=h5py.File('ss_ge.h5','r')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    angle=float(f[q]['Angle'][0])
    Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
    dI=Ie.mean()-Ig.mean(); dQ=Qe.mean()-Qg.mean()
    sep_angle=np.arctan2(dQ,dI)
    print(q, 'stored_angle=',angle, 'sep_angle=',sep_angle, 'diff=',angle-sep_angle, 'sep_angle+pi?', sep_angle+np.pi)
