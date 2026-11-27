import h5py, numpy as np

def parse(b):
    s=b.decode().strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')

with h5py.File('ss_ge.h5','r') as f:
    for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
        g=f[q]
        Ig=parse(g['I_g'][0]); Ie=parse(g['I_e'][0])
        Qg=parse(g['Q_g'][0]); Qe=parse(g['Q_e'][0])
        th=g['Angle'][0]
        mg=np.array([Ig.mean(),Qg.mean()]); me=np.array([Ie.mean(),Qe.mean()])
        sep=me-mg
        nat=np.arctan2(sep[1],sep[0])
        print(q, 'stored_angle=%.4f'%th, 'natural_sep_angle=%.4f'%nat, 'diff=%.4f'%(th-nat), 'medians_angle=%.4f'%np.arctan2(np.median(Ie)-np.median(Ig),0) if False else '')
        # rotated by stored angle: separation along I?
        for sign in [+1,-1]:
            a=sign*th
            Ig_r=Ig*np.cos(a)+Qg*np.sin(a) if False else Ig*np.cos(a)+ (Qg*np.sin(a) if sign>0 else -Qg*np.sin(a))
        # compute separation after rotating by -natural? 
