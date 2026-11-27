import h5py, numpy as np

def parse(b):
    s=b.decode().strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')

with h5py.File('ss_ge.h5','r') as f:
    for q in list(f.keys()):
        g=f[q]
        Ig=parse(g['I_g'][0]); Ie=parse(g['I_e'][0])
        Qg=parse(g['Q_g'][0]); Qe=parse(g['Q_e'][0])
        th=float(g['Angle'][0])
        stored=float(g['Fidelity'][0])
        Ig_r=Ig*np.cos(th)-Qg*np.sin(th)
        Ie_r=Ie*np.cos(th)-Qe*np.sin(th)
        med_g=np.median(Ig_r)
        fid_medg=0.5*(0.5 + (Ie_r>med_g).mean())
        # threshold at 0?
        fid_0=0.5*((Ig_r<=0).mean()+(Ie_r>0).mean())
        # threshold midpoint of means
        tm=(Ig_r.mean()+Ie_r.mean())/2
        fid_tm=0.5*((Ig_r<=tm).mean()+(Ie_r>tm).mean())
        print(f"{q}: stored={stored:.5f}  thr=med_g->{fid_medg:.5f}  thr=0->{fid_0:.5f}  thr=midmean->{fid_tm:.5f}")
