import h5py, numpy as np

def parse(b):
    s=b.decode().strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')

def acc_at(Ig,Ie,t):
    return 0.5*((Ig<=t).mean()+(Ie>t).mean())

with h5py.File('ss_ge.h5','r') as f:
    for q in list(f.keys()):
        g=f[q]
        Ig=parse(g['I_g'][0]); Ie=parse(g['I_e'][0])
        Qg=parse(g['Q_g'][0]); Qe=parse(g['Q_e'][0])
        th=float(g['Angle'][0])
        stored=float(g['Fidelity'][0])
        Ig_r=Ig*np.cos(th)-Qg*np.sin(th)
        Ie_r=Ie*np.cos(th)-Qe*np.sin(th)
        med_g=np.median(Ig_r); med_e=np.median(Ie_r)
        cg=Ig_r[Ig_r<=med_g]; ce=Ie_r[Ie_r>=med_e]
        sg=np.sqrt(np.mean((cg-med_g)**2)); se=np.sqrt(np.mean((ce-med_e)**2))
        # find threshold yielding stored
        vals=np.sort(np.unique(np.concatenate([Ig_r,Ie_r])))
        mids=np.concatenate(([vals[0]-1],(vals[1:]+vals[:-1])/2,[vals[-1]+1]))
        fs=np.array([acc_at(Ig_r,Ie_r,t) for t in mids])
        idx=np.argmin(np.abs(fs-stored))
        tstar=mids[idx]
        # k in sigma units from med_g and med_e
        kg=(tstar-med_g)/sg
        ke=(tstar-med_e)/se
        # also midpoint?
        print(f"{q}: stored={stored:.5f} t*={tstar:.2f} med_g={med_g:.2f} med_e={med_e:.2f} sg={sg:.2f} se={se:.2f} kg={kg:.3f} ke={ke:.3f}")
