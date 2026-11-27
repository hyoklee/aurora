import h5py, numpy as np

def parse(b):
    s=b.decode().strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')

def maxfid(Ig,Ie):
    # maximize 0.5*(mean(Ig<=t)+mean(Ie>t)) over thresholds (midpoints)
    vals=np.sort(np.unique(np.concatenate([Ig,Ie])))
    mids=np.concatenate(([vals[0]-1], (vals[1:]+vals[:-1])/2, [vals[-1]+1]))
    best=None
    for t in mids:
        f=0.5*((Ig<=t).mean()+(Ie>t).mean())
        if best is None or f>best[1]:
            best=(t,f)
    return best

with h5py.File('ss_ge.h5','r') as f:
    for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
        g=f[q]
        Ig=parse(g['I_g'][0]); Ie=parse(g['I_e'][0])
        Qg=parse(g['Q_g'][0]); Qe=parse(g['Q_e'][0])
        th=g['Angle'][0]
        stored=g['Fidelity'][0]
        # plus convention (matches paper SNR)
        Ig_r=Ig*np.cos(th)+Qg*np.sin(th)
        Ie_r=Ie*np.cos(th)+Qe*np.sin(th)
        t,f=maxfid(Ig_r,Ie_r)
        # binomial uncertainty
        N=Ig.size+Ie.size
        unc=np.sqrt(f*(1-f)/N)
        print(f"{q}: stored_F={stored:.5f}  computed_maxfid={f:.5f} ({f*100:.2f}%)  thr={t:.3f}  unc={unc*100:.3f}%  N={Ig.size}/{Ie.size}")
        # clean-side SNR
        med_g=np.median(Ig_r); med_e=np.median(Ie_r)
        cg=Ig_r[Ig_r<=med_g]; ce=Ie_r[Ie_r>=med_e]
        sg=np.sqrt(np.mean((cg-med_g)**2)); se=np.sqrt(np.mean((ce-med_e)**2))
        sb=np.sqrt((sg*sg+se*se)/2)
        print(f"     med_g={med_g:.2f} med_e={med_e:.2f} dmu={abs(med_e-med_g):.2f} sig_g={sg:.2f} sig_e={se:.2f} SNR={abs(med_e-med_g)/sb:.3f}")
