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
        # various thresholds
        for label, (a,b) in [('rawI',(Ig,Ie)),('rawQ',(Qg,Qe))].items() if False else []:
            pass
        print(f"\n{q} stored={stored:.5f} N_g={Ig.size} N_e={Ie.size}")
        print(f"  rawI: mean_g={Ig.mean():.1f} mean_e={Ie.mean():.1f}  acc@0={acc_at(Ig,Ie,0):.4f}")
        print(f"  rawI maxfid={max(acc_at(Ig,Ie,t) for t in np.unique(np.concatenate([Ig,Ie]))) if True else 0:.4f}" if False else '')
        # rotated minus
        Ig_r=Ig*np.cos(th)-Qg*np.sin(th); Ie_r=Ie*np.cos(th)-Qe*np.sin(th)
        print(f"  rotI: mean_g={Ig_r.mean():.1f} mean_e={Ie_r.mean():.1f}  acc@0={acc_at(Ig_r,Ie_r,0):.4f}")
        # try threshold = stored? find t giving stored
        vals=np.sort(np.unique(np.concatenate([Ig_r,Ie_r])))
        mids=np.concatenate(([vals[0]-1],(vals[1:]+vals[:-1])/2,[vals[-1]+1]))
        fs=np.array([acc_at(Ig_r,Ie_r,t) for t in mids])
        idx=np.argmin(np.abs(fs-stored))
        print(f"  rotI: threshold yielding stored F: t={mids[idx]:.2f} f={fs[idx]:.4f}")
        # min/max of acc
        print(f"  rotI: maxfid={fs.max():.4f} minfid={fs.min():.4f}")
