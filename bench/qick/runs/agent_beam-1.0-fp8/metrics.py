import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
def arr(raw):
    s=raw.decode('ascii') if isinstance(raw,(bytes,np.bytes_)) else str(raw)
    s=s.strip().strip('[]')
    return np.fromstring(s, sep=' ')

def cdf_contrast(xg, xe):
    # max over thresholds of 0.5*(CDFg(t)+(1-CDFe(t)))
    xs=np.sort(np.unique(np.concatenate([xg,xe])))
    # evaluate at midpoints
    ts=(xs[:-1]+xs[1:])/2
    # vectorized cumulative counts
    # For each t, CDFg = mean(xg<=t), CDFe=mean(xe<=t)
    # compute efficiently using searchsorted
    ng=len(xg); ne=len(xe)
    cg=np.searchsorted(np.sort(xg), ts, side='right')/ng
    ce=np.searchsorted(np.sort(xe), ts, side='right')/ne
    F=0.5*(cg + (1-ce))
    idx=np.argmax(F)
    D=np.max(cg-ce)  # KS
    return F[idx], ts[idx], D, 0.5*(1+D)

for q in f.keys():
    g=f[q]
    Ig=arr(g['I_g'][0]); Qg=arr(g['Q_g'][0]); Ie=arr(g['I_e'][0]); Qe=arr(g['Q_e'][0])
    th=g['Angle'][0]
    # conv A
    Xg=Ig*np.cos(th)-Qg*np.sin(th)
    Xe=Ie*np.cos(th)-Qe*np.sin(th)
    medg=np.median(Xg); mede=np.median(Xe)
    # clean sides
    clean_g=Xg[Xg<=medg]
    clean_e=Xe[Xe>=mede]
    sig_g=clean_g.std(ddof=1)
    sig_e=clean_e.std(ddof=1)
    sigbar=np.sqrt((sig_g**2+sig_e**2)/2)
    dmu=abs(mede-medg)
    SNR=dmu/sigbar
    Fopt, thr, D, Fks = cdf_contrast(Xg, Xe)
    # midpoint threshold
    thr_mid=(medg+mede)/2
    Fmid=0.5*(np.mean(Xg<=thr_mid)+np.mean(Xe>=thr_mid))
    print(f'{q}: stored={g["Fidelity"][0]:.4f} dmu={dmu:.2f} medg={medg:.2f} mede={mede:.2f}')
    print(f'   clean n_g={clean_g.size} n_e={clean_e.size} sig_g={sig_g:.3f} sig_e={sig_e:.3f} sigbar={sigbar:.3f} SNR={SNR:.3f}')
    print(f'   F_opt={Fopt:.4f} thr={thr:.2f} F_ks=(1+D)/2={Fks:.4f} D={D:.4f} F_mid={Fmid:.4f}')
