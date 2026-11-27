import h5py, numpy as np

def parse(b):
    s=b.decode().strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')

with h5py.File('ss_ge.h5','r') as f:
    g=f['Q1']
    Ig=parse(g['I_g'][0]); Ie=parse(g['I_e'][0])
    Qg=parse(g['Q_g'][0]); Qe=parse(g['Q_e'][0])
    th=float(g['Angle'][0])
    # means
    dI=Ie.mean()-Ig.mean(); dQ=Qe.mean()-Qg.mean()
    print('mean dI=%.3f dQ=%.3f'% (dI,dQ))
    # theta to zero Q separation under standard rotation I'=I cos - Q sin, Q'=I sin + Q cos
    theta_opt=np.arctan2(-dQ, dI)  # tan theta = -dQ/dI
    print('theta_opt (standard)=%.4f'%theta_opt)
    print('stored theta=%.4f'%th)
    # medians
    dIm=np.median(Ie)-np.median(Ig); dQm=np.median(Qe)-np.median(Qg)
    print('median dI=%.3f dQ=%.3f theta_opt=%.4f'%(dIm,dQm,np.arctan2(-dQm,dIm)))
    # Apply standard rotation with stored theta
    Ig_r=Ig*np.cos(th)-Qg*np.sin(th); Qg_r=Ig*np.sin(th)+Qg*np.cos(th)
    Ie_r=Ie*np.cos(th)-Qe*np.sin(th); Qe_r=Ie*np.sin(th)+Qe*np.cos(th)
    print('standard stored: Q sep median=%.3f I sep median=%.3f'%(np.median(Qe_r)-np.median(Qg_r), np.median(Ie_r)-np.median(Ig_r)))
    # Apply standard rotation with theta_opt
    t=theta_opt
    Ig_r=Ig*np.cos(t)-Qg*np.sin(t); Qg_r=Ig*np.sin(t)+Qg*np.cos(t)
    Ie_r=Ie*np.cos(t)-Qe*np.sin(t); Qe_r=Ie*np.sin(t)+Qe*np.cos(t)
    print('standard opt: Q sep median=%.3f I sep median=%.3f'%(np.median(Qe_r)-np.median(Qg_r), np.median(Ie_r)-np.median(Ig_r)))
