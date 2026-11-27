import h5py, numpy as np
f = h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
def parse(s):
    s=s.decode() if isinstance(s,bytes) else s
    t=s.strip()
    if t.startswith('['): t=t[1:]
    if t.endswith(']'): t=t[:-1]
    return np.fromstring(t, sep=' ')
for q in ['Q1','Q2','Q3','Q5','Q6']:
    g=f[q]
    Ig=parse(g['I_g'][()][0]); Qg=parse(g['Q_g'][()][0])
    Ie=parse(g['I_e'][()][0]); Qe=parse(g['Q_e'][()][0])
    ang=g['Angle'][()][0]
    # rotation Ic - Qs (best delta)
    Ig_r=Ig*np.cos(ang)-Qg*np.sin(ang)
    Ie_r=Ie*np.cos(ang)-Qe*np.sin(ang)
    mg=np.median(Ig_r); me=np.median(Ie_r)
    # threshold midpoint
    thr=(mg+me)/2
    # fidelity variant A: (g<=thr + e>thr)/(2N)
    FA=(np.sum(Ig_r<=thr)+np.sum(Ie_r>thr))/(2*len(Ig_r))
    # maximize over thresholds (unique midpoints between sorted combined)
    vals=np.sort(np.concatenate([Ig_r,Ie_r]))
    thrs=(vals[1:]+vals[:-1])/2
    # vectorized-ish loop
    best=0; bestthr=None
    for t in thrs[::5]:
        F=(np.sum(Ig_r<=t)+np.sum(Ie_r>t))/(2*len(Ig_r))
        if F>best: best=F; bestthr=t
    # exact best by scanning all
    best2=0; bestthr2=None
    for t in thrs:
        F=(np.sum(Ig_r<=t)+np.sum(Ie_r>t))/(2*len(Ig_r))
        if F>best2: best2=F; bestthr2=t
    print(q, 'stored', round(g['Fidelity'][()][0],4),
          'midpointF', round(FA,4),
          'maxcumF', round(best2,4), 'thr', round(bestthr2,2))
