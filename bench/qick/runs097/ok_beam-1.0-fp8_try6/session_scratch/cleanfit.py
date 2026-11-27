import h5py, numpy as np
f = h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
def parse(s):
    s=s.decode() if isinstance(s,bytes) else s
    t=s.strip()
    if t.startswith('['): t=t[1:]
    if t.endswith(']'): t=t[:-1]
    return np.fromstring(t, sep=' ')
for q in ['Q1']:
    g=f[q]
    Ig=parse(g['I_g'][()][0]); Qg=parse(g['Q_g'][()][0])
    Ie=parse(g['I_e'][()][0]); Qe=parse(g['Q_e'][()][0])
    ang=g['Angle'][()][0]
    Ig_r=Ig*np.cos(ang)-Qg*np.sin(ang)
    Ie_r=Ie*np.cos(ang)-Qe*np.sin(ang)
    mg=np.median(Ig_r); me=np.median(Ie_r)
    # clean-side: g: I<=mg (outer? paper: I<=mu_eg for g; e: I>=mu_ee)
    gclean=Ig_r[Ig_r<=mg]
    eclean=Ie_r[Ie_r>=me]
    print('N clean g/e', gclean.size, eclean.size)
    sg=gclean.std(ddof=1); se=eclean.std(ddof=1)
    print('clean sigmas', sg, se, 'avg', (sg+se)/2)
    dmu=abs(me-mg)
    print('dmu', dmu, 'SNR', dmu/np.sqrt((sg*sg+se*se)/2))
    # SNR as paper: dmu / sigma_bar where sigma_bar = sqrt((sg^2+se^2)/2)
    sb=np.sqrt((sg*sg+se*se)/2)
    print('SNR', dmu/sb)
    # fidelity from Gaussian core? paper: max cumulative histogram contrast
    # also compute overlap integral of two gaussians with widths sg,se separated dmu
    from scipy.special import erf
    # misclassification g as e: P(I>thr | g) etc. For equal priors and threshold midpoint, fidelity = 0.5*(1-Phi((thr-mg)/sg)) + 0.5*Phi((thr-me)/se) ... actually assign g if I<=thr
    # optimal threshold for two gaussians with different variances: solve. Use maxcum already.
    vals=np.sort(np.concatenate([Ig_r,Ie_r]))
    thrs=(vals[1:]+vals[:-1])/2
    best=0; bestthr=None
    for t in thrs:
        F=(np.sum(Ig_r<=t)+np.sum(Ie_r>t))/(2*len(Ig_r))
        if F>best: best=F; bestthr=t
    print('maxcum F', best, 'thr', bestthr)
