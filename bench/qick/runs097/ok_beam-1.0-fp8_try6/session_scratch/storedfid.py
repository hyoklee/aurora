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
    # stored fidelity maybe using threshold from Syst Config 'threshold' list?
    # threshold list for ro_ch; qubit 0 threshold -5.84 (but different channel)
    print('stored', g['Fidelity'][()][0])
    # fidelity if threshold = 0
    for thr in [0, (Ig_r.mean()+Ie_r.mean())/2, np.median(Ig_r), np.median(Ie_r)]:
        F=(np.sum(Ig_r<=thr)+np.sum(Ie_r>thr))/(2*len(Ig_r))
        print('thr', thr, 'F', F)
    # maybe stored fidelity is from a different qubit index? list qubit freqs...
    # try without rotation, threshold 0
    F=(np.sum(Ig<=0)+np.sum(Ie>0))/(2*len(Ig))
    print('raw I thr0 F', F)
    # try rotation plus, threshold midpoint
    Ig_r2=Ig*np.cos(ang)+Qg*np.sin(ang)
    Ie_r2=Ie*np.cos(ang)+Qe*np.sin(ang)
    thr=(Ig_r2.mean()+Ie_r2.mean())/2
    F=(np.sum(Ig_r2<=thr)+np.sum(Ie_r2>thr))/(2*len(Ig))
    print('plus meanthr F', F)
