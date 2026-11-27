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
    for conv in ['minus','plus']:
        if conv=='minus':
            Ig_r=Ig*np.cos(ang)-Qg*np.sin(ang)
            Ie_r=Ie*np.cos(ang)-Qe*np.sin(ang)
        else:
            Ig_r=Ig*np.cos(ang)+Qg*np.sin(ang)
            Ie_r=Ie*np.cos(ang)+Qe*np.sin(ang)
        print(conv, 'dg mean', Ig_r.mean(), 'de mean', Ie_r.mean(), 'delta', abs(Ie_r.mean()-Ig_r.mean()))
