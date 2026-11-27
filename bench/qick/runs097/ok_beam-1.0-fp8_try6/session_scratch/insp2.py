import h5py, numpy as np
f = h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
def parse(s):
    s=s.decode() if isinstance(s,bytes) else s
    t=s.strip()
    if t.startswith('['): t=t[1:]
    if t.endswith(']'): t=t[:-1]
    return np.fromstring(t, sep=' ')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    g=f[q]
    print('===',q,'Angle',g['Angle'][()][0],'Fid',g['Fidelity'][()][0])
    for name in ['I_g','I_e','Q_g','Q_e']:
        a=parse(g[name][()][0])
        print('   ',name, a.size, 'mean', round(a.mean(),2), 'std', round(a.std(),2))
