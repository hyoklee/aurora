import h5py, numpy as np
f = h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
def parse(s):
    s=s.decode() if isinstance(s,bytes) else s
    t=s.strip()
    if t.startswith('['): t=t[1:]
    if t.endswith(']'): t=t[:-1]
    return np.fromstring(t, sep=' ')
# Q4 is broken (Fid 0.0277, no separation) -> maybe wrong rotation? check raw scatter
for q in ['Q4']:
    g=f[q]
    Ig=parse(g['I_g'][()][0]); Qg=parse(g['Q_g'][()][0])
    Ie=parse(g['I_e'][()][0]); Qe=parse(g['Q_e'][()][0])
    print(q, 'Ig std', Ig.std(), 'Ie std', Ie.std())
    # maybe arrays mislabeled? check correlation / maybe qubit 4 data is noise only
    print('corr Ig,Ie', np.corrcoef(Ig,Ie)[0,1])
