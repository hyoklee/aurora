import h5py, numpy as np
f = h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    g=f[q]
    info={'Angle':g['Angle'][()][0],'Fidelity':g['Fidelity'][()][0]}
    for arrname in ['I_g','I_e','Q_g','Q_e']:
        s = g[arrname][()][0]
        if isinstance(s, bytes): s=s.decode()
        a = np.fromstring(s, sep=' ')
        info[arrname]=a.size
    print(q, info)
