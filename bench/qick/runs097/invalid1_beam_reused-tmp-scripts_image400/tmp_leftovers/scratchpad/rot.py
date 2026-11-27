import h5py, numpy as np
def load(g,k):
    raw=g[k][()][0].decode('utf-8').strip().strip('[]')
    return np.fromstring(raw, sep=' ')
f=h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick_runs/coder_beam-1.0-fp8/ss_ge.h5','r')
g=f['Q1']
Ig=load(g,'I_g'); Qg=load(g,'Q_g'); Ie=load(g,'I_e'); Qe=load(g,'Q_e')
ang=float(g['Angle'][()][0])
variants = {
 'A stored': lambda I,Q: (I*np.cos(ang)-Q*np.sin(ang), I*np.sin(ang)+Q*np.cos(ang)),
 'B neg stored': lambda I,Q: (I*np.cos(-ang)-Q*np.sin(-ang), I*np.sin(-ang)+Q*np.cos(-ang)),
 'C Icos+Qsin stored': lambda I,Q: (I*np.cos(ang)+Q*np.sin(ang), -I*np.sin(ang)+Q*np.cos(ang)),
}
for name,fn in variants.items():
    Ig2,Qg2=fn(Ig,Qg); Ie2,Qe2=fn(Ie,Qe)
    print(name, 'g Qmed=%.2f'%np.median(Qg2), 'e Qmed=%.2f'%np.median(Qe2), 'g Imed=%.2f'%np.median(Ig2), 'e Imed=%.2f'%np.median(Ie2))
