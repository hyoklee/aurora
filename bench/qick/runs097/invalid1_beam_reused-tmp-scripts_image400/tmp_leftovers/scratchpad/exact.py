import h5py, numpy as np
def load(g,k):
    raw=g[k][()][0].decode('utf-8').strip().strip('[]')
    return np.fromstring(raw, sep=' ')
f=h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick_runs/coder_beam-1.0-fp8/ss_ge.h5','r')
g=f['Q1']
Ig=load(g,'I_g'); Qg=load(g,'Q_g'); Ie=load(g,'I_e'); Qe=load(g,'Q_e')
ang=float(g['Angle'][()][0])
xg = Ig*np.cos(ang) - Qg*np.sin(ang)
xe = Ie*np.cos(ang) - Qe*np.sin(ang)
yg = Ig*np.sin(ang) + Qg*np.cos(ang)
ye = Ie*np.sin(ang) + Qe*np.cos(ang)
vals=np.sort(np.concatenate([xg,xe]))
mids=(vals[1:]+vals[:-1])/2
best=0; bestt=None
for t in mids:
    acc=(np.sum(xg<=t)+np.sum(xe>t))/(len(xg)+len(xe))
    if acc>best: best=acc; bestt=t
print('max contrast=%.6f'%best)
print('threshold=%.6f'%bestt)
N=len(xg)+len(xe)
print('N_total', N)
F=best
delta=np.sqrt(F*(1-F)/N)
print('binomial delta=%.4f'%(delta))
mg=np.median(xg); me=np.median(xe)
cg=xg[xg<=mg]; ce=xe[xe>=me]
sg=np.std(cg, ddof=1); se=np.std(ce, ddof=1)
sbar=np.sqrt((sg*sg+se*se)/2)
print('clean sg=%.3f se=%.3f sbar=%.3f dmu=%.3f snr=%.3f nclean=%d,%d'%(sg,se,sbar,abs(me-mg),abs(me-mg)/sbar,len(cg),len(ce)))
print('g<=thr', np.sum(xg<=bestt), 'e>thr', np.sum(xe>bestt))
