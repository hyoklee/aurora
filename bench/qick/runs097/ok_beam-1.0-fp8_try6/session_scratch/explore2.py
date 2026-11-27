import h5py, numpy as np
f = h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
def parse(s):
    s=s.decode() if isinstance(s,bytes) else s
    t=s.strip()
    if t.startswith('['): t=t[1:]
    if t.endswith(']'): t=t[:-1]
    return np.fromstring(t, sep=' ')
g=f['Q1']
Ig=parse(g['I_g'][()][0]); Qg=parse(g['Q_g'][()][0])
Ie=parse(g['I_e'][()][0]); Qe=parse(g['Q_e'][()][0])
ang=g['Angle'][()][0]
Ig_r=Ig*np.cos(ang)-Qg*np.sin(ang)
Ie_r=Ie*np.cos(ang)-Qe*np.sin(ang)
# full std
print('full std Ig_r', Ig_r.std(), 'Ie_r', Ie_r.std())
# threshold that yields stored 0.8365
stored=0.8364727454575762
vals=np.sort(np.concatenate([Ig_r,Ie_r]))
thrs=(vals[1:]+vals[:-1])/2
best=None
for t in thrs:
    F=(np.sum(Ig_r<=t)+np.sum(Ie_r>t))/(2*len(Ig_r))
    if best is None or abs(F-stored)<abs(best[1]-stored):
        best=(t,F)
print('closest threshold to stored:', best)
# maybe stored fidelity uses threshold=median of combined? 
thr=np.median(np.concatenate([Ig_r,Ie_r]))
print('combined median thr F', (np.sum(Ig_r<=thr)+np.sum(Ie_r>thr))/(2*len(Ig_r)))
# maybe stored uses a threshold from g/e means? try several
for name,thr in [('g_mean',Ig_r.mean()),('e_mean',Ie_r.mean()),('mid_means',(Ig_r.mean()+Ie_r.mean())/2),('g_median',np.median(Ig_r)),('e_median',np.median(Ie_r))]:
    F=(np.sum(Ig_r<=thr)+np.sum(Ie_r>thr))/(2*len(Ig_r))
    print(name, thr, F)
# maybe stored F is using Q (other quadrature) as discriminator
for arr_g,arr_e,label in [(Qg,Qe,'Q raw'),(Ig,Ie,'I raw')]:
    thr=(arr_g.mean()+arr_e.mean())/2
    F=(np.sum(arr_g<=thr)+np.sum(arr_e>thr))/(2*len(arr_g))
    print(label, 'meanthr F', F)
