import h5py, numpy as np

def parse(b):
    s=b.decode().strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')

def maxfid_threshold(Ig,Ie):
    vals=np.sort(np.unique(np.concatenate([Ig,Ie])))
    mids=np.concatenate(([vals[0]-1],(vals[1:]+vals[:-1])/2,[vals[-1]+1]))
    best=None
    for t in mids:
        f=0.5*((Ig<=t).mean()+(Ie>t).mean())
        if best is None or f>best[1]: best=(t,f)
    return best[0]

rng=np.random.default_rng(0)
with h5py.File('ss_ge.h5','r') as f:
    g=f['Q1']
    Ig=parse(g['I_g'][0]); Ie=parse(g['I_e'][0])
    Qg=parse(g['Q_g'][0]); Qe=parse(g['Q_e'][0])
    th=float(g['Angle'][0])
    Ig_r=Ig*np.cos(th)-Qg*np.sin(th)
    Ie_r=Ie*np.cos(th)-Qe*np.sin(th)
    # cross-val: threshold from half, eval on other half, 100 trials
    accs=[]
    for _ in range(200):
        ig=Ig_r.copy(); ie=Ie_r.copy()
        rng.shuffle(ig); rng.shuffle(ie)
        tr_g,te_g=np.split(ig,2); tr_e,te_e=np.split(ie,2)
        t=maxfid_threshold(tr_g,tr_e)
        acc=0.5*((te_g<=t).mean()+(te_e>t).mean())
        accs.append(acc)
    print('CV threshold (half/half) mean acc=%.4f ± %.4f'%(np.mean(accs),np.std(accs)))
    # 5-fold? 
    # threshold from g/e medians +0? 
    # Try threshold = median(g)+ (median(e)-median(g))*alpha
    med_g=np.median(Ig_r); med_e=np.median(Ie_r)
    for alpha in [0.1,0.2,0.3,0.4,0.5]:
        t=med_g+alpha*(med_e-med_g)
        acc=0.5*((Ig_r<=t).mean()+(Ie_r>t).mean())
        print(f'alpha={alpha}: t={t:.2f} acc={acc:.4f}')
    # stored t* was -221.6; express as fraction
    tstar=-221.59
    frac=(tstar-med_g)/(med_e-med_g)
    print('stored t* fraction from med_g to med_e: %.4f'%frac)
