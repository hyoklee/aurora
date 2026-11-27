import h5py, numpy as np

def parse(b):
    s=b.decode().strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')

def maxfid1d(a,b):
    vals=np.sort(np.unique(np.concatenate([a,b])))
    mids=np.concatenate(([vals[0]-1], (vals[1:]+vals[:-1])/2, [vals[-1]+1]))
    best=None
    for t in mids:
        f=0.5*((a<=t).mean()+(b>t).mean())
        if best is None or f>best[1]:
            best=(t,f)
    return best

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

with h5py.File('ss_ge.h5','r') as f:
    g=f['Q1']
    Ig=parse(g['I_g'][0]); Ie=parse(g['I_e'][0])
    Qg=parse(g['Q_g'][0]); Qe=parse(g['Q_e'][0])
    th=float(g['Angle'][0])
    # raw I
    t,f=maxfid1d(Ig,Ie); print('raw I maxfid=%.4f'%f)
    # raw Q
    t,f=maxfid1d(Qg,Qe); print('raw Q maxfid=%.4f'%f)
    # LDA 2D
    X=np.vstack([np.column_stack([Ig,Qg]),np.column_stack([Ie,Qe])])
    y=np.array([0]*len(Ig)+[1]*len(Ie))
    lda=LinearDiscriminantAnalysis()
    lda.fit(X,y)
    pred=lda.predict(X)
    print('LDA in-sample acc=%.4f'%accuracy_score(y,pred))
    # rotated plus 2D LDA
    Ig_r=Ig*np.cos(th)+Qg*np.sin(th); Qg_r=-Ig*np.sin(th)+Qg*np.cos(th)
    Ie_r=Ie*np.cos(th)+Qe*np.sin(th); Qe_r=-Ie*np.sin(th)+Qe*np.cos(th)
    Xr=np.vstack([np.column_stack([Ig_r,Qg_r]),np.column_stack([Ie_r,Qe_r])])
    lda.fit(Xr,y); pred=lda.predict(Xr)
    print('LDA rotated plus acc=%.4f'%accuracy_score(y,pred))
    # nearest centroid on I only after rotation (same as threshold) already
    # Gaussian overlap model: fidelity = 0.5*(1+erf(dmu/(2*sqrt2*sigbar)))? 
    import scipy.stats as st
    med_g=np.median(Ig_r); med_e=np.median(Ie_r)
    cg=Ig_r[Ig_r<=med_g]; ce=Ie_r[Ie_r>=med_e]
    sg=np.sqrt(np.mean((cg-med_g)**2)); se=np.sqrt(np.mean((ce-med_e)**2))
    sb=np.sqrt((sg*sg+se*se)/2)
    dmu=abs(med_e-med_g)
    # simple Gaussian overlap fidelity with equal priors, threshold at midpoint
    thr=(med_g+med_e)/2
    # P(g classified g)=Phi((thr-med_g)/sg); P(e classified e)=1-Phi((thr-med_e)/se)
    fid_g=st.norm.cdf((thr-med_g)/sg)
    fid_e=1-st.norm.cdf((thr-med_e)/se)
    print('Gaussian model fid=%.4f'%(0.5*(fid_g+fid_e)))
    print('paper SNR=%.3f'%(dmu/sb))
