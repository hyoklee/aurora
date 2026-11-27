import h5py, numpy as np
from scipy import stats
f=h5py.File('ss_ge.h5','r')
def parse(s):
    s=s.strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')
def load(q,k):
    s=f[q][k][()][0]
    if isinstance(s,(bytes,bytearray)): s=s.decode()
    return parse(s)
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
    ang=float(np.asarray(f[q]['Angle'][()]).ravel()[0])
    stored=float(np.asarray(f[q]['Fidelity'][()]).ravel()[0])
    c=np.cos(ang); s=np.sin(ang)
    xg=Ig*c - Qg*s
    xe=Ie*c - Qe*s
    if xe.mean()<xg.mean():
        xg=-xg; xe=-xe
    mg=np.median(xg); me=np.median(xe)
    gclean=xg[xg<=mg]; eclean=xe[xe>=me]
    sg=np.std(gclean,ddof=1); se=np.std(eclean,ddof=1)
    dmu=me-mg
    # empirical max threshold
    xs=np.sort(np.concatenate([xg,xe]))
    accs=np.array([(np.sum(xg<=t)+np.sum(xe>t))/(len(xg)+len(xe)) for t in xs])
    i=int(np.argmax(accs))
    tbest=xs[i]; femp=accs[i]
    # midpoint threshold
    tmid=(mg+me)/2
    fmid=(np.sum(xg<=tmid)+np.sum(xe>tmid))/(len(xg)+len(xe))
    # Gaussian model fidelity at midpoint using clean sigmas
    z=dmu/(2*np.sqrt((sg*sg+se*se)/2))  # not exactly
    # error probs with individual sigmas
    thr=tmid
    err_g=stats.norm.cdf(thr, loc=mg, scale=sg)  # g misclassified as e? g<=thr correct, so error g = P(x>thr|g) = 1-cdf
    err_g=1-stats.norm.cdf(thr, loc=mg, scale=sg)
    err_e=stats.norm.cdf(thr, loc=me, scale=se)
    fgauss=1-(err_g+err_e)/2
    # equal-variance gaussian
    se_bar=np.sqrt((sg*sg+se*se)/2)
    err=stats.norm.cdf(-dmu/(2*se_bar))
    feq=1-err
    print(f'{q}: stored={stored:.4f} femp_max={femp:.4f} (t={tbest:.2f}) fmid={fmid:.4f} fgauss_indiv={fgauss:.4f} feq={feq:.4f} SNR={dmu/se_bar:.2f}')
