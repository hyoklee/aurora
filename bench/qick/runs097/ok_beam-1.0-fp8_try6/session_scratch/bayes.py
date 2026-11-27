import h5py, numpy as np
from scipy.stats import norm
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
rg=Ig*np.cos(ang)-Qg*np.sin(ang)
re=Ie*np.cos(ang)-Qe*np.sin(ang)
mg=np.median(rg); me=np.median(re)
gc=rg[rg<=mg]; ec=re[re>=me]
sg=gc.std(ddof=1); se=ec.std(ddof=1)
# Bayesian optimal threshold (equal priors, different variances): solve log densities equal
best=None
for t in np.linspace(rg.min(),re.max(),200000):
    # log f_g(t)=log f_e(t)
    lhs=-0.5*((t-mg)/sg)**2 - np.log(sg)
    rhs=-0.5*((t-me)/se)**2 - np.log(se)
    val=abs(lhs-rhs)
    if best is None or val<best[0]:
        best=(val,t)
topt=best[1]
F=0.5*norm.cdf(topt,mg,sg)+0.5*(1-norm.cdf(topt,me,se))
print('Bayes optimal thr', topt, 'F', F)
print('stored thr', -221.59430332575388)
# Also, maybe stored fidelity uses threshold = mean of medians? midpoint of medians = -155.58 -> F 0.9177 (maxcum-ish)
# Let's find threshold where g_acc = ? maybe stored uses a fixed threshold from calibration 'threshold' in syst config
print('syst threshold list', f['Q1']['Syst Config'][()][0][:50])
