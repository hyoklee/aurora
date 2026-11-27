import h5py, numpy as np, ast
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Use Q1 (qubit 0): best fidelity, matches paper's 91.7%
q='Q1'
Ig=load(q,'I_g'); Ie=load(q,'I_e'); Qg=load(q,'Q_g'); Qe=load(q,'Q_e')
ang=float(np.asarray(f[q]['Angle'][()]).ravel()[0])
c=np.cos(ang); s=np.sin(ang)
# rotate so separation aligns with I (x); orient e to the right
xg=Ig*c - Qg*s
xe=Ie*c - Qe*s
yg=Ig*s + Qg*c
ye=Ie*s + Qe*c
if xe.mean()<xg.mean():
    xg=-xg; xe=-xe; yg=-yg; ye=-ye; ang=-ang
mg=np.median(xg); me=np.median(xe)
# clean-side (outer half) widths
gclean=xg[xg<=mg]; eclean=xe[xe>=me]
sg=np.std(gclean,ddof=1); se=np.std(eclean,ddof=1)
# optimal threshold
xs=np.sort(np.concatenate([xg,xe]))
accs=np.array([(np.sum(xg<=t)+np.sum(xe>t))/(len(xg)+len(xe)) for t in xs])
ti=int(np.argmax(accs))
thr=xs[ti]; F=accs[ti]
Ntot=len(xg)+len(xe)
dF=np.sqrt(F*(1-F)/Ntot)

# Gaussian core fits (clean-side sigma, median mean)
xx=np.linspace(min(xg.min(),xe.min())-3*max(sg,se), max(xg.max(),xe.max())+3*max(sg,se), 400)

fig, axes=plt.subplots(1,2,figsize=(11,5))
# (a) rotated I-Q scatter
ax=axes[0]
ax.scatter(xg,yg,s=6,c='tab:blue',alpha=0.35,label='|g>')
ax.scatter(xe,ye,s=6,c='tab:red',alpha=0.35,label='|e>')
ax.scatter([mg],[np.median(yg)],s=80,c='k',marker='o',zorder=5)
ax.scatter([me],[np.median(ye)],s=80,c='k',marker='o',zorder=5)
ax.set_xlabel('I (rotated) [a.u.]')
ax.set_ylabel('Q (rotated) [a.u.]')
ax.set_title('(a) Rotated I-Q scatter (3000 shots/state)')
ax.legend(loc='best',markerscale=2)
ax.grid(alpha=0.2)

# (b) projected histogram
ax=axes[1]
bins=np.linspace(min(xg.min(),xe.min()), max(xg.max(),xe.max()), 61)
ax.hist(xg,bins=bins,density=True,color='tab:blue',alpha=0.45,label='|g>')
ax.hist(xe,bins=bins,density=True,color='tab:red',alpha=0.45,label='|e>')
ax.plot(xx, 0.5*stats.norm.pdf(xx,loc=mg,scale=sg), 'b--', lw=2, label='Gaussian core |g>')
ax.plot(xx, 0.5*stats.norm.pdf(xx,loc=me,scale=se), 'r--', lw=2, label='Gaussian core |e>')
ax.axvline(thr,color='k',ls=':',lw=1.5,label=f'threshold={thr:.1f}')
ax.set_xlabel('I (rotated) [a.u.]')
ax.set_ylabel('Probability density')
ax.set_title(f'(b) Projected histogram, F={F*100:.1f}%')
ax.legend(loc='best',fontsize=8)
ax.grid(alpha=0.2)
fig.suptitle('Q1 (qubit 0) single-shot readout (ss_ge.h5)  |  F = %.2f%% ± %.2f%%'%(F*100,dF*100), fontsize=12)
fig.tight_layout()
fig.savefig('readout.png',dpi=150)
print('saved readout.png')
print('F=%.6f dF=%.6f thr=%.3f mg=%.3f me=%.3f sg=%.3f se=%.3f dmu=%.3f'%(F,dF,thr,mg,me,sg,se,me-mg))
