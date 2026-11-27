import h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

H5='/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5'
OUT='/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/readout.png'

def parse(s):
    s=s.decode() if isinstance(s,bytes) else s
    t=s.strip()
    if t.startswith('['): t=t[1:]
    if t.endswith(']'): t=t[:-1]
    return np.fromstring(t, sep=' ')

f=h5py.File(H5,'r')
g=f['Q1']
Ig=parse(g['I_g'][()][0]); Qg=parse(g['Q_g'][()][0])
Ie=parse(g['I_e'][()][0]); Qe=parse(g['Q_e'][()][0])
ang=float(g['Angle'][()][0])
# rotation aligning max separation with I (stored convention: I cos - Q sin)
Ig_r=Ig*np.cos(ang)-Qg*np.sin(ang)
Qg_r=Ig*np.sin(ang)+Qg*np.cos(ang)  # perpendicular (not used for hist but for scatter y)
Ie_r=Ie*np.cos(ang)-Qe*np.sin(ang)
Qe_r=Ie*np.sin(ang)+Qe*np.cos(ang)

mg=np.median(Ig_r); me=np.median(Ie_r)
gc=Ig_r[Ig_r<=mg]; ec=Ie_r[Ie_r>=me]
sg=gc.std(ddof=1); se=ec.std(ddof=1)
sb=np.sqrt((sg*sg+se*se)/2)
dmu=abs(me-mg)
snr=dmu/sb
vals=np.sort(np.concatenate([Ig_r,Ie_r]))
thrs=(vals[1:]+vals[:-1])/2
best=0; bestthr=None
for t in thrs:
    F=(np.sum(Ig_r<=t)+np.sum(Ie_r>t))/(2*len(Ig_r))
    if F>best: best=F; bestthr=t
F=best

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.5))
# (a) scatter
ax1.scatter(Ig_r, Qg_r, s=4, c='tab:blue', alpha=0.5, label='|g>')
ax1.scatter(Ie_r, Qe_r, s=4, c='tab:red', alpha=0.5, label='|e>')
ax1.plot(mg, np.median(Qg_r), 'k+', ms=12, mew=2)
ax1.plot(me, np.median(Qe_r), 'k+', ms=12, mew=2)
ax1.set_xlabel('I (rotated)')
ax1.set_ylabel('Q (rotated)')
ax1.set_title('(a) Rotated I-Q scatter (Q1)')
ax1.legend(markerscale=3, fontsize=9)
ax1.grid(alpha=0.3)

# (b) histogram
bins=np.linspace(min(Ig_r.min(),Ie_r.min()), max(Ig_r.max(),Ie_r.max()), 61)
bw=bins[1]-bins[0]
ax2.hist(Ig_r, bins=bins, color='tab:blue', alpha=0.5, density=False, label='|g>')
ax2.hist(Ie_r, bins=bins, color='tab:red', alpha=0.5, density=False, label='|e>')
x=np.linspace(bins[0],bins[-1],500)
# scale gaussian to counts: N*binwidth*pdf
ax2.plot(x, len(Ig_r)*bw*norm.pdf(x, mg, sg), 'b--', lw=1.5, label='g core fit')
ax2.plot(x, len(Ie_r)*bw*norm.pdf(x, me, se), 'r--', lw=1.5, label='e core fit')
ax2.axvline(bestthr, color='k', ls=':', lw=1.2, label=f'thr={bestthr:.1f}')
ax2.set_xlabel('I (rotated)')
ax2.set_ylabel('Counts')
ax2.set_title('(b) Projected histogram')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.suptitle(f'Q1 single-shot readout: SNR={snr:.2f}, F={F*100:.1f}% (max cumulative contrast), N=3000/state', fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print('saved', OUT)
print('F', F, 'SNR', snr, 'sg', sg, 'se', se, 'dmu', dmu)
