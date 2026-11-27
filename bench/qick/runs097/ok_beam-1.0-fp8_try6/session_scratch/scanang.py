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
best=None
for a in np.linspace(-np.pi,np.pi,361):
    Ig_r=Ig*np.cos(a)-Qg*np.sin(a)
    Ie_r=Ie*np.cos(a)-Qe*np.sin(a)
    mg=np.median(Ig_r); me=np.median(Ie_r)
    gclean=Ig_r[Ig_r<=mg]; eclean=Ie_r[Ie_r>=me]
    sg=gclean.std(ddof=1); se=eclean.std(ddof=1)
    sb=np.sqrt((sg*sg+se*se)/2)
    snr=abs(me-mg)/sb
    if best is None or snr>best[0]:
        best=(snr,a,sg,se,abs(me-mg))
print('best SNR', best[0], 'angle', best[1], 'sg', best[2], 'se', best[3], 'dmu', best[4])
print('stored angle', g['Angle'][()][0])
# also try other rotation convention + 
best2=None
for a in np.linspace(-np.pi,np.pi,361):
    Ig_r=Ig*np.cos(a)+Qg*np.sin(a)
    Ie_r=Ie*np.cos(a)+Qe*np.sin(a)
    mg=np.median(Ig_r); me=np.median(Ie_r)
    gclean=Ig_r[Ig_r<=mg]; eclean=Ie_r[Ie_r>=me]
    sg=gclean.std(ddof=1); se=eclean.std(ddof=1)
    sb=np.sqrt((sg*sg+se*se)/2)
    snr=abs(me-mg)/sb
    if best2 is None or snr>best2[0]:
        best2=(snr,a,sg,se,abs(me-mg))
print('plus best SNR', best2[0], 'angle', best2[1])
