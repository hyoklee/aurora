import h5py, numpy as np

def parse(b):
    s=b.decode().strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')

def rot(I,Q,theta):
    # try convention 1: I' = I cos + Q sin
    return I*np.cos(theta)+Q*np.sin(theta), -I*np.sin(theta)+Q*np.cos(theta)

with h5py.File('ss_ge.h5','r') as f:
    for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
        g=f[q]
        Ig=parse(g['I_g'][0]); Ie=parse(g['I_e'][0])
        Qg=parse(g['Q_g'][0]); Qe=parse(g['Q_e'][0])
        th=g['Angle'][0]
        for name,rr in [('Icos+Qsin', lambda I,Q: I*np.cos(th)+Q*np.sin(th)),
                        ('Icos-Qsin', lambda I,Q: I*np.cos(th)-Q*np.sin(th))]:
            Ig_r=rr(Ig,Qg); Ie_r=rr(Ie,Qe)
            # threshold maximizing (correct g below + correct e above)/2
            allv=np.concatenate([Ig_r,Ie_r])
            best=None
            for t in np.unique(np.round(allv,3)):
                fid=0.5*((Ig_r<=t).mean()+(Ie_r>=t).mean())
                if best is None or fid>best[1]:
                    best=(t,fid)
            # also threshold at midpoint of medians
            med_g=np.median(Ig_r); med_e=np.median(Ie_r)
            tmid=(med_g+med_e)/2
            fid_mid=0.5*((Ig_r<=tmid).mean()+(Ie_r>=tmid).mean())
            print(q, name, 'best_fid=%.4f'%best[1], 't=%.3f'%best[0], 'mid_fid=%.4f'%fid_mid, 'med_g=%.2f med_e=%.2f'%(med_g,med_e))
