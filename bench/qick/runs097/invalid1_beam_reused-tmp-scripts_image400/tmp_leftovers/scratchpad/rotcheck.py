import h5py, numpy as np

def parse(b):
    s=b.decode().strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')

with h5py.File('ss_ge.h5','r') as f:
    g=f['Q1']
    Ig=parse(g['I_g'][0]); Ie=parse(g['I_e'][0])
    Qg=parse(g['Q_g'][0]); Qe=parse(g['Q_e'][0])
    th=float(g['Angle'][0])
    for name, (Ig_r,Qg_r,Ie_r,Qe_r) in {
        'plus': (Ig*np.cos(th)+Qg*np.sin(th), -Ig*np.sin(th)+Qg*np.cos(th),
                 Ie*np.cos(th)+Qe*np.sin(th), -Ie*np.sin(th)+Qe*np.cos(th)),
        'minus': (Ig*np.cos(th)-Qg*np.sin(th), Ig*np.sin(th)+Qg*np.cos(th),
                  Ie*np.cos(th)-Qe*np.sin(th), Ie*np.sin(th)+Qe*np.cos(th)),
    }.items():
        print(name)
        print('  I sep (mean diff) =', Ig_r.mean()-Ie_r.mean() if False else (Ie_r.mean()-Ig_r.mean()))
        print('  Q sep (mean diff) =', Qe_r.mean()-Qg_r.mean())
        print('  I sep (median diff)=', np.median(Ie_r)-np.median(Ig_r))
        print('  Q sep (median diff)=', np.median(Qe_r)-np.median(Qg_r))
