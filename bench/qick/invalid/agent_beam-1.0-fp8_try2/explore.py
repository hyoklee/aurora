import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
def arr(raw):
    s=raw.decode('ascii') if isinstance(raw,(bytes,np.bytes_)) else str(raw)
    s=s.strip().strip('[]')
    return np.fromstring(s, sep=' ')
for q in f.keys():
    g=f[q]
    Ig=arr(g['I_g'][0]); Qg=arr(g['Q_g'][0]); Ie=arr(g['I_e'][0]); Qe=arr(g['Q_e'][0])
    th=g['Angle'][0]
    print('\n===',q,'theta',th,'storedF',g['Fidelity'][0])
    # convention A: Irot=I cos th - Q sin th; Qrot=I sin th + Q cos th
    for name,conv in [('A',1),('B',-1)]:
        if conv==1:
            IgA=Ig*np.cos(th)-Qg*np.sin(th); QgA=Ig*np.sin(th)+Qg*np.cos(th)
            IeA=Ie*np.cos(th)-Qe*np.sin(th); QeA=Ie*np.sin(th)+Qe*np.cos(th)
        else:
            IgA=Ig*np.cos(th)+Qg*np.sin(th); QgA=-Ig*np.sin(th)+Qg*np.cos(th)
            IeA=Ie*np.cos(th)+Qe*np.sin(th); QeA=-Ie*np.sin(th)+Qe*np.cos(th)
        print(f'  conv {name}: g I mean {IgA.mean():.2f} Q mean {QgA.mean():.2f} | e I mean {IeA.mean():.2f} Q mean {QeA.mean():.2f} | dI {IeA.mean()-IgA.mean():.2f} dQ {QeA.mean()-QgA.mean():.2f}')
