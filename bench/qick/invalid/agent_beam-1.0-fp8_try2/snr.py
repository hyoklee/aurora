import h5py, numpy as np
f=h5py.File('ss_ge.h5','r')
def arr(raw):
    s=raw.decode('ascii') if isinstance(raw,(bytes,np.bytes_)) else str(raw)
    s=s.strip().strip('[]')
    return np.fromstring(s, sep=' ')
q='Q1'; g=f[q]
Ig=arr(g['I_g'][0]); Qg=arr(g['Q_g'][0]); Ie=arr(g['I_e'][0]); Qe=arr(g['Q_e'][0])
th=g['Angle'][0]
Xg=Ig*np.cos(th)-Qg*np.sin(th); Xe=Ie*np.cos(th)-Qe*np.sin(th)
medg=np.median(Xg); mede=np.median(Xe)
cg=Xg[Xg<=medg]; ce=Xe[Xe>=mede]
print('Q1 dmu', mede-medg)
print('clean g n',cg.size,'clean e n',ce.size)
print('sample std g/e', cg.std(ddof=1), ce.std(ddof=1))
print('RMS about medg g', np.sqrt(np.mean((cg-medg)**2)), 'RMS about mede e', np.sqrt(np.mean((ce-mede)**2)))
trunc_factor=np.sqrt(1-2/np.pi)
print('trunc-corrected sig g/e', cg.std(ddof=1)/trunc_factor, ce.std(ddof=1)/trunc_factor)
sigbar=np.sqrt(((cg.std(ddof=1)/trunc_factor)**2+(ce.std(ddof=1)/trunc_factor)**2)/2)
print('SNR trunc-corrected', abs(mede-medg)/sigbar)
