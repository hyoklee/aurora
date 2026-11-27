import h5py, numpy as np

def parse(b):
    s=b.decode().strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')

with h5py.File('ss_ge.h5','r') as f:
    for q in ['Q1']:
        g=f[q]
        Ig=parse(g['I_g'][0]); Ie=parse(g['I_e'][0])
        Qg=parse(g['Q_g'][0]); Qe=parse(g['Q_e'][0])
        th=g['Angle'][0]
        for conv in ['plus','minus']:
            if conv=='plus':
                Ig_r=Ig*np.cos(th)+Qg*np.sin(th)
                Ie_r=Ie*np.cos(th)+Qe*np.sin(th)
            else:
                Ig_r=Ig*np.cos(th)-Qg*np.sin(th)
                Ie_r=Ie*np.cos(th)-Qe*np.sin(th)
            med_g=np.median(Ig_r); med_e=np.median(Ie_r)
            # clean side samples
            clean_g=Ig_r[Ig_r<=med_g]
            clean_e=Ie_r[Ie_r>=med_e]
            # RMS width about median? or std?
            sig_g=np.sqrt(np.mean((clean_g-med_g)**2))
            sig_e=np.sqrt(np.mean((clean_e-med_e)**2))
            dmu=abs(med_e-med_g)
            sigbar=np.sqrt((sig_g**2+sig_e**2)/2)
            snr=dmu/sigbar
            print(q, conv, 'theta=%.4f'%th)
            print('  med_g=%.3f med_e=%.3f dmu=%.3f'%(med_g,med_e,dmu))
            print('  sig_g=%.3f (n=%d) sig_e=%.3f (n=%d) sigbar=%.3f'%(sig_g,len(clean_g),sig_e,len(clean_e),sigbar))
            print('  SNR=%.4f'%(snr))
            # also std (ddof?) 
            print('  std_g=%.3f std_e=%.3f'%(clean_g.std(),clean_e.std()))
