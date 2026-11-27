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
    # minus convention
    Ig_r=Ig*np.cos(th)-Qg*np.sin(th)
    Ie_r=Ie*np.cos(th)-Qe*np.sin(th)
    med_g=np.median(Ig_r); med_e=np.median(Ie_r)
    cg=Ig_r[Ig_r<=med_g]; ce=Ie_r[Ie_r>=med_e]
    dmu=abs(med_e-med_g)
    # variant widths
    for label,sig_g,sig_e in [
        ('RMS_about_median', np.sqrt(np.mean((cg-med_g)**2)), np.sqrt(np.mean((ce-med_e)**2))),
        ('std_of_clean_half', cg.std(ddof=0), ce.std(ddof=0)),
        ('std_ddof1', cg.std(ddof=1), ce.std(ddof=1)),
    ]:
        sb=np.sqrt((sig_g**2+sig_e**2)/2)
        print(f'{label}: sig_g={sig_g:.3f} sig_e={sig_e:.3f} sb={sb:.3f} SNR={dmu/sb:.3f}')
    # corrected: sigma = std_half / sqrt(1-2/pi)  (full gaussian sigma from half-sample std)
    c= np.sqrt(1-2/np.pi)
    sg=cg.std(ddof=0)/c; se=ce.std(ddof=0)/c
    sb=np.sqrt((sg**2+se**2)/2)
    print(f'corrected sigma: sig_g={sg:.3f} sig_e={se:.3f} sb={sb:.3f} SNR={dmu/sb:.3f}')
    # maybe width = interquartile? 
    # Gaussian fit via MLE on truncated lower half (scipy)
    from scipy.stats import truncnorm
    # fit sigma by method of moments already above
    # Try fitting normal to clean half with location fixed at median? 
    print('paper targets: SNR=4.4, dmu=28 (scaled?), widths 6.4/6.5')
    print('our dmu=%.2f'%dmu)
