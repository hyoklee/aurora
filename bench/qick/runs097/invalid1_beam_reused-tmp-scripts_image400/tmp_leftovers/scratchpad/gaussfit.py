import h5py, numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

def parse(b):
    s=b.decode().strip()
    if s.startswith('['): s=s[1:]
    if s.endswith(']'): s=s[:-1]
    return np.fromstring(s, sep=' ')

def gauss(x,mu,sig):
    return np.exp(-0.5*((x-mu)/sig)**2)/(sig*np.sqrt(2*np.pi))

with h5py.File('ss_ge.h5','r') as f:
    g=f['Q1']
    Ig=parse(g['I_g'][0]); Ie=parse(g['I_e'][0])
    Qg=parse(g['Q_g'][0]); Qe=parse(g['Q_e'][0])
    th=float(g['Angle'][0])
    Ig_r=Ig*np.cos(th)-Qg*np.sin(th)
    Ie_r=Ie*np.cos(th)-Qe*np.sin(th)
    # clean sides
    med_g=np.median(Ig_r); med_e=np.median(Ie_r)
    cg=Ig_r[Ig_r<=med_g]; ce=Ie_r[Ie_r>=med_e]
    # fit gaussian to histogram of clean side, location free? paper: clean-side Gaussian core fits
    # Fit full normal to clean side (truncated) using MLE: mu=median-ish, sigma from std of clean side corrected
    # Also try fitting gaussian to outer half via least squares on histogram
    for name, data, clean in [('g',Ig_r,cg),('e',Ie_r,ce)]:
        mu0=np.median(clean); sig0=clean.std()
        # MLE for truncated normal (lower/upper truncation at median)
        # Use scipy truncnorm fit? approximate: sigma_full = sigma_half / sqrt(1-2/pi)
        sig_full=clean.std(ddof=0)/np.sqrt(1-2/np.pi)
        print(name, 'median=%.2f clean_std=%.2f corrected_sigma=%.2f'%(mu0,clean.std(ddof=0),sig_full))
    dmu=abs(med_e-med_g)
    sg= (cg.std(ddof=0))/np.sqrt(1-2/np.pi)
    se= (ce.std(ddof=0))/np.sqrt(1-2/np.pi)
    sb=np.sqrt((sg**2+se**2)/2)
    print('SNR corrected = %.3f'%(dmu/sb))
    # Gaussian-overlap fidelity with unequal widths, threshold optimized
    from scipy.stats import norm
    best=None
    for t in np.linspace(med_g,med_e,100000):
        fid=0.5*(norm.cdf(t,med_g,sg)+(1-norm.cdf(t,med_e,se)))
        if best is None or fid>best[1]: best=(t,fid)
    print('Gaussian overlap (corrected sig) maxfid=%.4f'%best[1])
    # with RMS widths (uncorrected)
    sg2=np.sqrt(np.mean((cg-med_g)**2)); se2=np.sqrt(np.mean((ce-med_e)**2))
    sb2=np.sqrt((sg2**2+se2**2)/2)
    best=None
    for t in np.linspace(med_g,med_e,100000):
        fid=0.5*(norm.cdf(t,med_g,sg2)+(1-norm.cdf(t,med_e,se2)))
        if best is None or fid>best[1]: best=(t,fid)
    print('Gaussian overlap (RMS) maxfid=%.4f'%best[1], 'SNR=%.3f'%(dmu/sb2))
