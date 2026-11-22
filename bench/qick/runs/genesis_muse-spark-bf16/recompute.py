import h5py, numpy as np
fpath="ss_ge.h5"

def get_arr(grp, name):
    raw=grp[name][0]
    txt=raw.decode() if isinstance(raw, bytes) else str(raw)
    arr=np.fromstring(txt[1:-1], sep=' ')
    return arr

with h5py.File(fpath,'r') as f:
  for q in sorted(f.keys()):
    grp=f[q]
    Ig=get_arr(grp,'I_g'); Qg=get_arr(grp,'Q_g')
    Ie=get_arr(grp,'I_e'); Qe=get_arr(grp,'Q_e')
    angle=float(grp['Angle'][0])
    fid_stored=float(grp['Fidelity'][0])
    def rotate(I,Q,theta):
        Ir = I*np.cos(theta) - Q*np.sin(theta)
        Qr = I*np.sin(theta) + Q*np.cos(theta)
        return Ir, Qr
    Ig_r, Qg_r = rotate(Ig,Qg, angle)
    Ie_r, Qe_r = rotate(Ie,Qe, angle)
    best_angle = None
    best_sep = -1
    for theta in np.linspace(-np.pi, np.pi, 720):
        Ig_t,Qg_t = rotate(Ig,Qg, theta)
        Ie_t,Qe_t = rotate(Ie,Qe, theta)
        sep = abs(np.median(Ie_t) - np.median(Ig_t))
        if sep>best_sep:
            best_sep=sep
            best_angle=theta
    def fidelity_max_contrast(Ig_proj, Ie_proj):
        Ig_sorted = np.sort(Ig_proj)
        Ie_sorted = np.sort(Ie_proj)
        mu_g = np.median(Ig_proj); mu_e = np.median(Ie_proj)
        flip = mu_e < mu_g
        all_vals = np.sort(np.concatenate([Ig_proj, Ie_proj]))
        thresholds = all_vals[::5]
        best_fid = 0
        best_T = None
        for T in thresholds:
            cdf_g = np.searchsorted(Ig_sorted, T, side='right')/len(Ig_sorted)
            cdf_e = np.searchsorted(Ie_sorted, T, side='right')/len(Ie_sorted)
            if not flip:
                fid = (cdf_g + (1 - cdf_e))/2
            else:
                fid = ((1 - cdf_g) + cdf_e)/2
            if fid > best_fid:
                best_fid = fid
                best_T = T
        return best_fid, best_T, flip
    fid_calc, T_opt, flip = fidelity_max_contrast(Ig_r, Ie_r)
    Ig_opt, _ = rotate(Ig,Qg,best_angle)
    Ie_opt, _ = rotate(Ie,Qe,best_angle)
    fid_opt,_,_ = fidelity_max_contrast(Ig_opt, Ie_opt)
    def clean_sigma(Ig_proj, Ie_proj):
        mu_g = np.median(Ig_proj); mu_e=np.median(Ie_proj)
        flip_local = mu_e < mu_g
        if not flip_local:
            clean_g = Ig_proj[Ig_proj <= mu_g]
            clean_e = Ie_proj[Ie_proj >= mu_e]
        else:
            clean_g = Ig_proj[Ig_proj >= mu_g]
            clean_e = Ie_proj[Ie_proj <= mu_e]
        def est_sigma(clean, mu):
            dev = clean - mu
            if len(dev)==0:
                return np.nan, np.nan, np.nan
            std_trunc = np.std(dev, ddof=0)
            sigma_corr = std_trunc / np.sqrt(1 - 2/np.pi) if len(dev)>1 else np.nan
            sigma_meanabs = np.sqrt(np.pi/2) * np.mean(np.abs(dev))
            sigma_rms_simple = np.sqrt(np.mean(dev**2))
            return sigma_corr, sigma_meanabs, sigma_rms_simple
        sig_g_corr, sig_g_meanabs, sig_g_rms = est_sigma(clean_g, mu_g)
        sig_e_corr, sig_e_meanabs, sig_e_rms = est_sigma(clean_e, mu_e)
        dmu = abs(mu_e - mu_g)
        def snr(sig_g, sig_e):
            sbar = np.sqrt((sig_g**2 + sig_e**2)/2)
            return dmu/sbar, sbar
        snr_corr,_ = snr(sig_g_corr, sig_e_corr)
        snr_meanabs,_ = snr(sig_g_meanabs, sig_e_meanabs)
        snr_rmssimple,_ = snr(sig_g_rms, sig_e_rms)
        return dict(mu_g=mu_g, mu_e=mu_e, dmu=dmu,
          sig_g_corr=sig_g_corr, sig_e_corr=sig_e_corr,
          sig_g_meanabs=sig_g_meanabs, sig_e_meanabs=sig_e_meanabs,
          sig_g_rms=sig_g_rms, sig_e_rms=sig_e_rms,
          snr_corr=snr_corr, snr_meanabs=snr_meanabs, snr_rms=snr_rmssimple,
          n_g=len(clean_g), n_e=len(clean_e), flip=flip_local)
    stats = clean_sigma(Ig_r, Ie_r)
    stats_opt = clean_sigma(Ig_opt, Ie_opt)
    print(q+": stored_fid=%.4f angle=%.4f best_angle=%.4f" % (fid_stored, angle, best_angle))
    print("  computed fid stored_angle = %.4f T=%.1f flip=%s" % (fid_calc, T_opt, flip))
    print("  fid opt angle=%.4f" % fid_opt)
    print("  dmu=%.2f mu_g=%.1f mu_e=%.1f" % (stats['dmu'], stats['mu_g'], stats['mu_e']))
    print("  sig_g rms=%.2f sig_e rms=%.2f snr_rms=%.2f" % (stats['sig_g_rms'], stats['sig_e_rms'], stats['snr_rms']))
    print("  sig_g corr=%.2f sig_e corr=%.2f snr_corr=%.2f" % (stats['sig_g_corr'], stats['sig_e_corr'], stats['snr_corr']))
    print("  sig_g meanabs=%.2f sig_e meanabs=%.2f snr_meanabs=%.2f" % (stats['sig_g_meanabs'], stats['sig_e_meanabs'], stats['snr_meanabs']))
    print("  opt dmu=%.2f snr_opt_rms=%.2f snr_opt_corr=%.2f" % (stats_opt['dmu'], stats_opt['snr_rms'], stats_opt['snr_corr']))
    print()
