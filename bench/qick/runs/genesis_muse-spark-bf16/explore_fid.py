import h5py, numpy as np
fpath="ss_ge.h5"

def get_arr(grp, name):
    raw=grp[name][0]
    txt=raw.decode() if isinstance(raw, bytes) else str(raw)
    arr=np.fromstring(txt[1:-1], sep=' ')
    return arr

def rotate(I,Q,theta):
    return I*np.cos(theta) - Q*np.sin(theta), I*np.sin(theta)+Q*np.cos(theta)

def fid_at_threshold(Ig, Ie, T):
    mu_g=np.median(Ig); mu_e=np.median(Ie)
    flip= mu_e < mu_g
    Ig_sorted=np.sort(Ig); Ie_sorted=np.sort(Ie)
    cdf_g = np.searchsorted(Ig_sorted, T, side='right')/len(Ig_sorted)
    cdf_e = np.searchsorted(Ie_sorted, T, side='right')/len(Ie_sorted)
    if not flip:
        fid = (cdf_g + (1 - cdf_e))/2
    else:
        fid = ((1 - cdf_g) + cdf_e)/2
    return fid

with h5py.File(fpath,'r') as f:
    for q in ['Q1','Q2','Q3','Q5','Q6']:
        grp=f[q]
        Ig=get_arr(grp,'I_g'); Qg=get_arr(grp,'Q_g')
        Ie=get_arr(grp,'I_e'); Qe=get_arr(grp,'Q_e')
        angle=float(grp['Angle'][0]); stored=float(grp['Fidelity'][0])
        Ig_r,Qg_r=rotate(Ig,Qg,angle)
        Ie_r,Qe_r=rotate(Ie,Qe,angle)
        mu_g=np.median(Ig_r); mu_e=np.median(Ie_r)
        mid=(mu_g+mu_e)/2
        # fid at median midpoint
        fid_mid=fid_at_threshold(Ig_r,Ie_r,mid)
        fid_zero=fid_at_threshold(Ig_r,Ie_r,0)
        # fid using only I without rotation
        fid_raw,_ ,_= (lambda: (lambda Igp,Iep: (lambda: __import__('functools').partial(lambda:0))())())()
        # compute raw I only
        def max_fid(Ig_p, Ie_p):
            Ig_sorted=np.sort(Igp:=Ig_p); Ie_sorted=np.sort(Ie_p)
            mu_g2=np.median(Ig_p); mu_e2=np.median(Ie_p)
            flip=mu_e2<mu_g2
            all_vals=np.sort(np.concatenate([Ig_p,Ie_p]))
            best=0; bestT=None
            for T in all_vals[::5]:
                cdf_g=np.searchsorted(Ig_sorted,T,side='right')/len(Ig_sorted)
                cdf_e=np.searchsorted(Ie_sorted,T,side='right')/len(Ie_sorted)
                fid=(cdf_g+(1-cdf_e))/2 if not flip else ((1-cdf_g)+cdf_e)/2
                if fid>best:
                    best=fid; bestT=T
            return best,bestT
        fid_rawI,_=max_fid(Ig, Ie)
        fid_rawQ,_=max_fid(Qg, Qe)
        # fid using 2D distance? Maybe they use 2D threshold (circular)?
        # Try fidelity using linear discriminant in 2D (project onto line between centroids)
        # That should be similar to rotation max, but let's compute via LDA
        # Compute centroids
        cent_g=np.array([np.mean(Ig), np.mean(Qg)])
        cent_e=np.array([np.mean(Ie), np.mean(Qe)])
        # Direction vector
        vec=cent_e-cent_g
        vec/=np.linalg.norm(vec)
        # Project all points onto vec
        proj_g = Ig*vec[0]+Qg*vec[1]
        proj_e = Ie*vec[0]+Qe*vec[1]
        fid_proj,_=max_fid(proj_g, proj_e)
        # What about fidelity defined as 1 - overlap integral of Gaussian fits using full sigma (not clean)?
        # Compute sigma full (std of all points) then compute theoretical fidelity via erfc
        def gauss_fid(Ig_p, Ie_p):
            mu_g=np.median(Ig_p); mu_e=np.median(Ie_p)
            sigma_g=np.std(Ig_p, ddof=1); sigma_e=np.std(Ie_p, ddof=1)
            dmu=abs(mu_e-mu_g)
            sbar=np.sqrt((sigma_g**2+sigma_e**2)/2)
            # theoretical optimal threshold for equal priors and Gaussian: between means weighted by sigma
            # approximate fidelity = 0.5*(1+erf(dmu/(2*sqrt2*sbar)))? For equal sigma, fidelity = 0.5+0.5*erf(dmu/(2*sqrt2*sbar))
            import math
            # for unequal sigma, more complex, but approximate using sbar
            fid_theory=0.5*(1+math.erf(dmu/(2*np.sqrt(2)*sbar)))
            return fid_theory, sigma_g, sigma_e, dmu, sbar
        fid_theory, sg, se, dmu, sbar = gauss_fid(Ig_r, Ie_r)
        # Also try using clean sigma but theoretical formula
        # clean sigma using rms simple (as before)
        def clean_sig(Ig_p, Ie_p):
            mu_g=np.median(Ig_p); mu_e=np.median(Ie_p)
            flip=mu_e<mu_g
            if not flip:
                cg=Ig_p[Ig_p<=mu_g]; ce=Ie_p[Ie_p>=mu_e]
            else:
                cg=Ig_p[Ig_p>=mu_g]; ce=Ie_p[Ie_p<=mu_e]
            sg=np.sqrt(np.mean((cg-mu_g)**2))
            se=np.sqrt(np.mean((ce-mu_e)**2))
            dmu=abs(mu_e-mu_g)
            sbar=np.sqrt((sg**2+se**2)/2)
            import math
            fid=0.5*(1+math.erf(dmu/(2*np.sqrt(2)*sbar)))
            return fid, sg, se, dmu, sbar
        fid_clean_theory, sg_c, se_c, dmu_c, sbar_c = clean_sig(Ig_r, Ie_r)

        # Try to brute find threshold that yields stored fidelity
        Ig_sorted=np.sort(Ig_r); Ie_sorted=np.sort(Ie_r)
        # search for T that gives stored fid
        # compute fid vs T curve
        thresholds=np.linspace(np.min(np.concatenate([Ig_r,Ie_r])), np.max(np.concatenate([Ig_r,Ie_r])), 1000)
        fids=[fid_at_threshold(Ig_r,Ie_r,T) for T in thresholds]
        # find T closest to stored
        idx=np.argmin(np.abs(np.array(fids)-stored))
        print(f"{q}: stored {stored:.4f} mid {fid_mid:.4f} zero {fid_zero:.4f} rawI {fid_rawI:.4f} rawQ {fid_rawQ:.4f} proj {fid_proj:.4f} theoryFull {fid_theory:.4f} theoryClean {fid_clean_theory:.4f}")
        print(f"   T_at_mid {mid:.1f} T_at_stored_approx {thresholds[idx]:.1f} fid_curve_max {max(fids):.4f}")
        print(f"   full sigma sg {sg:.1f} se {se:.1f} dmu {dmu:.1f} sbar {sbar:.1f} ; clean sg {sg_c:.1f} se {se_c:.1f} dmu_c {dmu_c:.1f} sbar_c {sbar_c:.1f}")
        # also check if stored corresponds to some quantile error? maybe they compute fidelity as P(g|g)*P(e|e) geometric?
        # Try product: fidelity_product = sqrt(P_gg * P_ee) at optimal T
        # At optimal T, P_gg = cdf_g, P_ee =1-cdf_e
        # At max_fid point, compute product
        # Find optimal T for max_fid, compute those
        best_fid=0; bestT=None; best_cdf_g=None; best_cdf_e=None
        for T in thresholds:
            cdf_g=np.searchsorted(Ig_sorted,T,side='right')/len(Ig_sorted)
            cdf_e=np.searchsorted(Ie_sorted,T,side='right')/len(Ie_sorted)
            mu_g_tmp=np.median(Ig_r); mu_e_tmp=np.median(Ie_r)
            flip_tmp=mu_e_tmp<mu_g_tmp
            fid_tmp=(cdf_g+(1-cdf_e))/2 if not flip_tmp else ((1-cdf_g)+cdf_e)/2
            if fid_tmp>best_fid:
                best_fid=fid_tmp; bestT=T; best_cdf_g=cdf_g; best_cdf_e=cdf_e
        print(f"   at max fid T {bestT:.1f} Pgg {best_cdf_g:.4f} Pee {1-best_cdf_e:.4f} product {np.sqrt(best_cdf_g*(1-best_cdf_e)):.4f}")
        print()
