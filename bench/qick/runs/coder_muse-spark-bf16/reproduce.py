#!/usr/bin/env python
import h5py, numpy as np, matplotlib.pyplot as plt

path="ss_ge.h5"
plt.rcParams['font.size']=9

def parse_arr(s_bytes):
    raw=s_bytes[0]
    if isinstance(raw, bytes):
        raw=raw.decode()
    txt=raw.strip()
    if txt.startswith('['): txt=txt[1:]
    if txt.endswith(']'): txt=txt[:-1]
    return np.fromstring(txt, sep=' ')

def get_data(f,q):
    I_g=parse_arr(f[q]['I_g'][()])
    Q_g=parse_arr(f[q]['Q_g'][()])
    I_e=parse_arr(f[q]['I_e'][()])
    Q_e=parse_arr(f[q]['Q_e'][()])
    angle=float(f[q]['Angle'][0])
    fid_stored=float(f[q]['Fidelity'][0])
    return I_g,Q_g,I_e,Q_e,angle,fid_stored

def rotate(I,Q,ang):
    c=np.cos(ang); s=np.sin(ang)
    return I*c - Q*s, I*s + Q*c

# Helper to compute fidelity via KS and average
def compute_fidelity(Ig, Ie):
    Ig_sorted=np.sort(Ig); Ie_sorted=np.sort(Ie)
    all_vals=np.sort(np.concatenate([Ig,Ie]))
    cdf_g=np.searchsorted(Ig_sorted, all_vals, side='right')/len(Ig_sorted)
    cdf_e=np.searchsorted(Ie_sorted, all_vals, side='right')/len(Ie_sorted)
    diff=cdf_g - cdf_e
    ks_pos=np.max(diff)
    ks_neg=np.max(-diff)
    ks=max(ks_pos, ks_neg)
    # if g left (median g < median e), ks_pos is relevant
    mu_g=np.median(Ig); mu_e=np.median(Ie)
    if mu_g < mu_e:
        ks_main=ks_pos
        t_opt=all_vals[np.argmax(diff)]
    else:
        ks_main=ks_neg
        t_opt=all_vals[np.argmax(-diff)]
    F_ks=ks_main  # stored convention
    F_avg=0.5+ks_main/2  # paper convention
    return F_ks, F_avg, ks, t_opt, all_vals, cdf_g, cdf_e, diff

def sigma_mirrored(I, mu, outer_left=True):
    if outer_left:
        clean=I[I<=mu]
    else:
        clean=I[I>=mu]
    # mirror to get symmetric estimate
    mirrored=np.concatenate([clean, 2*mu - clean])
    sigma=np.std(mirrored, ddof=1)
    return sigma, clean, mirrored

# Choose best qubit: Q1 has highest fidelity
best_q='Q1'
with h5py.File(path,'r') as f:
    # compute for all qubits to find best
    results={}
    for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
        I_g,Q_g,I_e,Q_e,angle,fid_stored=get_data(f,q)
        I_gr,Q_gr=rotate(I_g,Q_g,angle)
        I_er,Q_er=rotate(I_e,Q_e,angle)
        F_ks,F_avg,ks,t_opt,_,_,_,_=compute_fidelity(I_gr,I_er)
        results[q]=(F_avg,F_ks,fid_stored,angle)
        print(f"{q}: F_avg={F_avg:.5f} F_ks={F_ks:.5f} stored={fid_stored:.5f} angle={angle:.3f}")
    # select Q1
    q=best_q
    I_g,Q_g,I_e,Q_e,angle,fid_stored=get_data(f,q)
    I_gr,Q_gr=rotate(I_g,Q_g,angle)
    I_er,Q_er=rotate(I_e,Q_e,angle)

mu_g=np.median(I_gr); mu_e=np.median(I_er)
muQ_g=np.median(Q_gr); muQ_e=np.median(Q_er)
delta_mu=abs(mu_e-mu_g)
print(f"\nChosen {q} mu_g={mu_g:.2f} mu_e={mu_e:.2f} delta={delta_mu:.2f} muQ_g={muQ_g:.2f} muQ_e={muQ_e:.2f}")

# clean-side sigma via mirrored method
sigma_g, clean_g, mirrored_g = sigma_mirrored(I_gr, mu_g, outer_left=(mu_g < mu_e))
sigma_e, clean_e, mirrored_e = sigma_mirrored(I_er, mu_e, outer_left=False if mu_g < mu_e else True)
# Actually for e, outer is right if g left
outer_e_is_right = (mu_g < mu_e)
sigma_e2, clean_e2, mirrored_e2 = sigma_mirrored(I_er, mu_e, outer_left=not outer_e_is_right)
# Use correct
sigma_g, sigma_e = sigma_g, sigma_e2
sigma_bar=np.sqrt((sigma_g**2+sigma_e**2)/2)
SNR=delta_mu/sigma_bar
print(f"sigma_g={sigma_g:.2f} sigma_e={sigma_e:.2f} sigma_bar={sigma_bar:.2f} SNR={SNR:.2f}")

# Uncertainty estimation per paper eq 4-6
Ng=3000; Ne=3000; Nclean_g=len(clean_g); Nclean_e=len(clean_e2)
dmu_g=sigma_g*np.sqrt(np.pi/(2*Ng))
dmu_e=sigma_e*np.sqrt(np.pi/(2*Ne))
d_delta=np.sqrt(dmu_g**2+dmu_e**2)
d_sigma_g=sigma_g/np.sqrt(2*Nclean_g)
d_sigma_e=sigma_e/np.sqrt(2*Nclean_e)
# sigma_bar uncertainty via propagation: sigma_bar = sqrt((sg^2+se^2)/2)
# d sigma_bar = sqrt( (sg*dsg)^2 + (se*dse)^2 ) / (2*sigma_bar) ??? Let's derive: sigma_bar^2 = (sg^2+se^2)/2 => 2*sigma_bar*d_sigma_bar = (sg*dsg + se*dse)/? Actually differentiate: d(sigma_bar^2)=2 sigma_bar d_sigma_bar = (sg dsg + se dse) ??? Wait: derivative w.r.t sg: d(sigma_bar^2)/dsg = sg, so d(sigma_bar^2)= sg dsg + se dse over? Let's do proper: sigma_bar = ((sg^2+se^2)/2)^0.5
# d sigma_bar = (sg dsg + se dse) / (2 sigma_bar)
d_sigma_bar=np.sqrt((sigma_g*d_sigma_g)**2 + (sigma_e*d_sigma_e)**2) / (2*sigma_bar) *2  # actually need propagation: variance sum: (partial w.r.t sg * dsg)^2 + ...
# partial = sg/(2 sigma_bar) *2? Let's compute correctly: sigma_bar = sqrt((sg^2+se^2)/2) => d sigma_bar / d sg = sg/(2 sigma_bar)
partial_g=sigma_g/(2*sigma_bar)
partial_e=sigma_e/(2*sigma_bar)
# Wait factor 2? Check: sigma_bar^2 = (sg^2+se^2)/2 => 2 sigma_bar d sigma_bar = (2 sg dsg +2 se dse)/2 = sg dsg + se dse => d sigma_bar = (sg dsg + se dse)/(2 sigma_bar)
# But for error propagation with independent, variance = (sg/(2 sigma_bar) * dsg)^2 + (se/(2 sigma_bar) * dse)^2
d_sigma_bar=np.sqrt((sigma_g/(2*sigma_bar)*d_sigma_g)**2 + (sigma_e/(2*sigma_bar)*d_sigma_e)**2)
d_SNR=SNR*np.sqrt((d_delta/delta_mu)**2 + (d_sigma_bar/sigma_bar)**2)
print(f"d_delta={d_delta:.2f} d_sigma_bar={d_sigma_bar:.2f} d_SNR={d_SNR:.2f}")

F_ks,F_avg,ks,t_opt,all_vals,cdf_g,cdf_e,diff = compute_fidelity(I_gr,I_er)
print(f"Fidelity KS (stored style)={F_ks:.5f} F_avg={F_avg:.5f} threshold={t_opt:.2f}")
# Binomial uncertainty
Ntotal=6000
dF=np.sqrt(F_avg*(1-F_avg)/Ntotal)
dF_ks=np.sqrt(F_ks*(1-F_ks)/Ntotal) if 0<F_ks<1 else 0
print(f"dF_avg={dF:.4f} dF_ks={dF_ks:.4f} stored={fid_stored:.5f}")

# Check relation: F_ks vs stored
print(f"Relation: stored {fid_stored:.5f} vs F_ks {F_ks:.5f} diff {abs(fid_stored-F_ks):.5f} ; F_avg {F_avg:.5f} = (stored+1)/2? {(fid_stored+1)/2:.5f}")

# --- Plotting Fig14-like ---
fig, axes = plt.subplots(1,2, figsize=(9,4), gridspec_kw={'width_ratios':[1,1]})
ax0=axes[0]
# scatter with alpha
ax0.scatter(I_gr, Q_gr, s=6, alpha=0.35, color='#1f77b4', label='|g⟩ (N=3000)', rasterized=True)
ax0.scatter(I_er, Q_er, s=6, alpha=0.35, color='#d62728', label='|e⟩ (N=3000)', rasterized=True)
ax0.scatter([mu_g],[muQ_g], color='black', marker='x', s=80, linewidths=2, label='median g')
ax0.scatter([mu_e],[muQ_e], color='black', marker='o', s=40, facecolors='none', edgecolors='black', linewidths=1.5, label='median e')
ax0.axvline(t_opt, color='gray', linestyle='--', linewidth=1, label=f'threshold {t_opt:.1f}')
ax0.set_xlabel('I (a.u.) rotated')
ax0.set_ylabel('Q (a.u.) rotated')
ax0.set_title('(a) Rotated I–Q scatter (Q1 / qubit 0, 4195 MHz)\n3000 shots per state, angle %.3f rad' % angle)
ax0.legend(markerscale=1.5, fontsize=7, loc='upper left')
ax0.set_aspect('equal', adjustable='datalim')
# add grid
ax0.grid(alpha=0.3)

ax1=axes[1]
# histogram projected onto I
bins=80
# choose range covering both
combined=np.concatenate([I_gr,I_er])
rng=(combined.min()-5, combined.max()+5)
hist_g, edges=np.histogram(I_gr, bins=bins, range=rng)
hist_e, _=np.histogram(I_er, bins=bins, range=rng)
centers=0.5*(edges[:-1]+edges[1:])
width=edges[1]-edges[0]
ax1.bar(centers, hist_g, width=width*0.9, color='#1f77b4', alpha=0.5, label='|g⟩', edgecolor='none')
ax1.bar(centers, hist_e, width=width*0.9, color='#d62728', alpha=0.5, label='|e⟩', edgecolor='none')
# overlay Gaussian core fits from clean side: use mirrored sigma and mu, amplitude scaled to match histogram peak
def gauss(x, mu, sigma, N, binwidth):
    return N * binwidth / (sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
# For g, use N=3000, for e same, but clean-side fit should be half? Actually paper fits only outer side, dashed curves show Gaussian.
# We'll plot full Gaussian with peak corresponding to full N, but note it's fitted to outer side.
xg=np.linspace(rng[0], rng[1], 400)
yg=gauss(xg, mu_g, sigma_g, 3000, width)
ye=gauss(xg, mu_e, sigma_e, 3000, width)
ax1.plot(xg, yg, color='black', linestyle='--', linewidth=1.5, label=f'g fit σ={sigma_g:.1f}')
ax1.plot(xg, ye, color='black', linestyle=':', linewidth=1.5, label=f'e fit σ={sigma_e:.1f}')
ax1.axvline(t_opt, color='gray', linestyle='--', linewidth=1)
ax1.set_xlabel('I (a.u.) rotated')
ax1.set_ylabel('Counts per bin')
ax1.set_title('(b) Projected I-histogram\nΔµ=%.1f±%.1f, σ̄=%.1f±%.1f, SNR=%.1f±%.1f, F=%.1f±%.1f%% (KS=%.1f%%)' % (delta_mu, d_delta, sigma_bar, d_sigma_bar, SNR, d_SNR, F_avg*100, dF*100, F_ks*100))
ax1.legend(fontsize=7)
ax1.grid(alpha=0.3)

fig.suptitle('QICK Box single-shot readout reproduction — ss_ge.h5 Q1 (qubit 0)\nPaper Fig.14 reports Δµ=28, σg=6.4, σe=6.5, SNR=4.4, F=91.7% (a.u. scale differs)', fontsize=10)
plt.tight_layout(rect=[0,0,1,0.93])
plt.savefig('readout.png', dpi=300)
print("Saved readout.png")

# Also write summary txt
with open('recompute_summary.txt','w') as out:
    out.write(f"Q1 F_ks={F_ks:.5f} F_avg={F_avg:.5f} stored={fid_stored:.5f}\n")
    out.write(f"delta={delta_mu:.2f} sigma_g={sigma_g:.2f} sigma_e={sigma_e:.2f} SNR={SNR:.2f}\n")
    out.write(f"dF={dF:.4f} dSNR={d_SNR:.2f}\n")
