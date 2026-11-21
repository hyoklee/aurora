import h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fpath="ss_ge.h5"

def get_arr(grp, name):
    raw=grp[name][0]
    txt=raw.decode() if isinstance(raw, bytes) else str(raw)
    return np.fromstring(txt[1:-1], sep=' ')

# Choose Q1 as best qubit matching paper's 91.7%
qname="Q1"
with h5py.File(fpath,'r') as f:
    grp=f[qname]
    Ig=get_arr(grp,'I_g'); Qg=get_arr(grp,'Q_g')
    Ie=get_arr(grp,'I_e'); Qe=get_arr(grp,'Q_e')
    angle=float(grp['Angle'][0])
    stored=float(grp['Fidelity'][0])

print(f"Q1 stored angle {angle} stored fidelity {stored}")

def rotate(I,Q,theta):
    return I*np.cos(theta)-Q*np.sin(theta), I*np.sin(theta)+Q*np.cos(theta)

Ig_r, Qg_r = rotate(Ig,Qg,angle)
Ie_r, Qe_r = rotate(Ie,Qe,angle)

mu_g = np.median(Ig_r)
mu_e = np.median(Ie_r)
median_q_g = np.median(Qg_r)
median_q_e = np.median(Qe_r)
dmu = abs(mu_e - mu_g)

# Clean-side sigma estimation (RMS about median, paper's method)
# Paper: samples I <= mu_g for g, I >= mu_e for e, Gaussian core fits
clean_g = Ig_r[Ig_r <= mu_g] if mu_e > mu_g else Ig_r[Ig_r >= mu_g]
clean_e = Ie_r[Ie_r >= mu_e] if mu_e > mu_g else Ie_r[Ie_r <= mu_e]

# RMS estimator: sigma = sqrt(mean((I - mu)^2)) for clean side (half-Gaussian second moment = sigma^2)
sig_g = np.sqrt(np.mean((clean_g - mu_g)**2))
sig_e = np.sqrt(np.mean((clean_e - mu_e)**2))
sbar = np.sqrt((sig_g**2 + sig_e**2)/2)
snr = dmu / sbar
print(f"mu_g {mu_g:.2f} mu_e {mu_e:.2f} dmu {dmu:.2f} sig_g {sig_g:.2f} sig_e {sig_e:.2f} sbar {sbar:.2f} SNR {snr:.2f}")

# Fidelity via max cumulative histogram contrast
Ig_sorted = np.sort(Ig_r)
Ie_sorted = np.sort(Ie_r)
all_vals = np.sort(np.concatenate([Ig_r, Ie_r]))
thresholds = all_vals[::2]  # sample
best_fid=0; best_T=None
for T in thresholds:
    cdf_g = np.searchsorted(Ig_sorted, T, side='right')/len(Ig_sorted)
    cdf_e = np.searchsorted(Ie_sorted, T, side='right')/len(Ie_sorted)
    fid = (cdf_g + (1 - cdf_e))/2 if mu_e>mu_g else ((1-cdf_g)+cdf_e)/2
    if fid>best_fid:
        best_fid=fid; best_T=T
print(f"Best threshold {best_T:.2f} fidelity {best_fid:.4f} stored {stored:.4f}")
# Binomial uncertainty
Ntotal=6000
deltaF = np.sqrt(best_fid*(1-best_fid)/Ntotal)  # Actually paper says F(1-F)/N but likely sqrt? We'll compute both
deltaF2 = best_fid*(1-best_fid)/Ntotal
print(f"deltaF sqrt {deltaF:.4f} linear {deltaF2:.5f}")

# For plot, need histogram along I
fig, axes = plt.subplots(1,2, figsize=(12,5), gridspec_kw={'width_ratios':[1.2,1]})
ax0=axes[0]
ax0.scatter(Ig_r, Qg_r, s=8, alpha=0.5, c='tab:blue', label='|g> (3000 shots)', edgecolors='none')
ax0.scatter(Ie_r, Qe_r, s=8, alpha=0.5, c='tab:red', label='|e> (3000 shots)', edgecolors='none')
ax0.scatter([mu_g], [median_q_g], s=120, c='black', marker='D', label='median', zorder=5, edgecolors='white')
ax0.scatter([mu_e], [median_q_e], s=120, c='black', marker='D', zorder=5, edgecolors='white')
ax0.axvline(best_T, color='k', linestyle='--', linewidth=1.2, alpha=0.7, label=f'threshold {best_T:.0f}')
ax0.set_xlabel('I (a.u.)  [rotated]')
ax0.set_ylabel('Q (a.u.)  [rotated]')
ax0.set_title(f'(a) Rotated I-Q scatter  Q1 (qubit 0)\nangle={angle:.3f} rad  SNR={snr:.1f}  $\\Delta\\mu$={dmu:.0f}')
ax0.legend(fontsize=8, loc='best')
ax0.grid(alpha=0.2)

ax1=axes[1]
# histogram bins
bins = np.linspace(min(all_vals)-10, max(all_vals)+10, 80)
counts_g, edges = np.histogram(Ig_r, bins=bins)
counts_e, _ = np.histogram(Ie_r, bins=bins)
centers = (edges[:-1]+edges[1:])/2
binw = edges[1]-edges[0]
width_frac=0.8
ax1.hist(Ig_r, bins=bins, alpha=0.6, color='tab:blue', label='|g>', density=False)
ax1.hist(Ie_r, bins=bins, alpha=0.6, color='tab:red', label='|e>', density=False)
# Gaussian core overlays: dashed curves scaled to histogram
xs = np.linspace(edges[0], edges[-1], 500)
def gauss(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)
# Scale: expected counts = N * binw * pdf
Ng=len(Ig_r); Ne=len(Ie_r)
pdf_g = gauss(xs, mu_g, sig_g)
pdf_e = gauss(xs, mu_e, sig_e)
# Scale to match histogram peak: multiply by N*binw
ax1.plot(xs, Ng*binw*pdf_g, color='blue', linestyle='--', linewidth=1.8, label=f'Gauss core $\\sigma_g$={sig_g:.1f}')
ax1.plot(xs, Ne*binw*pdf_e, color='red', linestyle='--', linewidth=1.8, label=f'Gauss core $\\sigma_e$={sig_e:.1f}')
ax1.axvline(best_T, color='k', linestyle='--', linewidth=1.2, alpha=0.7)
ax1.axvline(mu_g, color='blue', linestyle=':', alpha=0.8)
ax1.axvline(mu_e, color='red', linestyle=':', alpha=0.8)
ax1.set_xlabel('I (a.u.)  [projected]')
ax1.set_ylabel('Counts')
ax1.set_title(f'(b) Projected histogram  F={best_fid*100:.1f}%  $\\Delta\\mu$={dmu:.0f}')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.2)

fig.suptitle(f'Single-shot readout Q1 (qubit index 0)  3000 shots/state  Fidelity={best_fid*100:.1f}% (stored {stored*100:.1f}%)', fontsize=11)
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('readout.png', dpi=200)
print("Saved readout.png")
