###############################################################################
# example.py
###############################################################################
# Written: Gabriel Collin (MIT) and Nick Rodd (Berkeley) - 28/08/19
###############################################################################
#
# Here we demonstrate how to use "posterior.h5" in order to extract the
# posterior and prior determined in the paper "A Search for Neutrino Point-
# Source Populations in 7 Years of IceCube Data with Neutrino-count Statistics"
#
# We show how to calculate the Bayes Factor and pointwise likelihood ratio
# for specific source-count functions. Further, we demonstrate how to reproduce
# Fig. 10 from the paper.
#
# We will use the Isotropic spatial template throughout, but the following
# templates are available:
#  - Isotropic
#  - Galactic_disk
#  - Fermi_bubble
#  - SFD_dust
#  - Northern_sky
#
###############################################################################


import numpy as np
import tables as tb
import scipy.stats


##################
# Load Posterior #
##################

# Below demonstrates how to load the posterior and prior for a template
# Here we consider the example of an isotropic template
h5_file = tb.open_file("posterior.h5")

posterior_samples = h5_file.root.Isotropic

# Extract source-count function parameters and stack into an array of samples 
# NB: the natural log (not base 10) of A and Fb are stored (also for the prior)
sample_array = np.vstack([ 
        posterior_samples.cols.ln_A, 
        posterior_samples.cols.ln_Fb, 
        posterior_samples.cols.n1, 
        posterior_samples.cols.n2
    ])

# Estimate the postrior from the samples via  kernel density estimation
posterior_estimate = scipy.stats.gaussian_kde(sample_array)

# Build up the prior cube as a product of the priors used in the model
prior = h5_file.root._v_attrs.prior_ln_A \
      * h5_file.root._v_attrs.prior_ln_Fb \
      * h5_file.root._v_attrs.prior_n1 \
      * h5_file.root._v_attrs.prior_n2


############
# BF and M #
############

# Define the Bayes Factor and pointwise likelihood ratio (M) which will be
# used to quantify various models

def compute_bayes_factor(params, post_estimate, param_prior):
    if isinstance(params, dict):
        params = [ params[k] for k in ['ln_A', 'ln_Fb', 'n1', 'n2'] ]
    # If p( A | d ) = p( d | A ) p(A) / p(d)
    # Then ln p( d | A ) = ln p(d) + ln p( A | d ) - ln p(A)
    #                    = ln (evidence conditioned on A) 
    #                    = ln (evidence) + ln (posterior) - ln (prior)
    ln_model_evidence = posterior_samples.attrs.NP_log_evidence \
                      + post_estimate.logpdf(params) - np.log(param_prior)
    ln_alternative_evidence = posterior_samples.attrs.P_log_evidence

    ln_Bayes_factor = ln_model_evidence - ln_alternative_evidence
    return ln_Bayes_factor


def compute_M_factor(params, post_estimate, param_prior, model_prior):
    if isinstance(params, dict):
        params = [ params[k] for k in ['ln_A', 'ln_Fb', 'n1', 'n2'] ]
    ln_model_evidence = posterior_samples.attrs.NP_log_evidence \
                      + post_estimate.logpdf(params) - np.log(param_prior)
    # Write ln( p(d | Poiss) p(Poiss) + p(d | non-Poiss) p (non-Poiss) ) in a
    # numerically stable manner
    ln_model_space = np.logaddexp(posterior_samples.attrs.P_log_evidence
                   + np.log(1-model_prior), 
                   posterior_samples.attrs.NP_log_evidence + np.log(model_prior))

    # ln M = ln p( d | A, non-Poiss ) + ln p(non-Poiss) 
    # - ln( p(d | Poiss) p(Poiss) + p(d | non-Poiss) p (non-Poiss) )
    ln_M_factor = ln_model_evidence + np.log(model_prior) - ln_model_space
    return ln_M_factor


# As an example of how to use these, we calculate the BF for two models
model_params_example  = dict(ln_A = np.log(1e16), ln_Fb = np.log(1e-12), 
                             n1 = 1.9, n2 = -2) 
model_params_example2 = dict(ln_A = np.log(1e17), ln_Fb = np.log(1e-13),
                             n1 = 1.9, n2 = -2) 

for mp in [model_params_example, model_params_example2]:
    print("Log Bayes Factor for model {} = {}".format(mp, 
          compute_bayes_factor(mp, posterior_estimate, prior)[0]))


#####################
# Reproduce Fig. 10 #
#####################

# As an extended example, we demonstrate how to reproduce Fig. 10 of the paper
# This plot shows contours of the pointwise likelihood ratio calculated for
# standard candle populations

# Define the figure bounds, and parameter space
Leff_lims = np.array([52, 55.5])
rho0_lims = np.array([-12, -5]) 
log10_Leff_points = np.linspace(*Leff_lims, num=100)
log10_rho0_points = np.linspace(*rho0_lims, num=100)

# Convert L_eff to F_b, and rho_0 to A using factor derived from FIRESONG
log10_Fb_points = log10_Leff_points + (-13 - 52)
log10_A_points = log10_rho0_points + (18 - -7)

# Switch to natural logarithms
ln_Fb_points = log10_Fb_points*np.log(10)
ln_A_points = log10_A_points*np.log(10)

# Setup a grid of coordinates
ln_A_meshpoints, ln_Fb_meshpoints = np.meshgrid(ln_A_points, ln_Fb_points)
log10_rho0_meshpoints, log10_Leff_meshpoints = np.meshgrid(log10_rho0_points, 
                                                           log10_Leff_points)
N_evalpoints = len(ln_A_meshpoints.ravel())

# Compute the M factor on this grid
posterior_evalpoints = np.vstack([ln_A_meshpoints.ravel(), 
                                  ln_Fb_meshpoints.ravel(), 
                                  1.9*np.ones(N_evalpoints), 
                                  -2*np.ones(N_evalpoints)])
ln_M_meshpoints = compute_M_factor(posterior_evalpoints, posterior_estimate, 
                                   prior, 0.5).reshape(ln_A_meshpoints.shape)

# Define the contour levels to show, and their labels.
cont_levels = [-2*np.log(10), -1*np.log(10), 1000]
cont_level_labels = [r"10^{-2}", r"10^{-1}", "max"]

# Load plotting details
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = 10/1.1, 8/1.1
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['legend.fontsize'] = 20
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Plot and fill the contours
cont = plt.contour(10**log10_Leff_meshpoints, 10**log10_rho0_meshpoints, 
                   ln_M_meshpoints, levels=cont_levels, colors=['k', 'w', 'w'])
contf = plt.contourf(10**log10_Leff_meshpoints, 10**log10_rho0_meshpoints, 
                     ln_M_meshpoints, levels=cont_levels, 
                     colors=['indigo', 'mediumseagreen', 'b'])

# Create legend
proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
         for pc in contf.collections]
legend_labels = [r"$\mathcal{M}(d; \boldsymbol{\phi}) > " + lbl + r"$" 
                 for lbl in cont_level_labels[:-1] ]
plt.legend(proxy, legend_labels, loc=1, fancybox=False)

# Overplot 4 electromagnetic standard candle parameters, taken from 1411.4385

# Standard candle parameters.
candle_points_rho0, candle_points_Leff = np.array([
    [1.000000000000004e-9, 1.0642092440647181e+55], # FSRQ
    [6.866488450043026e-8, 1e+54], # BL LAC
    [0.000001346640589965585, 1.545927736419466e+53], # Gal. Clusters
    [5.179474679231223e-8, 1.282649830528052e+52]]).T # FR II

# Place the standard candles as points on the plot
plt.scatter(candle_points_Leff[:-1], candle_points_rho0[:-1], marker="x",c='black',s=60)
plt.scatter(candle_points_Leff[-1], candle_points_rho0[-1], marker="x",c='white',s=60)
for i, txt in enumerate(["FSRQ", "BL LAC", "Galaxy Clusters", "FR-II"]):
    if txt == 'FSRQ':
        plt.annotate(txt, (candle_points_Leff[i]*0.9, candle_points_rho0[i]*0.35),fontsize=22)
    if txt == 'BL LAC':
        plt.annotate(txt, (candle_points_Leff[i]*0.9, candle_points_rho0[i]*0.35),fontsize=22)
    if txt == 'Galaxy Clusters':
        plt.annotate('Galaxy', (candle_points_Leff[i]*0.305, candle_points_rho0[i]*0.35),fontsize=22,color='white')
        plt.annotate('Clusters', (candle_points_Leff[i]*0.85, candle_points_rho0[i]*0.34),fontsize=22)
    if txt == 'FR-II':
        plt.annotate(txt, (candle_points_Leff[i]*0.9, candle_points_rho0[i]*0.35),fontsize=22,color='white')

plt.xscale('log')
plt.yscale('log')
plt.xlim(*10.0**Leff_lims)
plt.ylim(*10.0**rho0_lims)
plt.ylabel(r"$\rho_0$ (Mpc${}^{-3}$)")
plt.xlabel(r"$L_{\text{eff}}$ (erg yr${}^{-1}$)")
plt.tight_layout()
plt.savefig("standard_candle.pdf")
plt.close()

h5_file.close()
