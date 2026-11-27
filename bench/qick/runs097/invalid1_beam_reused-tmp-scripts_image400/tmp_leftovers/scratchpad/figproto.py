import time, os, sys
D='/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/ice/runs/agent097_beam-1.0-fp8'
os.chdir(D)
sys.path.insert(0,D)
import cte_ice as ci
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t0=time.time()
Z='20190904_Bayesian_posterior_for_IceCube_7-year_point-source_search_with_neutrino-count_statistics.zip'
base='ice/'+Z+'/posterior.h5'
ra=ci.read_attributes(base)
arr=ci.read_table(base+'/Isotropic')
ta=ci.read_attributes(base+'/Isotropic')
read_time=time.time()-t0
print('read_time', read_time, arr.shape)

posterior_samples=arr
sample_array=np.vstack([arr['ln_A'],arr['ln_Fb'],arr['n1'],arr['n2']])
posterior_estimate=scipy.stats.gaussian_kde(sample_array)
prior=ra['prior_ln_A']*ra['prior_ln_Fb']*ra['prior_n1']*ra['prior_n2']
NP_log_evidence=ta['NP_log_evidence']; P_log_evidence=ta['P_log_evidence']

def compute_bayes_factor(params, post_estimate, param_prior):
    if isinstance(params, dict):
        params=[params[k] for k in ['ln_A','ln_Fb','n1','n2']]
    ln_model_evidence=NP_log_evidence+post_estimate.logpdf(params)-np.log(param_prior)
    ln_alternative_evidence=P_log_evidence
    return ln_model_evidence-ln_alternative_evidence

def compute_M_factor(params, post_estimate, param_prior, model_prior):
    if isinstance(params, dict):
        params=[params[k] for k in ['ln_A','ln_Fb','n1','n2']]
    ln_model_evidence=NP_log_evidence+post_estimate.logpdf(params)-np.log(param_prior)
    ln_model_space=np.logaddexp(P_log_evidence+np.log(1-model_prior), NP_log_evidence+np.log(model_prior))
    return ln_model_evidence+np.log(model_prior)-ln_model_space

mp1=dict(ln_A=np.log(1e16),ln_Fb=np.log(1e-12),n1=1.9,n2=-2)
mp2=dict(ln_A=np.log(1e17),ln_Fb=np.log(1e-13),n1=1.9,n2=-2)
for mp in [mp1,mp2]:
    print('BF', compute_bayes_factor(mp,posterior_estimate,prior)[0])

Leff_lims=np.array([52,55.5]); rho0_lims=np.array([-12,-5])
log10_Leff_points=np.linspace(*Leff_lims,num=100)
log10_rho0_points=np.linspace(*rho0_lims,num=100)
log10_Fb_points=log10_Leff_points+(-13-52)
log10_A_points=log10_rho0_points+(18- -7)
ln_Fb_points=log10_Fb_points*np.log(10)
ln_A_points=log10_A_points*np.log(10)
ln_A_meshpoints, ln_Fb_meshpoints=np.meshgrid(ln_A_points, ln_Fb_points)
log10_rho0_meshpoints, log10_Leff_meshpoints=np.meshgrid(log10_rho0_points, log10_Leff_points)
N_evalpoints=len(ln_A_meshpoints.ravel())
posterior_evalpoints=np.vstack([ln_A_meshpoints.ravel(), ln_Fb_meshpoints.ravel(), 1.9*np.ones(N_evalpoints), -2*np.ones(N_evalpoints)])
ln_M_meshpoints=compute_M_factor(posterior_evalpoints, posterior_estimate, prior, 0.5).reshape(ln_A_meshpoints.shape)

cont_levels=[-2*np.log(10), -1*np.log(10), 1000]
cont_level_labels=[r"10^{-2}", r"10^{-1}", "max"]

mpl.rcParams? 
import matplotlib as mpl
mpl.rcParams['figure.figsize']=10/1.1,8/1.1
mpl.rcParams['xtick.labelsize']=22
mpl.rcParams['ytick.labelsize']=22
mpl.rcParams['axes.labelsize']=24
mpl.rcParams['legend.fontsize']=20
plt.rc('text', usetex=False)

cont=plt.contour(10**log10_Leff_meshpoints, 10**log10_rho0_meshpoints, ln_M_meshpoints, levels=cont_levels, colors=['k','w','w'])
contf=plt.contourf(10**log10_Leff_meshpoints, 10**log10_rho0_meshpoints, ln_M_meshpoints, levels=cont_levels, colors=['indigo','mediumseagreen','b'])
print('collections', len(contf.collections))
proxy=[plt.Rectangle((0,0),1,1,fc=pc.get_facecolor()[0]) for pc in contf.collections]
legend_labels=[r"$\mathcal{M}(d; \boldsymbol{\phi}) > " + lbl + r"$" for lbl in cont_level_labels[:-1]]
plt.legend(proxy, legend_labels, loc=1, fancybox=False)

candle_points_rho0, candle_points_Leff=np.array([
 [1.000000000000004e-9, 1.0642092440647181e+55],
 [6.866488450043026e-8, 1e+54],
 [0.000001346640589965585, 1.545927736419466e+53],
 [5.179474679231223e-8, 1.282649830528052e+52]]).T

plt.scatter(candle_points_Leff[:-1], candle_points_rho0[:-1], marker="x",c='black',s=60)
plt.scatter(candle_points_Leff[-1], candle_points_rho0[-1], marker="x",c='white',s=60)
for i, txt in enumerate(["FSRQ","BL LAC","Galaxy Clusters","FR-II"]):
    if txt=='FSRQ':
        plt.annotate(txt, (candle_points_Leff[i]*0.9, candle_points_rho0[i]*0.35),fontsize=22)
    if txt=='BL LAC':
        plt.annotate(txt, (candle_points_Leff[i]*0.9, candle_points_rho0[i]*0.35),fontsize=22)
    if txt=='Galaxy Clusters':
        plt.annotate('Galaxy', (candle_points_Leff[i]*0.305, candle_points_rho0[i]*0.35),fontsize=22,color='white')
        plt.annotate('Clusters', (candle_points_Leff[i]*0.85, candle_points_rho0[i]*0.34),fontsize=22)
    if txt=='FR-II':
        plt.annotate(txt, (candle_points_Leff[i]*0.9, candle_points_rho0[i]*0.35),fontsize=22,color='white')

plt.xscale('log'); plt.yscale('log')
plt.xlim(*10.0**Leff_lims); plt.ylim(*10.0**rho0_lims)
plt.ylabel(r"$\rho_0$ (Mpc${}^{-3}$)")
plt.xlabel(r"$L_{\text{eff}}$ (erg yr${}^{-1}$)")
plt.tight_layout()
out='/tmp/scratchpad/figproto.png'
plt.savefig(out)
plt.close()
print('saved', out)
