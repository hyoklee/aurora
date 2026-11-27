import os, sys, time
os.chdir('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/ice/runs/agent097_beam-1.0-fp8')
sys.path.insert(0,'.')
import cte_ice as ci
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t0=time.time()
root='ice/20190904_Bayesian_posterior_for_IceCube_7-year_point-source_search_with_neutrino-count_statistics.zip/posterior.h5'
root_attrs=ci.read_attributes(root)
table_tag=root+'/Isotropic'
tab_attrs=ci.read_attributes(table_tag)
post=ci.read_table(table_tag)
print('read_time', time.time()-t0)

sample_array=np.vstack([post['ln_A'],post['ln_Fb'],post['n1'],post['n2']])
post_est=scipy.stats.gaussian_kde(sample_array)
prior=root_attrs['prior_ln_A']*root_attrs['prior_ln_Fb']*root_attrs['prior_n1']*root_attrs['prior_n2']

NP_log_evidence=tab_attrs['NP_log_evidence']
P_log_evidence=tab_attrs['P_log_evidence']

def compute_M_factor(params, model_prior=0.5):
    if isinstance(params, dict):
        params=[params[k] for k in ['ln_A','ln_Fb','n1','n2']]
    ln_model_evidence=NP_log_evidence+post_est.logpdf(params)-np.log(prior)
    ln_model_space=np.logaddexp(P_log_evidence+np.log(1-model_prior),
                                NP_log_evidence+np.log(model_prior))
    return (ln_model_evidence+np.log(model_prior)-ln_model_space)

Leff_lims=np.array([52,55.5])
rho0_lims=np.array([-12,-5])
log10_Leff_points=np.linspace(*Leff_lims,num=100)
log10_rho0_points=np.linspace(*rho0_lims,num=100)
log10_Fb_points=log10_Leff_points+(-13-52)
log10_A_points=log10_rho0_points+(18- -7)
ln_Fb_points=log10_Fb_points*np.log(10)
ln_A_points=log10_A_points*np.log(10)
ln_A_mesh,ln_Fb_mesh=np.meshgrid(ln_A_points,ln_Fb_points)
log10_rho0_mesh,log10_Leff_mesh=np.meshgrid(log10_rho0_points,log10_Leff_points)
N_eval=len(ln_A_mesh.ravel())
post_eval=np.vstack([ln_A_mesh.ravel(),ln_Fb_mesh.ravel(),1.9*np.ones(N_eval),-2*np.ones(N_eval)])
ln_M=compute_M_factor(post_eval).reshape(ln_A_mesh.shape)

cont_levels=[-2*np.log(10),-1*np.log(10),1000]
mpl_rc={'figure.figsize':(10/1.1,8/1.1)}
import matplotlib as mpl
mpl.rcParams['figure.figsize']=10/1.1,8/1.1
mpl.rcParams['xtick.labelsize']=22
mpl.rcParams['ytick.labelsize']=22
mpl.rcParams['axes.labelsize']=24
mpl.rcParams['legend.fontsize']=20
plt.rc('text', usetex=False)

cont=plt.contour(10**log10_Leff_mesh,10**log10_rho0_mesh,ln_M,levels=cont_levels,colors=['k','w','w'])
contf=plt.contourf(10**log10_Leff_mesh,10**log10_rho0_mesh,ln_M,levels=cont_levels,colors=['indigo','mediumseagreen','b'])
proxy=[plt.Rectangle((0,0),1,1,fc=pc.get_facecolor()[0]) for pc in contf.collections]
legend_labels=[r"$\mathcal{M}(d; \boldsymbol{\phi}) > $"+lbl for lbl in [r"10^{-2}",r"10^{-1}"]]
plt.legend(proxy,legend_labels,loc=1,fancybox=False)

candle=np.array([
 [1.000000000000004e-9,1.0642092440647181e+55],
 [6.866488450043026e-8,1e+54],
 [0.000001346640589965585,1.545927736419466e+53],
 [5.179474679231223e-8,1.282649830528052e+52]]).T
plt.scatter(candle[1][:-1],candle[0][:-1],marker='x',c='black',s=60)
plt.scatter(candle[1][-1],candle[0][-1],marker='x',c='white',s=60)
for i,txt in enumerate(['FSRQ','BL LAC','Galaxy Clusters','FR-II']):
    if txt=='FSRQ':
        plt.annotate(txt,(candle[1][i]*0.9,candle[0][i]*0.35),fontsize=22)
    if txt=='BL LAC':
        plt.annotate(txt,(candle[1][i]*0.9,candle[0][i]*0.35),fontsize=22)
    if txt=='Galaxy Clusters':
        plt.annotate('Galaxy',(candle[1][i]*0.305,candle[0][i]*0.35),fontsize=22,color='white')
        plt.annotate('Clusters',(candle[1][i]*0.85,candle[0][i]*0.34),fontsize=22)
    if txt=='FR-II':
        plt.annotate(txt,(candle[1][i]*0.9,candle[0][i]*0.35),fontsize=22,color='white')

plt.xscale('log'); plt.yscale('log')
plt.xlim(*10.0**Leff_lims); plt.ylim(*10.0**rho0_lims)
plt.ylabel(r"$\rho_0$ (Mpc${}^{-3}$)")
plt.xlabel(r"$L_{\text{eff}}$ (erg yr${}^{-1}$)")
plt.tight_layout()
plt.savefig('/tmp/scratchpad/fig10_test.png',dpi=150)
print('saved')
