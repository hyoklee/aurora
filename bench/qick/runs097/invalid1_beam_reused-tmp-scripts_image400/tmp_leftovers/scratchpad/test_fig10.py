import time, os
os.chdir('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/ice/runs/agent097_beam-1.0-fp8')
import sys; sys.path.insert(0,'.')
import cte_ice as ci
import numpy as np, scipy.stats

Z='20190904_Bayesian_posterior_for_IceCube_7-year_point-source_search_with_neutrino-count_statistics.zip'
base=f'ice/{Z}'
t0=time.time()
root_attrs=ci.read_attributes(f'{base}/posterior.h5')
table_tag=f'{base}/posterior.h5/Isotropic'
samples=ci.read_table(table_tag)
table_attrs=ci.read_attributes(table_tag)
example=ci.read_text(f'{base}/example.py')
readme=ci.read_text(f'{base}/readme.txt')
t1=time.time()
print('read time', t1-t0)
print('samples', samples.shape, samples.dtype)
print('root', root_attrs)
print('table attrs keys', list(table_attrs.keys()))
print('NP', table_attrs['NP_log_evidence'], 'P', table_attrs['P_log_evidence'])

sample_array=np.vstack([samples['ln_A'], samples['ln_Fb'], samples['n1'], samples['n2']])
post=scipy.stats.gaussian_kde(sample_array)
prior=root_attrs['prior_ln_A']*root_attrs['prior_ln_Fb']*root_attrs['prior_n1']*root_attrs['prior_n2']
print('prior', prior)

def compute_bayes_factor(params, post_estimate, param_prior):
    if isinstance(params, dict):
        params=[params[k] for k in ['ln_A','ln_Fb','n1','n2']]
    ln_model_evidence = table_attrs['NP_log_evidence'] + post_estimate.logpdf(params) - np.log(param_prior)
    ln_alternative = table_attrs['P_log_evidence']
    return ln_model_evidence - ln_alternative

mp1=dict(ln_A=np.log(1e16), ln_Fb=np.log(1e-12), n1=1.9, n2=-2)
mp2=dict(ln_A=np.log(1e17), ln_Fb=np.log(1e-13), n1=1.9, n2=-2)
for mp in [mp1,mp2]:
    print(mp, compute_bayes_factor(mp,post,prior)[0])
