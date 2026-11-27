import time, os
os.chdir('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/ice/runs/agent097_beam-1.0-fp8')
import sys; sys.path.insert(0,'.')
source_env = None
import cte_ice as ci
import numpy as np
import scipy.stats

t0 = time.time()
root = 'ice/20190904_Bayesian_posterior_for_IceCube_7-year_point-source_search_with_neutrino-count_statistics.zip/posterior.h5'
root_attrs = ci.read_attributes(root)
table_tag = root + '/Isotropic'
tab_attrs = ci.read_attributes(table_tag)
post = ci.read_table(table_tag)
read_time = time.time()-t0
print('read_time', read_time)

sample_array = np.vstack([post['ln_A'], post['ln_Fb'], post['n1'], post['n2']])
print('sample shape', sample_array.shape)
t1=time.time()
post_est = scipy.stats.gaussian_kde(sample_array)
print('kde build time', time.time()-t1)

prior = root_attrs['prior_ln_A']*root_attrs['prior_ln_Fb']*root_attrs['prior_n1']*root_attrs['prior_n2']
print('prior', prior)

NP_log_evidence = tab_attrs['NP_log_evidence']
P_log_evidence = tab_attrs['P_log_evidence']

def compute_bayes_factor(params):
    params = [params['ln_A'], params['ln_Fb'], params['n1'], params['n2']]
    ln_model_evidence = NP_log_evidence + post_est.logpdf(params) - np.log(prior)
    ln_alternative = P_log_evidence
    return (ln_model_evidence - ln_alternative)[0]

m1 = dict(ln_A=np.log(1e16), ln_Fb=np.log(1e-12), n1=1.9, n2=-2)
m2 = dict(ln_A=np.log(1e17), ln_Fb=np.log(1e-13), n1=1.9, n2=-2)
print('BF1', compute_bayes_factor(m1))
print('BF2', compute_bayes_factor(m2))
