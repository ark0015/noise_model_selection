#!/usr/bin/env python
# coding: utf-8

# # Final Noise Analyses and Factorized Likelihood Run

import numpy as np
import glob, os, sys, json, string, pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging, inspect, copy
logging.basicConfig(level=logging.WARNING)

import enterprise
from enterprise.pulsar import Pulsar

current_path = os.getcwd()
splt_path = current_path.split("/")
# top_path_idx = splt_path.index("nanograv")
# top_path_idx = splt_path.index("akaiser")
top_path_idx = splt_path.index("ark0015")
top_dir = "/".join(splt_path[0 : top_path_idx + 1])
e_e_path = top_dir + "/enterprise_extensions/"
sys.path.insert(0, e_e_path)

import enterprise_extensions
from enterprise_extensions import models, model_utils, blocks
from enterprise_extensions.models import model_singlepsr_noise
from enterprise_extensions.chromatic import solar_wind, chromatic
from enterprise_extensions.hypermodel import HyperModel
from enterprise_extensions.sampler import setup_sampler


# # Free Spectral Red-noise model selection on 12.5yr Dataset

# ## Load Pickle File 

# In[4]:


psrname = 'J2043+1711'#'B1855+09'#'J1911+1347'
filepath = './no_dmx_pickles/'
filepath += '{0}_ng12p5yr_v3_nodmx_ePSR.pkl'.format(psrname)
with open(filepath,'rb') as fin:
    psr=pickle.load(fin)


# # Testing models with GP DM variations
# 
# __Very Import:__ What follows is an __example__ of noise model selection. For *most* pulsars the choice of noise models used in any given model selection analysis will be different than the ones chosen here. Those working on pulsars highlighted in the 11-year noise model analysis should include those models in their analyses and also use the best combination of models from that work in any final model selection that is done.

# ## Setup GP model selection
# 
# Load the pulsar's favored model

writeHotChains = True
print('Parallel Tempering?',writeHotChains)
model_kwargs_path = f'./chains/{psrname}/round_free_spectrum_run_model_B_round_5_powerlaw_psd_no_chrom_gp_periodic_dm_nondiag_k_2_positive_cusps_4_2_indx/model_kwargs.json'

with open(model_kwargs_path, 'r') as fin:
    model_kwargs = json.load(fin)
  


# In[9]:

"""
#Here I'm assumed that the model we want is the third one.
fs_kwargs = copy.deepcopy(model_kwargs)
fs_kwargs.update({'psd':'spectrum'})


# In[10]:


pta = model_singlepsr_noise(psr, **fs_kwargs)
"""

# ### NOTE: Should use an empirical distribution made from a prior run of this model!!!

# In[ ]:


#emp_dist_path = './wn_emp_dists/J1911+1347_ng12p5yr_v3_std_plaw_emp_dist.pkl'
#emp_dist_path = './twoD_distr_round_6_model_C.pkl'
#emp_dist_path = f'./{psrname}_oneD_distr_round_3_model_B.pkl'
emp_dist_path = f'./{psrname}_oneD_distr_round_5_model_B.pkl'

print("Empirical Distribution?",os.path.isfile(emp_dist_path))

outdir = f'./chains/{psrname}/{psrname}_factorized_like_run/'
"""
sampler = setup_sampler(pta, outdir=outdir,
                        empirical_distr=emp_dist_path)


# In[20]:


tspan = model_utils.get_tspan([psr])
achrom_freqs = np.linspace(1/tspan,30/tspan,30)
np.savetxt(sampler.outDir + 'achrom_rn_freqs.txt', achrom_freqs, fmt='%.18e')


# In[21]:


with open(sampler.outDir+'/model_kwargs.json' , 'w') as fout:
    json.dump(fs_kwargs, fout, sort_keys=True,
              indent=4, separators=(',', ': '))


# In[11]:


N = 500_000
x0 = np.hstack(p.sample() for p in pta_crn.params)
Sampler.sample(x0, , SCAMweight=30, AMweight=15,
               DEweight=30, burn=10000)

"""
# ## Factorized Likelihood Run 

# Here we substitute in the kwargs needed for a factorized likelihood analysis. Notice that the time span used here is the time span of the full data set. This ensures that the frequencies used in the red noise model and the "GWB" model are the same. The number of components is set to 5 to replicate the factorized likelihood runs from the 12.5 year analysis. 

# In[14]:


fLike_kwargs = copy.deepcopy(model_kwargs)
Tspan = 407576851.48121357
print(Tspan/(365.25*24*3600),' yrs')
fLike_kwargs.update({'factorized_like':True,
                     'Tspan':Tspan,
                     'fact_like_gamma':13./3,
                     'gw_components':5,'psd':'powerlaw'})


# In[15]:


pta_fL = model_singlepsr_noise(psr, **fLike_kwargs)


# In[22]:


sampler = setup_sampler(pta_fL, 
                        outdir=outdir,
                        empirical_distr=emp_dist_path)


# In[23]:


achrom_freqs_fL = np.linspace(1/Tspan,10/Tspan,10)
np.savetxt(sampler.outDir + 'achrom_rn_freqs.txt', achrom_freqs_fL, fmt='%.18e')


# In[24]:


with open(sampler.outDir+'/model_kwargs.json' , 'w') as fout:
    json.dump(fLike_kwargs, fout, sort_keys=True,
              indent=4, separators=(',', ': '))


# In[ ]:


N = 1000000
x0 = np.hstack(p.sample() for p in pta_fL.params)
sampler.sample(x0,N, SCAMweight=30, AMweight=15,
               DEweight=30, burn=10000,writeHotChains=writeHotChains,hotChain=False,)

