#!/usr/bin/env python
# coding: utf-8

# # Noise model selection on NANOGrav pulsars

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
from enterprise_extensions import sampler


psrname = 'J2043+1711'#'B1855+09'#'J1911+1347'
filepath = './no_dmx_pickles/'
filepath += '{0}_ng12p5yr_v3_nodmx_ePSR.pkl'.format(psrname)
with open(filepath,'rb') as fin:
    psr=pickle.load(fin)


# # Testing models with GP DM variations
# 
# __Very Import:__ What follows is an __example__ of noise model selection. For *most* pulsars the choice of noise models used in any given model selection analysis will be different than the ones chosen here. Those working on pulsars highlighted in the 11-year noise model analysis should include those models in their analyses and also use the best combination of models from that work in any final model selection that is done.

# ## Setup GP model selection

# Choices for `psd`s (i.e. `diag` kernels):
# 	* `powerlaw`
# 	* `spectrum`
# 	* `tprocess`
# 	* `tprocess_adapt`
# 	
# Choices for `nondiag` kernels:
# 	* `sq_exp`
# 	* `periodic`
# 	* `sq_exp_rfband`
# 	* `periodic_rfband`
# 	
# Running a chromatic gp:
# 	* set `chrom_gp=True` or in the `kwarg` dictionary {`chrom_gp`:True}
# 	* set which general type of kernel you'll use `chrom_gp_kernel`= `['diag','nondiag']`
# 	* set the specific kernel, either `chrom_kernel='sq_exp'` or `chrom_psd='turnover'`, for instance
# 	* set the index, 1/f^idx, Default is 4, `chrom_idx=4.4`
# 
# ```
# dipcusp_kwargs = {'dm_expdip':True,
#                   'dmexp_sign': 'negative',
#                   'num_dmdips':2,
#                   'dm_cusp_idx':[2,4],
#                   'dm_expdip_tmin':[54700,57450],
#                   'dm_expdip_tmax':[54850,57560],
#                   'dmdip_seqname':'ism',
#                   'dm_cusp':False,
#                   'dm_cusp_sign':'negative',
#                   'dm_cusp_idx':[2,4],
#                   'dm_cusp_sym':False,
#                   'dm_cusp_tmin':None,
#                   'dm_cusp_tmax':None,
#                   'num_dm_cusps':2, 
#                   'dm_dual_cusp':True,
#                   'dm_dual_cusp_tmin':[54700,57450],
#                   'dm_dual_cusp_tmax':[54850,57560],}
# ```

# In[6]:


red_psd = 'spectrum'
#dm_nondiag_kernel = ['periodic','sq_exp','periodic_rfband','sq_exp_rfband']
#dm_nondiag_kernel = ['sq_exp', 'periodic']
dm_nondiag_kernel = 'periodic'
dm_sw_gp = False
dm_annual = False
#chrom_gps = [True,False]
chrom_gp = False
chrom_gp_kernel = 'nondiag'
#chrom_kernels = ['periodic','sq_exp']
chrom_kernel = 'periodic'
chrom_index = 4.
dm_cusp = True
cusp = 2
white_vary = True


# Use the inspect package to pull the arguments from `model_singlepsr_noise` and make a template for the keyword arguments (kwargs) dictionary we will be using to keep track of these various models. 

# In[7]:


args = inspect.getfullargspec(model_singlepsr_noise)
keys = args[0][1:]
vals = args[3]
model_template = dict(zip(keys,vals))
print(model_template)


# Here we show one work flow where we set up a `for` loop to go through the various models. Make sure to save the `model_labels` and `model_kwargs`. The former will be useful for making noise flower plots, while the latter will be the final product for a pulsar in this analysis.

# In[ ]:
kwargs = copy.deepcopy(model_template)

kwargs.update({'dm_var':True,
              'dmgp_kernel':'nondiag',
              'psd':red_psd,
              'white_vary':white_vary,
              'dm_nondiag_kernel':dm_nondiag_kernel,
              'dm_sw_deter':True,
              'dm_sw_gp':dm_sw_gp,
              'dm_annual': dm_annual,
              'swgp_basis': 'powerlaw',
              'chrom_gp_kernel':chrom_gp_kernel,
              'chrom_kernel':chrom_kernel,
              'chrom_gp':chrom_gp,
              'chrom_idx':chrom_index,
              'dm_cusp':dm_cusp,
              'num_dm_cusps':num_cusp,
              'dm_cusp_sign':list(np.repeat('vary',num_cusp))})

# Instantiate single pulsar noise model
pta = model_singlepsr_noise(psr, **kwargs)


print(kwargs)


# ## Set the out directory for you chains and other sampler setup
# ### !!! Important !!! Please set the chain directory outside of the git repository (easier) or at least do not try and commit your chains to the repo. 

# In[ ]:
round_number = 'free_spectrum_run_model_B_round_3_powerlaw_psd_no_chrom_gp_periodic_dm_nondiag_k_2_cusps'
writeHotChains = True
print('Parallel Tempering?',writeHotChains)
print(round_number)
outdir = './chains/{}/round_{}'.format(psr.name,round_number)
print('Will Save to: ',outdir)
#emp_distr_path = './wn_emp_dists/{0}_ng12p5yr_v3_std_plaw_emp_dist.pkl'.format(psr.name)
#emp_distr_path = f'./twoD_distr_round_6_model_C.pkl'
emp_distr_path = f'./{psrname}_twoD_distr_round_3_model_B.pkl'
print("Empirical Distribution?",os.path.isfile(emp_distr_path))
sampler = sampler.setup_sampler(pta,resume=True, outdir=outdir,
                                    empirical_distr=emp_distr_path)

with open(outdir+'/model_kwargs.json' , 'w') as fout:
    json.dump(kwargs, fout, sort_keys=True,
              indent=4, separators=(',', ': '))


# In[ ]:

#print(s)
# sampler for N steps
N = int(5e6)
x0 = np.hstack([p.sample() for p in pta.params])


# In[ ]:


sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, burn=100000,writeHotChains=writeHotChains,hotChain=False,)

