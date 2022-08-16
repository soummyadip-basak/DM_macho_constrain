#!/home1/soummyadip.basak/bilby/ve3/bilby_som/bin/python
##!/home/share/miniconda3/envs/igwn-py38/bin/python

import os, sys, string 
import numpy as np 

date = '2021-12-24'
outdir = '../../sim_data/Soummyadip_1/astro_sim_data/'
mass_dist_pwl_index = -2.35
approx = 'IMRPhenomD'
N_sim = int(1e7) 
y0 = 5 

# parameters to parallelize the run 
zs_dist_vec = ['Belczynski', 'Dominik', 'uniform']
psd_model_vec = ['H1L1_O1_psd', 'H1L1V1_O2_psd', 'H1L1V1_O3a_psd', 'H1L1V1_O3b_psd']

zs_dist, psd_model = np.meshgrid(zs_dist_vec, psd_model_vec)
zs_dist, psd_model = np.ravel(zs_dist), np.ravel(psd_model)

python = '/home1/soummyadip.basak/bilby/ve3/bilby_som/bin/python'

for arg in sys.argv[1:]:
    node = int(arg)

    if psd_model[node] == 'H1L1_O1_psd':
        zs_max = 1.2
    elif psd_model[node] == 'H1L1V1_O2_psd':
        zs_max = 1.5
    elif psd_model[node] == 'H1L1V1_O3a_psd':
        zs_max = 1.5
    elif psd_model[node] == 'H1L1V1_O3b_psd':
        zs_max = 1.5

    runtag = '%s_%s_plawidx_%.2f_zsmax_%.1f_y0_%.1f_%s_%s_Nsim_%.0e' %(zs_dist[node],
                                        approx, mass_dist_pwl_index, zs_max, y0, psd_model[node], date, N_sim)

    cmd = '%s machodm_astro_sim.py --psd_model %s --zs_dist %s --run_tag %s' %(python, psd_model[node], zs_dist[node], runtag)
    
    print(cmd) 
    os.system(cmd) 
    print('The process is over.')
