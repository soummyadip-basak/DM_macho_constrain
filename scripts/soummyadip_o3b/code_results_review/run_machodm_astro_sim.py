#!/home1/soummyadip.basak/bilby/ve3/bilby_som/bin/python
##!/home/share/miniconda3/envs/igwn-py38/bin/python

import os, sys, string 
import numpy as np 

date = '2022-01-10'
outdir = '../../../sim_data/soummyadip_o3b/code_results_review/'
#mass_dist_pwl_index = -2.35
mass_model = 'power_law_peak'
approx = 'IMRPhenomD'
N_sim = int(1e5) 
y0 = 5 

## parameters to parallelize the run 
#zs_dist_vec = ['Belczynski', 'Dominik', 'uniform', 'O3cosmo', 'Madau-Dickinson SFR']
zs_dist = 'Madau-Dickinson'
f_upp_vec = [1024, 2048]
approx_vec = ['IMRPhenomD', 'IMRPhenomXPHM']
#psd_model_vec = ['H1L1_O1_psd', 'H1L1V1_O2_psd', 'H1L1V1_O3a_psd', 'H1L1V1_O3b_psd']
psd_model = 'H1L1V1_O3b_psd'

#f_upp, approx = np.meshgrid(f_upp_vec, approx_vec)
#f_upp, approx = np.ravel(f_upp)[1:3], np.ravel(approx)[1:3]

approx, f_upp = np.meshgrid(approx_vec, f_upp_vec)
approx, f_upp = np.ravel(approx)[:2], np.ravel(f_upp)[:2]

python = '/home1/soummyadip.basak/bilby/ve3/bilby_som/bin/python'     # Insert the proper python path of your use here and at the beginning

for arg in sys.argv[1:]:
    node = int(arg)

    if psd_model == 'H1L1_O1_psd':
        zs_max = 1.2
    elif psd_model == 'H1L1V1_O2_psd':
        zs_max = 1.5
    elif psd_model == 'H1L1V1_O3a_psd':
        zs_max = 1.5
    elif psd_model == 'H1L1V1_O3b_psd':
        zs_max = 1.5

    runtag = '%s_%s_plawidx_%s_zsmax_%.1f_y0_%.1f_%s_%s_Nsim_%.0e_fupp_%.0f' %(zs_dist, approx[node], mass_model, zs_max, y0,\
                                                                               psd_model, date, N_sim, f_upp[node])

    cmd = '%s machodm_astro_sim.py --psd_model %s --zs_dist %s --approx %s --f_upp %f --run_tag %s' %(python, psd_model, zs_dist,\
                                                                                                      approx[node], f_upp[node], runtag)
    
    print(cmd) 
    os.system(cmd) 
    print('The process is over.')
