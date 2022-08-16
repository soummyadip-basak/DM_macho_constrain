#!/home1/soummyadip.basak/bilby/ve3/bilby_som/bin/python
##!/home/share/miniconda3/envs/igwn-py38/bin/python

import os, sys
import numpy as np 

#mass_dist_pwl_index = -2.35
mass_model = 'power_law_peak'
approx = 'IMRPhenomD'
N_sim = int(1e7) 
y0 = 5 

date = '2022-01-10'
outdir = '../../sim_data/soummyadip_o3b/ff_data/'

## parameters to parallelize the run
psd_model = ['H1L1_O1_psd', 'H1L1V1_O2_psd', 'H1L1V1_O3a_psd', 'H1L1V1_O3b_psd'] 
zs_dist_vec = ['Belczynski', 'Dominik', 'uniform', 'O3cosmo', 'O3pop']
m_lens_vec = np.logspace(2, 5, 13)
f_dm_vec = [1, 0.8, 0.6, 0.4, 0.2]

psd_model, zs_dist, m_lens, f_dm = np.meshgrid(psd_model, zs_dist_vec, m_lens_vec, f_dm_vec)
psd_model, zs_dist, m_lens, f_dm = np.ravel(psd_model), np.ravel(zs_dist), np.ravel(m_lens), np.ravel(f_dm)

python = '/home1/soummyadip.basak/bilby/ve3/bilby_som/bin/python'   # Insert the proper python path of your use, here and at the beginning 

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

    runtag = '%s_%s_plawidx_%s_zsmax_%.1f_y0_%.1f_%s_%s_Nsim_%.0e_lw' %(zs_dist[node],
                                        approx, mass_model, zs_max, y0, psd_model[node], date, N_sim)
    simfname = '../../sim_data/soummyadip_o3b/astro_sim_data/machodm_lensing_astro_sim_%s' %(runtag)

    cmd = '%s machodm_calc_ff_lw.py --m_lens %e --f_dm %f --psd_model %s --sim_fname %s --run_tag %s --out_dir %s' %(python, m_lens[node], 
            f_dm[node], psd_model[node], simfname, runtag, outdir)
    
    print(cmd) 
    os.system(cmd) 
    print('The process is over.')
