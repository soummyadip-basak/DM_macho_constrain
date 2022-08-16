#!/usr/bin/env python
# coding: utf-8

# # MACHO Lensing project: Calculate the wave optics lensing Bayes factor from simulated BBH events 
# 
# P. Ajith <ajith@icts.res.in>, June 2021 

## Preamble


from __future__ import division
import numpy as np
from cosmology_models import LCDM
import scipy
from  scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import math
import mpmath
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy import constants as const
import time 
import lal 
from scipy.optimize import minimize
import scipy.special as special
import pickle 
from math import isnan
from scipy.stats import truncnorm
import os, argparse

import pycbc.filter as filter
from pycbc import detector
from pycbc.waveform import get_td_waveform, get_fd_waveform
import pycbc.psd as psd

c = const.c       #speed of light (in m/s)
c_km = c/1000     #speed of light (in km/s)


rc_params = {'axes.labelsize': 18,
             'axes.titlesize': 18,
             'font.size': 18,
             'lines.linewidth' : 3,
             'legend.fontsize': 18,
             'xtick.labelsize': 16,
             'ytick.labelsize': 16,
             'text.usetex' : True,
            }
rcParams.update(rc_params)

from matplotlib import rc

rc('text.latex', preamble='\\usepackage{txfonts}')
rc('text', usetex=True)
rc('font', family='serif')
rc('font', serif='times')
rc('mathtext', default='sf')
rc("lines", markeredgewidth=1)
rc("lines", linewidth=2)


# cosmological parameters - use Planck 2013 values 
from astropy.cosmology import Planck13
H_0 = Planck13.H0.value        #Hubble's constant at present epoch (in km/(s*Mpc))
Odm0 = Planck13.Odm0           #Dark matter density at current epoch
lcdm = LCDM(Om0=Planck13.Om0)


# Fitting factor calculation 

##  Define functions to generate lensed waveforms  


def lens_ampl_function_wave_optics(f, M_lz, y_l, intrp_hyp1f1=None):
    """ frequency dependent lensing magnification """
        
    w = 8*np.pi*M_lz*lal.MTSUN_SI*f
    x_m = (y_l + np.sqrt(y_l**2 +4)) / 2.
    phi_m = (x_m - y_l)**2/2. - np.log(x_m)
    
    if intrp_hyp1f1==None:
        # evaluate the hypergeometric fun using mpmath package 
        hyp_fn = np.vectorize(mpmath.hyp1f1)(0.5*w*1j, 1., 0.5*w*y_l**2*1j, maxterms=1e6)
        hyp_fn = np.array(hyp_fn.tolist(),dtype=complex)  # convert to numpy array from mpmath format 
    else: 
        # evaluate the interpolated function 
        hyp_fn = 10**intrp_hyp1f1['log_abs'](w, y_l)*np.exp(1j*intrp_hyp1f1['arg'](w,y_l))
    
    return np.exp(np.pi*w/4.+0.5*w*1j*(np.log(w/2.)-2*phi_m))*special.gamma(np.ones(len(w))-0.5*w*1j)*hyp_fn

def lens_ampl_factor_geom_optics(f, M_lz, y): 
    """ frequency dependent lensing magnification (geometric optics limit) """
            
    w = 8*np.pi*M_lz*lal.MTSUN_SI*f
    # time delay 
    sqrt_ysqr_p4 = np.sqrt(y**2+4)
    delta_Td_hat = 0.5*y*sqrt_ysqr_p4 + np.log((sqrt_ysqr_p4+y)/(sqrt_ysqr_p4-y)) # this is dimensionless 

    # magnification 
    mu_p = 0.5+(y**2+2)/(2*y*sqrt_ysqr_p4)
    mu_m = 0.5-(y**2+2)/(2*y*sqrt_ysqr_p4)

    # magnification function 
    return np.sqrt(np.abs(mu_p)) - 1j*np.sqrt(np.abs(mu_m))*np.exp(1j*w*delta_Td_hat)


def lens_ampl_function(f, M_lz, y, intrp_hyp1f1=None):
    """ frequency dependent lensing magnification: hybrid using wave optics and geometric optics """
    
#    M_lz = np.power(10,x)
    w = 8*np.pi*M_lz*lal.MTSUN_SI*f
    
    # if y > 3 geometric optics provides a good approximation 
    if y > 3: 
        F_f = lens_ampl_factor_geom_optics(f, M_lz, y)

    # if y <= 3, use a combination of wave optics (low freq) and geom optics (high freq)    
    else:
        w_max = 150
        w_1 = w[np.where(w <= w_max)[0]]
        w_2 = w[np.where(w > w_max)[0]]
        f_1 = w_1/(8*np.pi*M_lz*lal.MTSUN_SI)
        f_2 = w_2/(8*np.pi*M_lz*lal.MTSUN_SI)
        
        if len(f_1) != 0:
            F_w = lens_ampl_function_wave_optics(f_1, M_lz, y, intrp_hyp1f1=None)
            F_g = lens_ampl_factor_geom_optics(f_2, M_lz, y)
            F_f = np.concatenate((F_w,F_g))
            
        else:
            F_f = lens_ampl_factor_geom_optics(f_2, M_lz, y)
            
    return F_f


def calc_chirptime(m1, m2, f_min=20.):
    m = float(m1+m2)
    eta = float(m1*m2/m**2)
    m_sec = m*lal.MTSUN_SI
    return (5./256.)*m_sec/((np.pi*m_sec*f_min)**(8./3.)*eta) + 1e4*m_sec


def get_lensed_fd_waveform(approx, m1, m2, dL, spin1z, spin2z, iota, phi0, df, f_low, f_upp, 
                           M_lz, y_l, intrp_hyp1f1=None): 
    
    # generate template waveform 
    hp_f, hc_f = get_fd_waveform(approximant=approx, mass1=m1, mass2=m2, distance=dL, spin1z=spin1z, 
            spin2z=spin2z, inclination=iota, coa_phase=phi0, delta_f=df, f_lower=f_low, f_final=f_upp)
    
    # calculate the lensing amplification  
    f = hp_f.get_sample_frequencies().data
    
    F_f = lens_ampl_function(f, M_lz, y_l, intrp_hyp1f1)
    F_f[np.isnan(F_f)] = 1

    return hp_f*F_f, hc_f*F_f


def calc_lensing_timedelay(Mlz, y): 
    """calculate the time delay between multiple images produced by a point mass lens 
    (GO approximation). Ref. Eq (18) of Takahashi&Nakamura ApJ 595:1039–1051, 2003"""
    
    ry2p4 = np.sqrt(y**2+4) 
    return 4*Mlz*lal.MTSUN_SI*(0.5*y*ry2p4 + np.log((ry2p4+y)/(ry2p4-y)))


##  Define match and fitting factor functions 

def calc_mismatch(h, approx, mchirp, eta, eff_spin, df, f_low, f_upp, Sh):
    """ return the log mismatch of the input waveform with a template waveform """
    
    mtot = mchirp * np.power(eta, -3./5)
    fac = np.sqrt(1. - 4.*eta)
    m1, m2 = (mtot * (1. + fac) / 2., mtot * (1. - fac) / 2.)
    spin1z, spin2z = eff_spin, eff_spin 
    

    if isnan(m1) or isnan(m2) or isnan(eff_spin) or m1 < 2. or m2 < 2 or m1/m2 < 1./18 or m1/m2 > 18 or m1+m2 > 800 or eff_spin < -0.99 or eff_spin > 0.99:
        log_mismatch = 1e6   
    else: 
        # generate template waveform 
        hp_templ, hc_templ = get_fd_waveform(approximant=approx, mass1=m1, mass2=m2, distance=dL, spin1z=spin1z, 
            spin2z=spin2z, inclination=iota, coa_phase=phi0, delta_f=df, f_lower=f_low, f_final=f_upp)
        
        # calculate the match 
        match, shift_idx =filter.matchedfilter.match(h, hp_templ, psd=Sh, low_frequency_cutoff=f_low, 
                                    high_frequency_cutoff=f_upp, v1_norm=None, v2_norm=None)
        log_mismatch = np.log10(1-match)
        
        f = hp_templ.get_sample_frequencies()
        fidx = np.logical_and(f >= f_low, f <= f_upp)
    
    return log_mismatch


def calc_mismatch_lensed_waveforms(approx, mchirp, eta, eff_spin, f_low, f_upp, Sh, f_Sh,  
                                   M_lz, y_l, intrp_hyp1f1=None):
        
    mtot = mchirp * np.power(eta, -3./5)
    fac = np.sqrt(1. - 4.*eta)
    m1, m2 = (mtot * (1. + fac) / 2., mtot * (1. - fac) / 2.)
    spin1z, spin2z = eff_spin, eff_spin 
    
    # check inputs  
    if m1 < 2. or m2 < 2 or m1/m2 < 1./18 or m1/m2 > 18 or m1+m2 > 800 or eff_spin < -0.99 or eff_spin > 0.99:
        log_mismatch = 1e6   
    else:
        
         # determine freq resolution using chirp time
        tau = calc_chirptime(m1, m2, f_min=f_low)+8 
        df = 1/2**np.ceil(np.log2(tau))
        
        # generate the lensed waveform - target 
        hp, hc = get_lensed_fd_waveform(approx, m1, m2, dL, spin1z, spin2z, iota, phi0, df, f_low, f_upp, 
                                        M_lz, y_l, intrp_hyp1f1)

        # interpolate the PSD to the required resolution  
        Sh = psd.read.from_numpy_arrays(freq_data=f_Sh, noise_data=Sh, length=len(hp.data.data), 
                                        delta_f=df, low_freq_cutoff=f_low)
        
        # calculate the mismatch with the unlensed waveform 
        log_mismatch = calc_mismatch(hp, approx, mchirp, eta, eff_spin, df, f_low, f_upp, Sh)
        
    return log_mismatch


def minimize_mismatch(fun_log_mismatch, mchirp_0, eta_0, eff_spin_0): 
       
   
   # minimize the function  
   res = minimize(fun_log_mismatch, (mchirp_0, eta_0, eff_spin_0), method='Nelder-Mead', 
                      options={'adaptive':True})    
   
   return res.fun 


def calc_ff_lensed_waveforms(approx, mchirp, eta, spin1z, spin2z, f_low, f_upp, Sh, f_Sh, 
                            M_lz, y_l, intrp_hyp1f1=None, N_iter=1):
    """ calculate the FF of lensed waveform with an unlensed waveform family 
    Note: Sh is a numpy array of PSD corresponding to the freq vector f_Sh. 
    It will be interpolated to the required resolution """
        
    mtot = mchirp * np.power(eta, -3./5)
    fac = np.sqrt(1. - 4.*eta)
    m1, m2 = (mtot * (1. + fac) / 2., mtot * (1. - fac) / 2.)
    eff_spin = (m1*spin1z + m2*spin2z)/(m1+m2)
        
    # apply some boundary 
    if m1 < 2. or m2 < 2 or m1/m2 < 1./18 or m1/m2 > 18 or m1 > 600 or m2 > 600 or m1+m2 > 700 or        eff_spin < -0.99 or eff_spin > 0.99:
        log_mismatch = 1e6   
    else:
        
        # determine freq resolution using chirp time
        tau = calc_chirptime(m1, m2, f_min=f_low)+2 
        df = 1/2**np.ceil(np.log2(tau))

        # generate the lensed waveform - target 
        hp, hc = get_lensed_fd_waveform(approx, m1, m2, dL, spin1z, spin2z, iota, phi0, df, f_low, f_upp, 
                                        M_lz, y_l, intrp_hyp1f1)
        
        # interpolate the PSD to the required resolution  
        Sh = psd.read.from_numpy_arrays(freq_data=f_Sh, noise_data=Sh, length=len(hp.data.data), 
                                        delta_f=df, low_freq_cutoff=f_low)
                        
        # function to be minimized 
        fun_log_mismatch = lambda x: calc_mismatch(hp, approx, x[0], x[1], x[2], df, f_low, f_upp, Sh)

        if N_iter > 1: 
            # spread of the distribution of starting points around the true value 
            sigma_mc, sigma_eta, sigma_spin = 0.5, 0.1, 0.1

            # generate truncated Gaussian variables centered around the true value 
            mchirp_0 = truncnorm.rvs(-3, 3, size=10*N_iter)*sigma_mc+mchirp
            eta_0 = truncnorm.rvs(-3, 3, size=10*N_iter)*sigma_eta+eta
            eff_spin_0 = truncnorm.rvs(-3, 3, size=10*N_iter)*sigma_spin+eff_spin

            # make sure that the random paramers are in the allowed region; append the true values 
            idx = (mchirp_0>1) & (mchirp_0<200) & (eta_0>0.02) & (eta_0<=0.25) & (eff_spin_0>-0.99) & (eff_spin_0<0.99)

            mchirp_0 = np.append(mchirp, np.random.choice(mchirp_0[idx], N_iter-1))
            eta_0 = np.append(eta, np.random.choice(eta_0[idx], N_iter-1))
            eff_spin_0 = np.append(eff_spin, np.random.choice(eff_spin_0[idx], N_iter-1))

            log_mismatch = np.min(np.vectorize(minimize_mismatch)(fun_log_mismatch, mchirp_0, eta_0, eff_spin_0))
        else: 
            log_mismatch = minimize_mismatch(fun_log_mismatch, mchirp, eta, eff_spin)

    print('log_mismatch = {}'.format(log_mismatch))
                
    return log_mismatch


#########################################################################################
############################## MAIN PROGRAM #############################################
#########################################################################################

parser = argparse.ArgumentParser(description='Calculate lensing bayes factor from the astro simulations.')
parser.add_argument('--m_lens', metavar='M', type=float, 
                    help='lens mass (M_sun)')
parser.add_argument('--f_dm', metavar='d', type=float, 
                    help='fraction of dark matter in the form of MACHOs')
parser.add_argument('--psd_model', metavar='p', type=str, 
                    help='psd model (H1L1V1_O3_actual_psd/H1L1V1_O4_psd')
parser.add_argument('--sim_fname', metavar='f', type=str, 
                    help='file containing the simulation results') 
parser.add_argument('--run_tag', metavar='r', type=str, 
                    help='run tag (to name the output files)') 
parser.add_argument('--out_dir', metavar='o', type=str, 
                    help='output directory') 

args = parser.parse_args()


# some parameters are hardcoded - FIXME 
approx = 'IMRPhenomD'
mkplots = True 
f_low, f_upp = 20, 1024 
dL = 400 
iota, phi0 = 0., 0. 
y0 = 5 
hyp_fn_intp = 'intrp_hyp1f1_data_final_y_3_w_150_Ngrid_1500x1499.pkl'

# get the other parameters 
f_dm = args.f_dm 
m_lens = args.m_lens
psd_model = args.psd_model
simfname = args.sim_fname
runtag = args.run_tag
outdir = args.out_dir 
mlens_min, mlens_max = m_lens-1e-6, m_lens+1e-6

os.system('mkdir -p %s' %outdir) 

# load the interpolant of the hypergeometric fun -- for the calculation of lensing magnificaiton 
open_file = open(hyp_fn_intp, "rb")
intrp_hyp1f1 = pickle.load(open_file, encoding='latin1')
open_file.close()

# PSD files 
if psd_model == 'H1L1_O1_psd':
    zs_max = 1.2
    psd_fname_H1 = '../../psds/2015_10_24_15_09_43_H1_O1_strain.txt'
    psd_fname_L1 = '../../psds/2015_10_24_15_10_43_L1_O1_strain.txt'
    psd_fname_V1 = '../../psds/Virgo_O1_fake.txt'
elif psd_model == 'H1L1V1_O2_psd':
    zs_max = 1.5
    psd_fname_H1 = '../../psds/2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt'
    psd_fname_L1 = '../../psds/2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt'
    psd_fname_V1 = '../../psds/Hrec_hoft_V1O2Repro2A_16384Hz.txt'
elif psd_model == 'H1L1V1_O3a_psd': 
    zs_max = 1.5
    psd_fname_H1 = '../../psds/aligo_O3actual_H1.txt'
    psd_fname_L1 = '../../psds/aligo_O3actual_L1.txt'
    psd_fname_V1 = '../../psds/avirgo_O3actual.txt'
elif psd_model == 'H1L1V1_O3b_psd': 
    zs_max = 1.5
    psd_fname_H1 = '../../psds/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt'
    psd_fname_L1 = '../../psds/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'
    psd_fname_V1 = '../../psds/O3-V1_sensitivity_strain_asd.txt'
elif psd_model == 'H1L1V1_O4_psd':
    zs_max = 2
    psd_fname_H1 = '../../psds/aligo_O4high.txt'
    psd_fname_L1 = '../../psds/aligo_O4high.txt'
    psd_fname_V1 = '../../psds/avirgo_O4high_NEW.txt'
elif psd_model == 'H1L1V1_O5_psd':
    zs_max = 3
    psd_fname_H1 = '../../psds/AplusDesign.txt'
    psd_fname_L1 = '../../psds/AplusDesign.txt'
    psd_fname_V1 = '../../psds/avirgo_O5high_NEW.txt'
else:
    raise ValueError('unknown psd model')


## Load the PSD data 
f_H1, asd_H1 = np.loadtxt(psd_fname_H1, unpack=True)
f_L1, asd_L1 = np.loadtxt(psd_fname_L1, unpack=True)
f_V1, asd_V1 = np.loadtxt(psd_fname_V1, unpack=True)

Sh_H1 = asd_H1**2 
Sh_L1 = asd_L1**2 
Sh_V1 = asd_V1**2 

if mkplots == True: 
    plt.figure()
    plt.loglog(f_H1, asd_H1, label='H1')
    plt.loglog(f_L1, asd_L1, label='L1')
    plt.loglog(f_V1, asd_V1, label='V1')
    plt.xlabel('$f$ [Hz]'); plt.ylabel('ASD')
    plt.legend(frameon=False)
    plt.savefig('%s/psds_%s.pdf' %(outdir, runtag))


## Calculate fitting factors from the simulations 

# load the simulation data 
D = np.load(simfname+'.npz', allow_pickle=True)
runtagff = '%s_f_dm_%.1f' %(runtag, f_dm)

# get the indices of lensed events for each f_dm and the corresponding lens redshift 
lens_idx_dic=D['lens_idx_dic'].ravel()[0]
z_l_dic = D['z_l_dic'].ravel()[0]
lens_idx = lens_idx_dic[f_dm]
z_l = z_l_dic[f_dm]

# read the simulation data 
num_det=len(D['m1_s'])
m1_s=D['m1_s'][lens_idx]
m2_s=D['m2_s'][lens_idx]
spin1_s=D['spin1_s'][lens_idx]
spin2_s=D['spin2_s'][lens_idx]
z_s=D['z_s'][lens_idx]
ldist_s=D['ldist_s'][lens_idx]
snr_L1=D['snr_L1'][lens_idx]
snr_H1=D['snr_H1'][lens_idx]
snr_V1=D['snr_V1'][lens_idx]

# calculate redshifted chirp mass 
mc_det_s = (m1_s*m2_s)**(3./5)/(m1_s+m2_s)**(1./5)
mc_det_s *= (1+z_s)
m1_det_s = m1_s*(1+z_s)
m2_det_s = m2_s*(1+z_s)
eta_s = m1_s*m2_s/(m1_s+m2_s)**2
snr_net = np.sqrt(snr_L1**2 + snr_H1**2 + snr_V1**2)


# generate lens mass 
M_l = np.logspace(np.log10(mlens_min), np.log10(mlens_max), len(mc_det_s))
np.random.shuffle(M_l)
M_lz = M_l*(1+z_l)

# pick y_l values from this vector - interpolation grid 
y_l_vec = np.linspace(0.01, 6, 3000) 
y_l_vec = y_l_vec[y_l_vec <= y0]
y_l = np.random.choice(y_l_vec, len(mc_det_s), p=y_l_vec/np.sum(y_l_vec))

if mkplots==True: 
    plt.figure(figsize=(13,5))
    plt.subplot(121)
    plt.hist(np.log10(M_lz), bins=25)
    plt.xlabel('$\log_{10} M_l^z$')
    plt.ylabel('$P(\log_{10} M_l^z)$')
    plt.subplot(122)
    plt.hist(y_l, bins=25)
    plt.xlabel('$y$')
    plt.ylabel('$P(y)$')
    plt.tight_layout()
    plt.savefig('%s/dist_Mlz_y_%s_v2.pdf' %(outdir, runtag))


# select only those events for which lensing time delay is less than the chirp time 
deltaT_l = np.vectorize(calc_lensing_timedelay)(M_lz, y_l)   #lensing time delay 
tau_s =  np.vectorize(calc_chirptime)(m1_det_s, m2_det_s, f_min=20.) # chirp time 
wo_idx = np.where(deltaT_l <= tau_s)[0]


# plot the lensing time delay as a fun of Mlz and y
if mkplots==True: 

    plt.figure(figsize=(15,6))
    plt.subplot(121)
    plt.scatter(np.log10(M_lz), y_l, c=np.log10(deltaT_l), s=8, cmap='hot')
    plt.plot(np.log10(M_lz[wo_idx]), y_l[wo_idx], ms=2, color='k', ls='none', marker='.')
    plt.colorbar()
    plt.xlabel('$\log_{10}~ M_{l}^z~(M_\odot$')
    plt.ylabel('$y$')
    plt.title('lensing time delay $\Delta T_\mathrm{l}$ (s)')
    plt.subplot(122)
    plt.plot(deltaT_l, tau_s, 'r.')
    plt.plot(deltaT_l[wo_idx], tau_s[wo_idx], 'k.')
    plt.plot(deltaT_l, deltaT_l, 'k')
    plt.xlabel('$\Delta T_\mathrm{l}$ (s)')
    plt.ylabel('$\\tau_\mathrm{chirp}$ (s)')
    plt.savefig('%s/lensing_td_chirptime_%s_v2.png' %(outdir, runtag), dpi=300)

# initialize the FF and Blu vectors with zero 
log_one_minus_ff_H1 = np.zeros_like(mc_det_s)   
log_one_minus_ff_L1 = np.zeros_like(mc_det_s)   
log_one_minus_ff_V1 = np.zeros_like(mc_det_s)   
ln_Blu = np.zeros_like(mc_det_s)  

t0 = time.time()
print('t0 = {} s'.format(t0))

# calculate the fitting factor - only for events where we expect wave optics effects - H1 
log_one_minus_ff_H1[wo_idx] = np.vectorize(calc_ff_lensed_waveforms, excluded={7,8})(approx, 
                mc_det_s[wo_idx], eta_s[wo_idx], spin1_s[wo_idx], spin2_s[wo_idx], f_low, f_upp, 
                Sh_H1, f_H1, M_lz[wo_idx], y_l[wo_idx], intrp_hyp1f1=intrp_hyp1f1)

print('log_one_minus_ff_H1 calculation is over.')

# calculate the fitting factor - only for events where we expect wave optics effects - L1 
log_one_minus_ff_L1[wo_idx] = np.vectorize(calc_ff_lensed_waveforms, excluded={7,8})(approx, 
                mc_det_s[wo_idx], eta_s[wo_idx], spin1_s[wo_idx], spin2_s[wo_idx], f_low, f_upp, 
                Sh_L1, f_L1, M_lz[wo_idx], y_l[wo_idx], intrp_hyp1f1=intrp_hyp1f1)

# calculate the fitting factor - only for events where we expect wave optics effects - V1 
log_one_minus_ff_V1[wo_idx] = np.vectorize(calc_ff_lensed_waveforms, excluded={7,8})(approx, 
                mc_det_s[wo_idx], eta_s[wo_idx], spin1_s[wo_idx], spin2_s[wo_idx], f_low, f_upp, 
                Sh_V1, f_V1, M_lz[wo_idx], y_l[wo_idx], intrp_hyp1f1=intrp_hyp1f1)

print ('FF calculation: time taken %f sec' %(time.time()-t0))


# calculate the Bayes factor - select only successful calculations 
good_idx = (log_one_minus_ff_H1 < 0) & (log_one_minus_ff_L1 < 0) & (log_one_minus_ff_V1 < 0)
good_idx =np.where(good_idx==True)[0]

print ('succcessful FF calculations = %d out of %d' %(len(good_idx), len(wo_idx)))

ln_Blu[good_idx] = 10**log_one_minus_ff_H1[good_idx]*snr_H1[good_idx]**2 +     10**log_one_minus_ff_L1[good_idx]*snr_L1[good_idx]**2 +     10**log_one_minus_ff_V1[good_idx]*snr_V1[good_idx]**2


if mkplots==True: 
    plt.figure(figsize=(14,5.9))
    plt.subplot(121)
    plt.scatter(np.log10(M_lz), y_l, c=ln_Blu, s=ln_Blu+2, cmap='Reds')
    plt.clim(0, 0.5)
    plt.ylim(min(y_l), max(y_l))
    plt.xlabel('$\log_{10}~M_\ell^z~(M_\odot)$')
    plt.ylabel('$y$')
    plt.colorbar(label='$\ln~\mathcal{B}^\mathrm{L}_\mathrm{U}$')
    plt.subplot(122)
    plt.scatter(np.log10(M_l), y_l, c=ln_Blu, s=ln_Blu+2, cmap='Reds')
    plt.clim(0, 0.5)
    plt.ylim(min(y_l), max(y_l))
    plt.xlabel('$\log_{10}~M_\ell~(M_\odot)$')
    plt.ylabel('$y$')
    plt.colorbar(label='$\ln~\mathcal{B}^\mathrm{L}_\mathrm{U}$')
    plt.tight_layout()
    plt.savefig('%s/blu_scatter_plot_Ml_y_%s.png' %(outdir, runtagff), dpi=300)


# save the Blu data 
outfile = '%s/machodm_ffdata_%s_fdm_%.1f_m_lens%.1e' %(outdir, runtag, f_dm, m_lens)
np.savez(outfile, num_det=num_det, m1_s=m1_s, m2_s=m2_s, spin1_s=spin1_s, spin2_s=spin2_s, 
         z_s=z_s, ldist_s=ldist_s, snr_L1=snr_L1, snr_H1=snr_H1, snr_V1=snr_V1, 
         lens_idx=lens_idx, z_l=z_l, M_lz=M_lz, y_l=y_l, wo_idx=wo_idx, good_idx=good_idx, ln_Blu=ln_Blu)

