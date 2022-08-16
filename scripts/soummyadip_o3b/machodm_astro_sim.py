#!/usr/bin/env python
# Astrophysical lensing simulations for the MACHO project

## Preamble

from __future__ import division
import numpy as np
from cosmology_models import LCDM
import scipy
from scipy.integrate import quad
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy import constants as const
import time, datetime
import matplotlib as mpl
plt.rcParams['agg.path.chunksize'] = 100000000

import pycbc.filter as filter
from pycbc import detector
from pycbc.waveform import get_td_waveform, get_fd_waveform
import pycbc.psd as psd
import argparse
import gwpopulation as pop


c = const.c       #speed of light (in m/s)
c_km = c/1000     #speed of light (in km/s)


# cosmological parameters - use Planck 2013 values 
from astropy.cosmology import Planck13
H_0 = Planck13.H0.value        #Hubble's constant at present epoch (in km/(s*Mpc))
Odm0 = Planck13.Odm0           #Dark matter density at current epoch
lcdm = LCDM(Om0=Planck13.Om0)
h_H0 = H_0/100.

path_plot = '../../sim_data/soummyadip_o3b/plots/'

approx = 'IMRPhenomD'
N_sim = int(1e7) 
mass_dist_pwl_index = -2.35       # source: https://arxiv.org/pdf/1602.03842.pdf, eq. no 19
#mass_model = 'power_law'
mass_model = 'power_law_peak'
m1_min, m1_max = 2.5, 200         # The choice of m_max and m_min here (power-law) is different from the power-law + peak case.
m_min, m_max = 5, 200             # m represents the total mass.
date = '2022-01-10'               # This date represents when we are running our simulations. This is just for a runtag purpose. 
y0 = 5                            # A high enough value of y0 is chosen, so that beyond that, lensing is not impotant.
#f_dm = 1 
snr_network_thresh = 8 

parser = argparse.ArgumentParser(description='Astrophysical simulation for MACHO_DM')
parser.add_argument('--psd_model', metavar='psd', type=str, help='Give PSD model as specified in the script')
parser.add_argument('--zs_dist', metavar='zs', type=str, help='zs distribution model') 
parser.add_argument('--run_tag', metavar='r', type=str, help='run tag (to name the output files)')

args = parser.parse_args()
psd_model = args.psd_model
zs_dist = args.zs_dist
runtag = args.run_tag


if psd_model == 'H1L1_O1_psd': 
    zs_max = 1.2
    psd_fname_H1 = '../../psds/2015_10_24_15_09_43_H1_O1_strain.txt'   # source: https://dcc.ligo.org/LIGO-G1600150/public
    psd_fname_L1 = '../../psds/2015_10_24_15_10_43_L1_O1_strain.txt'   # source: https://dcc.ligo.org/LIGO-G1600151/public
    psd_fname_V1 = '../../psds/Virgo_O1_fake.txt'                      # Fake O1 PSD for Virgo. Chosen values are one.

elif psd_model == 'H1L1V1_O2_psd': 
    zs_max = 1.5
    psd_fname_H1 = '../../psds/2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt'   # source: https://dcc.ligo.org/LIGO-G1801950/public
    psd_fname_L1 = '../../psds/2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt'   # source: https://dcc.ligo.org/LIGO-G1801952/public
    psd_fname_V1 = '../../psds/Hrec_hoft_V1O2Repro2A_16384Hz.txt'                    
    # O2 V1 PSD source: https://dcc.ligo.org/public/0157/P1800374/001/Hrec_hoft_V1O2Repro2A_16384Hz.txt

elif psd_model == 'H1L1V1_O3a_psd':                     
    zs_max = 1.5
    psd_fname_H1 = '../../psds/aligo_O3actual_H1.txt'   # source: https://dcc.ligo.org/public/0165/T2000012/001/aligo_O3actual_H1.txt
    psd_fname_L1 = '../../psds/aligo_O3actual_L1.txt'   # source: https://dcc.ligo.org/public/0165/T2000012/001/aligo_O3actual_L1.txt
    psd_fname_V1 = '../../psds/avirgo_O3actual.txt'     # source: https://dcc.ligo.org/public/0165/T2000012/001/avirgo_O3actual.txt

elif psd_model == 'H1L1V1_O3b_psd':                     # source: https://zenodo.org/record/5571767#.YfVenltBwnS, # figure02.tar.gz
    zs_max = 1.5                                      
    psd_fname_H1 = '../../psds/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt'
    psd_fname_L1 = '../../psds/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'
    psd_fname_V1 = '../../psds/O3-V1_sensitivity_strain_asd.txt'

elif psd_model == 'H1L1V1_O4_psd': 
    zs_max = 2
    psd_fname_H1 = '../../psds/aligo_O4high.txt'       # source: https://dcc.ligo.org/public/0165/T2000012/001/aligo_O4high.txt
    psd_fname_L1 = '../../psds/aligo_O4high.txt'       # source: https://dcc.ligo.org/public/0165/T2000012/001/aligo_O4high.txt
    psd_fname_V1 = '../../psds/avirgo_O4high_NEW.txt'  # source: https://dcc.ligo.org/public/0165/T2000012/001/avirgo_O4high_NEW.txt

elif psd_model == 'H1L1V1_O5_psd': 
    zs_max = 3
    psd_fname_H1 = '../../psds/AplusDesign.txt'        # source: https://dcc.ligo.org/public/0165/T2000012/001/AplusDesign.txt
    psd_fname_L1 = '../../psds/AplusDesign.txt'        # source: https://dcc.ligo.org/public/0165/T2000012/001/AplusDesign.txt
    psd_fname_V1 = '../../psds/avirgo_O5high_NEW.txt'  # source: https://dcc.ligo.org/public/0165/T2000012/001/avirgo_O5high_NEW.txt
else: 
    raise ValueError('unknown psd model')


#rc_params = {'axes.labelsize': 18,             # We can use these rc_parmas settings if we don't use the computing nodes in condor jobs
#             'axes.titlesize': 18,
#             'font.size': 18,
#             'lines.linewidth' : 3,
#             'legend.fontsize': 18,
#             'xtick.labelsize': 16,
#             'ytick.labelsize': 16,
#             'text.usetex' : True,
#            }
#rcParams.update(rc_params)

#from matplotlib import rc

#rc('text.latex', preamble='\\usepackage{txfonts}')
#rc('text', usetex=True)
#rc('font', family='serif')
#rc('font', serif='times')
#rc('mathtext', default='sf')
#rc("lines", markeredgewidth=1)
#rc("lines", linewidth=2)


## Differential optical depth 

# create an interpolant of the z as a fn of comoving dist 
z = np.linspace(0,10,int(1e3))
chi = np.vectorize(lcdm.comoving_distance_z)(z)
z_from_comoving_dist = interp1d(chi,z)


def differential_optical_depth(zl,zs):
    """ differential optical depth for y0=1 and f_dm=1 Ref. Eq 21 of 2001.07891v2 """
 
    chi_s = lcdm.comoving_distance_z(zs) # comoving dist - source 
    chi_l = lcdm.comoving_distance_z(zl) # comoving dist - lens 
    Ds = chi_s/(1+zs) # angular diameter dist - source 
    Dl = chi_l/(1+zl) # angular diameter dist - lens 
    Dls = (chi_s-chi_l)/(1+zs)   
    
    Ez = lcdm.hubble_normalized_z(zl)
    return 1.5*Odm0*H_0*(1+zl)**2/(c_km*Ez)*Dl*Dls/Ds  

def generate_zl(zs): 
    """ generate random samples of z_l according to the differential optical depth"""
    
    zl_vec = np.linspace(0, zs, int(1e3))
    dtau_dzl = np.vectorize(differential_optical_depth)(zl_vec, zs)
    return np.random.choice(zl_vec, 1, p=dtau_dzl/np.sum(dtau_dzl))


##  Optical depth 

def calc_optical_depth(zs, y0, f_dm):
    """ given y0 and f_dm calculate the optical depth for a given zs vector. 
    Ref. Eq 21 of 2001.07891v2"""
     
    # integrate the differential optical depth from 0 to zs 
    return y0**2*f_dm*quad(differential_optical_depth,0,zs,args=(zs))[0]


def differential_lensing_prob(zl, zs, y0, f_dm):
    """ differential lensing prob: dP/dzl = exp[-\tau(zl)] d\tau/dzl """
        
    dtau_dzl = y0**2*f_dm*differential_optical_depth(zl,zs)
    tau_zl = calc_optical_depth(zl, y0, f_dm)
    
    return np.exp(-tau_zl)*dtau_dzl


zs = np.linspace(1e-6, 2, 100)
y0 = 3.5 

plt.figure(figsize=(5.5,5.3))
for f_dm in [0.1, 0.5, 1]:
    tau = np.vectorize(calc_optical_depth)(zs,y0,f_dm)
    P_lens = 1-np.exp(-tau)
    plt.plot(zs, tau, label='$f_\mathrm{DM} = %.1f$' %f_dm)
    plt.plot(zs, P_lens, lw=1, color='k', ls='--')
plt.xlabel(r"$z_s$",fontsize=20)
plt.ylabel(r"$\tau(z_s)$ and $P_\ell(z_s)$",fontsize=20)
plt.xlim(min(zs), max(zs))
plt.ylim(0, 1)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(path_plot+'optical_depth_vs_zs_y0_%.1f_%s.pdf' %(y0, runtag))
plt.close()


def calc_redshift_dist(zs, zs_dist, zs_max):
    """ redshift distribution of binaries """
     
    # choose the redshift distribution of binaries 
    if zs_dist == 'uniform': # uniform in comoving volume 
        z = zs
        chi_s = np.vectorize(lcdm.comoving_distance_z)(zs)
        Ez = lcdm.hubble_normalized_z(zs)
        Pb_z = 4*np.pi*c_km*chi_s**2/(1+zs)/H_0/Ez 
        
    elif zs_dist == 'Dominik' or zs_dist == 'Belczynski' or zs_dist == 'popIII' or zs_dist == 'primordial': 
            
        # read the redshift distribution data 
        z, Pz_Dominik, Pz_Belczynski, Pz_PopIII, Pz_primo = np.loadtxt('../../data/redshift_data/z_PDF.dat',\
                                                                       unpack=True, skiprows=1)

    elif zs_dist == 'O3cosmo':  
        Gpc3 = 1e9
        R0h67 = 22./Gpc3    # Mpc^3/yr # Local value of merger rate
        R0 = R0h67/h_H0**3  # convert rate to h^3/Mpc^3/yr

        kap = 2.86
        gam = 4.59
        zpk = 2.47
        R0p = 22./Gpc3

        Fac = 1 + np.power((1+zpk), (-gam-kap)) 
        denomfac = lambda z: np.power ((1+z)/(1+zpk), (gam+kap))
        RzO3 = lambda z: R0p*Fac * (1+z)**gam / (1 + denomfac(z))
        Rzfun = RzO3(zs)
        z = zs
        chi_s = np.vectorize(lcdm.comoving_distance_z)(zs)
        Ez = lcdm.hubble_normalized_z(zs)
        Pb_z = 4*np.pi*c_km*chi_s**2*Rzfun/(1+zs)/H_0/Ez 
        
    elif zs_dist == 'O3pop':
        z = zs
        chi_s = np.vectorize(lcdm.comoving_distance_z)(zs)
        Ez = lcdm.hubble_normalized_z(zs)
        kappa = 2.9
        Pb_z = (1+z)**kappa*4*np.pi*c_km*chi_s**2/(1+zs)/H_0/Ez   # source: https://arxiv.org/pdf/2111.03634.pdf, eq. 8, fig 13

    elif zs_dist == 'Madau-Dickinson':
        z = zs
        chi_s = np.vectorize(lcdm.comoving_distance_z)(zs)   
        Ez = lcdm.hubble_normalized_z(zs)
        psi_z = 0.015*(1+zs)**2.7/(1+((1+zs)/2.9)**5.6)
        Pb_z = psi_z*4*np.pi*c_km*chi_s**2/(1+zs)/H_0/Ez     # source: https://arxiv.org/pdf/1805.10270.pdf, eq. 8 and 9

    else: 
        raise ValueError('unknown zs_dist')
        
    if zs_dist == 'Dominik':
        Pb_z = Pz_Dominik
    elif zs_dist == 'Belczynski':
        Pb_z = Pz_Belczynski
    elif zs_dist == 'popIII':
        Pb_z = Pz_PopIII
    elif zs_dist == 'primordial':
        Pb_z = Pz_primo

    # interpolate to the given samples of zs 
    Pb_zs = np.interp(zs, z, Pb_z)
    Pb_zs /= np.sum(Pb_zs)*np.mean(np.diff(zs))
             
    return Pb_zs


# plot the redshift distributions 
zs_vec = np.linspace(0, 2, 100)

plt.figure(figsize=(6.2,5.9))
for zdist in ['uniform', 'Dominik', 'Belczynski', 'O3cosmo', 'O3pop']:
    P_zs_vec = calc_redshift_dist(zs_vec, zdist, zs_max=2)
    
    plt.plot(zs_vec, P_zs_vec, label=zdist)

plt.xlabel('$z_s$')
plt.ylabel('$P(z_s)$')
plt.xlim(min(zs_vec), max(zs_vec))
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(path_plot+'binary_redshift_dist_%s.pdf' %runtag)
plt.close()


## Fraction of lensed events: Analytical calculation 

def calc_lensing_frac_integrant(zs, zs_dist, zs_max, y0, f_dm):
    """ integrant for the lensing faction calculation """

    Pb_zs = calc_redshift_dist(zs, zs_dist, zs_max) # redshift distribution of binaries 
    tau = calc_optical_depth(zs,y0,f_dm) # lensing optical depth 
    P_lens = 1-np.exp(-tau)              # lensing prob
    
    return Pb_zs*P_lens


def calc_lensing_frac_quad(zs_dist, zs_max, y0, f_dm): 
    """ calculate the fraction of lensed mergers"""
    return quad(calc_lensing_frac_integrant,0,zs_max,args=(zs_dist, z_max, y0, f_dm))[0]


def calc_lensing_frac(zs_dist, zs_max, y0, f_dm): 
    """ calculate the fraction of lensed mergers"""

    zs = np.linspace(1e-9, zs_max, 100)
    Pb_zs = calc_redshift_dist(zs, zs_dist, zs_max) # redshift distribution of binaries 
    tau = np.vectorize(calc_optical_depth)(zs,y0,f_dm) # lensing optical depth 
    P_lens = 1-np.exp(-tau)   
    
    return np.trapz(Pb_zs*P_lens, zs)


# plot the fraction of lensed events as a function of f_dm  
f_dm_vec = np.linspace(0, 1, 10)

plt.figure(figsize=(6.2,5.9))

for zdist in ['uniform', 'Dominik', 'Belczynski', 'O3cosmo', 'O3pop']:
    
    u_vec = np.vectorize(calc_lensing_frac)(zdist, zs_max, y0, f_dm_vec)
    plt.plot(f_dm_vec, u_vec, label=zdist)

plt.xlabel('$f_\mathrm{DM}$')
plt.ylabel('lensing fraction, $u$')
plt.legend(frameon=False)
plt.xlim(0,1)
plt.ylim(0,np.max(u_vec)*1.05)
plt.tight_layout()
plt.savefig(path_plot+'lensing_frac_vs_fdm_analytic_%s.pdf' %runtag)
plt.close()


## Simulations 

#def generate_random_numbers_power_law_dist(x0, x1, n, N_sampl):
#    """generate a random number x with a power-law distribition P(x) = x^n
#    distributed between x0 and x1"""
#    
#    y = np.random.uniform(0, 1, N_sampl)
#    return ((x1**(n+1) - x0**(n+1))*y + x0**(n+1))**(1/(n+1))

def generate_m1_and_q_samples(m1_min, m1_max, mass_dist_pwl_index, N_sampl, model='power_law'):
    """generate mass samples with a power-law distribition P(m1) = m1^n"""
    
    m1_vec = np.linspace(m1_min, m1_max, 2*N_sampl)
    q_vec = np.linspace(1/18, 1, 2*N_sampl)
    
    if model=='power_law': 
        P_m1_vec = m1_vec**mass_dist_pwl_index
        P_q_vec = np.ones_like(q_vec)

    elif model=='power_law_peak':
        lam = 0.03          
        alpha = 3.78        
        beta = 0.81         
        mpp = 32.27         
        sigpp = 3.88        
        mmax = 112.5        # The choice of m_max and m_min here (power-law + peak) is different 
        mmin = 4.98         # from the power-law case.
        delta_m = 4.8  
        dataset = {}
        dataset["mass_1"] = m1_vec
        
        mass_model = pop.models.mass.SinglePeakSmoothedMassDistribution(mmin=mmin, mmax=mmax)
        P_m1_vec = mass_model.p_m1(dataset, mmin=mmin, mmax=mmax, delta_m=delta_m, lam=lam, alpha=alpha, mpp=mpp, sigpp=sigpp)
        
        dataset["mass_ratio"] = q_vec
        P_q_vec = mass_model.p_q(dataset, beta=beta, mmin=mmin, delta_m=delta_m)

    else: 
        raise ValueError('Unknown mass model')
        
    m1_sampl = np.random.choice(m1_vec, N_sampl, p=P_m1_vec/np.sum(P_m1_vec))
    q_sampl = np.random.choice(q_vec, N_sampl, p=P_q_vec/np.sum(P_q_vec))
    
    return m1_sampl, q_sampl


def generate_z_samples(zs_dist, zs_max, N_sampl):
    """ generate samples from a chosen redshift distribution """
    
    zs_vec = np.linspace(0, zs_max, 2*N_sampl)
    P_zs_vec = calc_redshift_dist(zs_vec, zs_dist, zs_max=zs_max)
    
    return np.random.choice(zs_vec, N_sampl, p=P_zs_vec/np.sum(P_zs_vec))


## Generate binary parameters 

#m1_s = generate_random_numbers_power_law_dist(m1_min, m1_max, mass_dist_pwl_index, N_sim)
#q_s = np.random.uniform(1/18., 1, N_sim) 
m1_s, q_s = generate_m1_and_q_samples(m1_min, m1_max, mass_dist_pwl_index, N_sim, model=mass_model)
m2_s = m1_s*q_s
m_s = m1_s + m2_s 
z_s = generate_z_samples(zs_dist, zs_max, N_sim)

## select samples that satisfy m_min < m < m_max and z < z_max 
idx = (m2_s >= m1_min) & (m_s >= m_min) & (m_s <= m_max) & (z_s <= zs_max)
m1_s, m2_s, m_s, q_s, z_s = m1_s[idx], m2_s[idx], m_s[idx], q_s[idx], z_s[idx]

N_sim = len(m1_s)
ra_s = np.random.uniform(0, 2*np.pi, N_sim)             # RA  
sindec_s = np.random.uniform(-1, 1, N_sim)              # sin(dec) 
pol_s = np.random.uniform(0, 2*np.pi, N_sim)            # polarization 
spin1_s = np.random.uniform(-0.99, 0.99, N_sim)         # spin1z 
spin2_s = np.random.uniform(-0.99, 0.99, N_sim)         # spin2z 
cosiota_s = np.random.uniform(-1, 1, N_sim)             # cos(iota)
phi0_s = np.random.uniform(0, 2*np.pi, N_sim)           # phi_0 

dec_s = np.arcsin(sindec_s)                             # dec 
iota_s = np.arccos(cosiota_s)                           # iota 
m1z_s, m2z_s = m1_s*(1+z_s), m2_s*(1+z_s)               # redshifted masses 
ldist_s = np.vectorize(lcdm.luminosity_distance_z)(z_s) # luminosity distance 

print('binary samples are generated.')

plt.figure(figsize=(5,5))
plt.plot(m1_s, m2_s, 'r.', ms=5)
plt.xlabel('$m_1 (\mathrm{M_\odot})$')
plt.ylabel('$m_2 (\mathrm{M_\odot})$')
plt.ylim(0,m1_max)


mvec = np.linspace(5,200,1000)
zs_vec = np.linspace(0,zs_max,1000)
P_zs_vec = calc_redshift_dist(zs_vec, zs_dist, zs_max=zs_max)

plt.figure(figsize=(14,5.5))
plt.subplot(251)
plt.hist(m1_s, 20, density=True)
plt.plot(mvec, 10*mvec**-2.35, 'r')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('$m_1~(\mathrm{M_\odot})$')
plt.ylabel('$P(m_1)$')
plt.subplot(252)
plt.hist(q_s, 20, density=True)
plt.xlabel('$q$')
plt.ylabel('$P(q)$')
plt.subplot(253)
plt.hist(z_s, 20, density=True)
plt.plot(zs_vec, P_zs_vec, 'r')
plt.xlim(0, zs_max)
plt.xlabel('$z$')
plt.ylabel('$P(z)$')
plt.subplot(254)
plt.hist(ra_s, 20, density=True)
plt.xlabel('$\\alpha$')
plt.ylabel('$P(\\alpha)$')
plt.subplot(255)
plt.hist(sindec_s, 20, density=True)
plt.xlabel('$\sin \delta$')
plt.ylabel('$P(\sin \delta)$')
plt.subplot(256)
plt.hist(pol_s, 20, density=True)
plt.xlabel('$\psi$')
plt.ylabel('$P(\psi)$')
plt.subplot(257)
plt.hist(spin1_s, 20, density=True)
plt.xlabel('$a_1$')
plt.ylabel('$P(a_1)$')
plt.subplot(258)
plt.hist(spin2_s, 20, density=True)
plt.xlabel('$a_2$')
plt.ylabel('$P(a_2)$')
plt.subplot(259)
plt.hist(cosiota_s, 20, density=True)
plt.xlabel('$\cos \iota$')
plt.ylabel('$P(\cos \iota)$')
plt.subplot(2,5,10)
plt.hist(phi0_s, 20, density=True)
plt.xlabel('$\phi_0$')
plt.ylabel('$P(\phi_0)$')

plt.tight_layout()
plt.savefig(path_plot+'param_dist_inj_%s.pdf' %runtag)

print('param_dist_inj plot is done.')

## Calculate SNR in 3 detectors 

def calc_snr(m1, m2, spin1z, spin2z, dL, ra, dec, pol, iota, phi0, f_low, f_upp, df, t0_gps, approx): 
    """ calculate the optimal SNR at H1, L1, V1 detectors
    Note: Requires the detector objects H1, L1, V1 to be set globally. Also psd_H1, psd_L1, psd_V1
    """
    
    # generate waveform 
    hp, hc = get_fd_waveform(approximant=approx, mass1=m1, mass2=m2, distance=dL, spin1z=spin1z, spin2z=spin2z, inclination=iota,\
                             coa_phase=phi0, delta_f=df, f_lower=f_low, f_final=f_upp)

    # calculate SNR 
    snr = np.zeros(3)
    det_obj_list = [H1, L1, V1]            # H1, L1, V1 are detector objects (global variables)
    psd_list = [psd_H1, psd_L1, psd_V1]    # these are psds of 3 detectors (global variables )
        
    for i_det, det_obj in enumerate(det_obj_list):
        
        Fp, Fc = det_obj.antenna_pattern(ra, dec, pol, t0_gps)
        h = Fp*hp + Fc*hc
        snr[i_det] = filter.matchedfilter.sigma(h, psd=psd_list[i_det], low_frequency_cutoff=f_low, high_frequency_cutoff=f_upp)

    return snr[0], snr[1], snr[2]


df = 1.          # make sure that df is an integer fraction of f_upp 
f_low = 20.
f_upp = 1024.
length = int(f_upp/df) 

t0_gps = np.random.uniform(1230336018, 1261872018, N_sim)  # This corresponds to 1st January 2019, 00:00:00 UTC to 1st January 2020,\
                                                           # 00:00:00 UTC. The choice of t0_gps is to mimic a LIGO observing run.

# create detector objects - global variables 
H1 = detector.Detector('H1')
L1 = detector.Detector('L1')
V1 = detector.Detector('V1')

psd_H1 = psd.read.from_txt(psd_fname_H1, length, df, f_low, is_asd_file=True)
psd_L1 = psd.read.from_txt(psd_fname_L1, length, df, f_low, is_asd_file=True)
psd_V1 = psd.read.from_txt(psd_fname_V1, length, df, f_low, is_asd_file=True)

# plot the PSDs 
f = psd_H1.sample_frequencies
plt.figure()
plt.loglog(f, np.sqrt(psd_H1), label='H1')
plt.loglog(f, np.sqrt(psd_L1), label='L1', ls='--')
plt.loglog(f, np.sqrt(psd_V1), label='V1')
plt.xlabel('$f$ [Hz]')
plt.ylabel('Strain$[1/\sqrt{\mathrm{Hz}}]$')
plt.legend(frameon=False)
plt.xlim(f_low, f_upp)
plt.tight_layout()
plt.savefig(path_plot+'det_psds_%s.pdf' %runtag)
plt.close()

print(datetime.datetime.now().strftime('%d/%m/%Y-%H:%M:%S'))

time0 = time.time()

snr_H1, snr_L1, snr_V1 = np.vectorize(calc_snr)(m1z_s, m2z_s, spin1_s, spin2_s, ldist_s, ra_s, dec_s, pol_s,\
                                                iota_s, phi0_s, f_low, f_upp, df, t0_gps, approx)

time1 = time.time()

print(datetime.datetime.now().strftime('%d/%m/%Y-%H:%M:%S'))

print ("time taken for SNR calculation = %f s" %(time1-time0))


## Apply SNR threshold 

# identify detectable binaries 
snr_network = np.sqrt(snr_H1**2 + snr_L1**2 + snr_V1**2)
det_idx = np.where(snr_network >= snr_network_thresh)[0]


plt.figure(figsize=(7,5))
plt.scatter(m1z_s+m2z_s, z_s, c=np.log10(snr_network), s=np.log10(snr_network), cmap='Reds')
plt.plot(m1z_s[det_idx]+m2z_s[det_idx], z_s[det_idx], ms=1, color='k', marker='.', ls='none')

plt.xscale('log'); 
plt.ylim(0, zs_max)
plt.xlabel('$M_z (\mathrm{M_\odot})$')
plt.ylabel('$z$')
plt.title('$\log_{10}$ network SNR')
plt.colorbar()
plt.savefig(path_plot+'snr_scatter_plot_%s.png' %runtag, dpi=200)
plt.close()


# plot the detectable binaries on top of the simulated binaries 
plt.figure(figsize=(14,5.5))
plt.subplot(251)
plt.hist(m1_s, 20, density=False)
plt.hist(m1_s[det_idx], 20, density=False)
plt.plot(mvec, 10*mvec**-2.35, 'r')
plt.yscale('log')
plt.xlabel('$m_1 (\mathrm{M_\odot})$')
plt.ylabel('$P(m_1)$')
plt.subplot(252)
plt.hist(q_s, 20, density=False)
plt.hist(q_s[det_idx], 20, density=False)
plt.xlim(0,1)
plt.xlabel('$q$')
plt.ylabel('$P(q)$'); plt.yscale('log')
plt.subplot(253)
plt.hist(z_s, 20, density=False)
plt.hist(z_s[det_idx], 20, density=False)
plt.plot(zs_vec, P_zs_vec, 'r')
plt.xlim(0, zs_max)
plt.xlabel('$z$')
plt.ylabel('$P(z)$'); plt.yscale('log')
plt.subplot(254)
plt.hist(ra_s, 20, density=False)
plt.hist(ra_s[det_idx], 20, density=False)
plt.xlim(0,2*np.pi)
plt.xlabel('$\\alpha$')
plt.ylabel('$P(\\alpha)$'); plt.yscale('log')
plt.subplot(255)
plt.hist(sindec_s, 20, density=False)
plt.hist(sindec_s[det_idx], 20, density=False)
plt.xlim(-1,1)
plt.xlabel('$\sin \delta$')
plt.ylabel('$P(\sin \delta)$'); plt.yscale('log')
plt.subplot(256)
plt.hist(pol_s, 20, density=False)
plt.hist(pol_s[det_idx], 20, density=False)
plt.xlim(0,2*np.pi)
plt.xlabel('$\psi$')
plt.ylabel('$P(\psi)$'); plt.yscale('log')
plt.subplot(257)
plt.hist(spin1_s, 20, density=False)
plt.hist(spin1_s[det_idx], 20, density=False)
plt.xlim(-1,1)
plt.xlabel('$a_1$')
plt.ylabel('$P(a_1)$'); plt.yscale('log')
plt.subplot(258)
plt.hist(spin2_s, 20, density=False)
plt.hist(spin2_s[det_idx], 20, density=False)
plt.xlim(-1,1)
plt.xlabel('$a_2$')
plt.ylabel('$P(a_2)$'); plt.yscale('log')
plt.subplot(259)
plt.hist(cosiota_s, 20, density=False)
plt.hist(cosiota_s[det_idx], 20, density=False)
plt.xlim(-1,1)
plt.xlabel('$\cos \iota$')
plt.ylabel('$P(\cos \iota)$'); plt.yscale('log')
plt.subplot(2,5,10)
plt.hist(phi0_s, 20, density=False)
plt.hist(phi0_s[det_idx], 20, density=False)
plt.xlim(0, 2*np.pi)
plt.xlabel('$\phi_0$')
plt.ylabel('$P(\phi_0)$'); plt.yscale('log')
plt.tight_layout(pad=0.01)
plt.savefig(path_plot+'param_dist_inj_and_det_%s.pdf' %runtag)
plt.close()


## Identify lensed events from simulations 
time2 = time.time()
print ("plotted param distributions. Time taken = %f s" %(time2-time1))


f_dm_vec = [1, 0.8, 0.6, 0.4, 0.2]

lens_idx_dic = {}
z_l_dic = {}

for f_dm in f_dm_vec: 
      
    # calculate the lensing probability -- only for the detectable events 
    tau_l = np.vectorize(calc_optical_depth)(z_s[det_idx], y0, f_dm)
    P_l = 1-np.exp(-tau_l)

    # identify the lensed events among them. This is done by checking if P_l >= a uniform random number 
    r = np.random.uniform(0, 1, len(P_l))
    lens_idx = np.where(P_l >= r)[0]

    # randomly pick redshift of the lens from the differential optical depth (dtau/dzl ~ dP/dzl) 
    z_l = np.vectorize(generate_zl)(z_s[det_idx[lens_idx]])
 
    # save the indices and redshift of lensed events as a dictionary 
    lens_idx_dic[f_dm] = lens_idx
    z_l_dic[f_dm] = z_l

    plt.figure(figsize=(14,4.5))
    plt.subplot(131)
    plt.plot(z_s[det_idx], P_l, 'r.')
    plt.xlabel('$z_s$'); plt.ylabel('$P_\ell(z_s)$')
    plt.xlim(0,2)
    plt.subplot(132)
    plt.hist(z_s[det_idx], 30, label='det events')
    plt.hist(z_s[det_idx[lens_idx]], 30, label='lensed')
    plt.xlabel('$z_s$'); plt.ylabel('$N$')
    plt.legend(frameon=False)
    plt.xlim(0,2)
    plt.subplot(133)
    plt.plot(z_s[det_idx[lens_idx]], z_l, 'k.')
    plt.plot(z_s, z_s, color='k', lw=1)
    plt.xlabel('$z_s$'); plt.ylabel('$z_\ell$')
    plt.xlim(0,2)
    plt.ylim(0,2)
    plt.tight_layout()
    plt.savefig(path_plot+'lens_redshift_dist_from_sim_%s_fdm_%.1f.png' %(runtag, f_dm), dpi=300)
    plt.close()

time3 = time.time()
print ("identified lensed events. Time taken = %f s" %(time3-time2))

# save the data 
outfile = '../../sim_data/soummyadip_o3b/astro_sim_data/machodm_lensing_astro_sim_%s' %(runtag)
np.savez(outfile, m1_s=m1_s, m2_s=m2_s, spin1_s=spin1_s, spin2_s=spin2_s, z_s=z_s, ldist_s=ldist_s, 
         ra_s=ra_s, dec_s=dec_s, pol_s=pol_s, iota_s=iota_s, phi0_s=phi0_s, t0_gps=t0_gps, 
         snr_L1=snr_L1, snr_H1=snr_H1, snr_V1=snr_V1, det_idx=det_idx, df=df, f_low=f_low, f_upp=f_upp, 
         z_l=z_l, lens_idx_dic=lens_idx_dic, z_l_dic=z_l_dic)

time4 = time.time()
print ("saved data. time taken  = %f s" %(time4-time3))


# save only detected events 
outfile = '../../sim_data/soummyadip_o3b/astro_sim_data/machodm_lensing_astro_sim_%s_lw' %(runtag)
np.savez(outfile, m1_s=m1_s[det_idx], m2_s=m2_s[det_idx], spin1_s=spin1_s[det_idx], spin2_s=spin2_s[det_idx], 
         z_s=z_s[det_idx], ldist_s=ldist_s[det_idx], ra_s=ra_s[det_idx], dec_s=dec_s[det_idx], 
         pol_s=pol_s[det_idx], iota_s=iota_s[det_idx], phi0_s=phi0_s[det_idx], t0_gps=t0_gps[det_idx], 
         snr_L1=snr_L1[det_idx], snr_H1=snr_H1[det_idx], snr_V1=snr_V1[det_idx], df=df, f_low=f_low, f_upp=f_upp, 
         lens_idx_dic=lens_idx_dic, z_l_dic=z_l_dic)

time5 = time.time()
print ("saved lw data. time taken  = %f s" %(time5-time4))

## Plot the distribution of lensed events along with all and detected events 
plt.figure(figsize=(14,5.1))
plt.subplot(251)
plt.hist(m1_s, 20, density=False, facecolor='lightgray')
plt.hist(m1_s[det_idx], 20, density=False, facecolor='gray')
plt.hist(m1_s[det_idx[lens_idx_dic[1]]], 20, density=False, facecolor='k')
plt.xlim(min(m1_s), max(m1_s))
plt.yscale('log')
plt.xlabel('$m_1 (\mathrm{M_\odot})$')
plt.ylabel('$P(m_1)$')
plt.subplot(252)
plt.hist(q_s, 20, density=False, facecolor='lightgray')
plt.hist(q_s[det_idx], 20, density=False, facecolor='gray')
plt.hist(q_s[det_idx[lens_idx_dic[1]]], 20, density=False, facecolor='k')
plt.xlim(0,1)
plt.xlabel('$q$')
plt.ylabel('$P(q)$'); plt.yscale('log')
plt.subplot(253)
plt.hist(z_s, 20, density=False, facecolor='lightgray')
plt.hist(z_s[det_idx], 20, density=False, facecolor='gray')
plt.hist(z_s[det_idx[lens_idx_dic[1]]], 20, density=False, facecolor='k')
plt.xlim(0, zs_max)
plt.xlabel('$z_s$')
plt.ylabel('$P(z_s)$'); 
plt.yscale('log')
plt.subplot(254)
plt.hist(ra_s, 20, density=False, facecolor='lightgray')
plt.hist(ra_s[det_idx], 20, density=False, facecolor='gray')
plt.hist(ra_s[det_idx[lens_idx_dic[1]]], 20, density=False, facecolor='k')
plt.xlim(0,2*np.pi)
plt.xlabel('$\\alpha$')
plt.ylabel('$P(\\alpha)$'); plt.yscale('log')
plt.subplot(255)
plt.hist(sindec_s, 20, density=False, facecolor='lightgray')
plt.hist(sindec_s[det_idx], 20, density=False, facecolor='gray')
plt.hist(sindec_s[det_idx[lens_idx_dic[1]]], 20, density=False, facecolor='k')

plt.xlim(-1,1)
plt.xlabel('$\sin \delta$')
plt.ylabel('$P(\sin \delta)$'); plt.yscale('log')
plt.subplot(256)
plt.hist(pol_s, 20, density=False, facecolor='lightgray')
plt.hist(pol_s[det_idx], 20, density=False, facecolor='gray')
plt.hist(pol_s[det_idx[lens_idx_dic[1]]], 20, density=False, facecolor='k')

plt.xlim(0,2*np.pi)
plt.xlabel('$\psi$')
plt.ylabel('$P(\psi)$'); plt.yscale('log')
plt.subplot(257)
plt.hist(spin1_s, 20, density=False, facecolor='lightgray')
plt.hist(spin1_s[det_idx], 20, density=False, facecolor='gray')
plt.hist(spin1_s[det_idx[lens_idx_dic[1]]], 20, density=False, facecolor='k')


plt.xlim(-1,1)
plt.xlabel('$a_1$')
plt.ylabel('$P(a_1)$'); plt.yscale('log')
plt.subplot(258)
plt.hist(spin2_s, 20, density=False, facecolor='lightgray')
plt.hist(spin2_s[det_idx], 20, density=False, facecolor='gray')
plt.hist(spin2_s[det_idx[lens_idx_dic[1]]], 20, density=False, facecolor='k')
plt.xlim(-1,1)
plt.xlabel('$a_2$')
plt.ylabel('$P(a_2)$'); plt.yscale('log')
plt.subplot(259)
plt.hist(cosiota_s, 20, density=False, facecolor='lightgray')
plt.hist(cosiota_s[det_idx], 20, density=False, facecolor='gray')
plt.hist(cosiota_s[det_idx[lens_idx_dic[1]]], 20, density=False, facecolor='k')
plt.xlim(-1,1)
plt.xlabel('$\cos \iota$')
plt.ylabel('$P(\cos \iota)$'); plt.yscale('log')
plt.subplot(2,5,10)
plt.hist(phi0_s, 20, density=False, facecolor='lightgray')
plt.hist(phi0_s[det_idx], 20, density=False, facecolor='gray')
plt.hist(phi0_s[det_idx[lens_idx_dic[1]]], 20, density=False, facecolor='k')
plt.xlim(0, 2*np.pi)
plt.xlabel('$\phi_0$')
plt.ylabel('$P(\phi_0)$'); plt.yscale('log')
plt.tight_layout(pad=0.5)
plt.savefig(path_plot+'param_dist_inj_det_lens_%s.pdf' %runtag)
plt.close()


time6 = time.time()
print ("total time taken  = %f s" %(time6-time0))


