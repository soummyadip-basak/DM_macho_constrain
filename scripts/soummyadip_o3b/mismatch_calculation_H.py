#!/home1/soummyadip.basak/bilby/ve3/bilby_som/bin/python

from __future__ import division
import matplotlib
# matplotlib.use('Agg')
import numpy as np
import scipy
from scipy.integrate import quad
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy import constants as const
import time 
from scipy.optimize import minimize
import mpmath 
import scipy.special as special
import multiprocessing as mp
from multiprocessing import Pool
import pickle
import pandas as pd
import scipy.stats
from scipy import integrate, interpolate
import time
import datetime
import lal
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.types import TimeSeries, FrequencySeries, zeros
from pycbc.detector import Detector
from pycbc.filter.matchedfilter import match, sigma
from pycbc.psd import aLIGOZeroDetHighPower

def lens_ampl_function(f, M_lz, y):         # input of f should always be an array
                                                               
    """ frequency dependent lensing magnification """
    
    w = 8*np.pi*M_lz*lal.MTSUN_SI*f
    x_m = (y + np.sqrt(y**2+4))/2.
    phi_m = (x_m - y)**2/2. - np.log(x_m)
        
    if y<=1:

        hyp_fn = np.vectorize(mpmath.hyp1f1)(0.5*w*1j, 1., 0.5*w*y**2*1j, maxterms=1e7)
        hyp_fn = np.array(hyp_fn.tolist(), dtype=complex)  # convert to numpy array from mpmath format 
        
        np_gamma = np.frompyfunc(mpmath.gamma, 1, 1)
        mp_gamma = np_gamma((np.ones(len(w))-0.5*w*1j))

        np_exp = np.frompyfunc(mpmath.exp, 1, 1)
        mp_exp = np_exp(np.pi*w/4.+0.5*w*1j*(np.log(w/2.)-2*phi_m))

        np_gamma_exp = np.array(mp_gamma*mp_exp.tolist(), dtype='complex64')
        
        F_f = np_gamma_exp*hyp_fn
        
        return F_f
        
    else:
        
        w2 = lal.MTSUN_SI*1024*8*np.pi*1e4
        w_1 = w[np.where(w <= w2)[0]]
        w_2 = w[np.where(w > w2)[0]]
        
        hyp_fn = np.vectorize(mpmath.hyp1f1)(0.5*w_1*1j, 1., 0.5*w_1*y**2*1j, maxterms=1e7)
        hyp_fn = np.array(hyp_fn.tolist(), dtype=complex)  # convert to numpy array from mpmath format 
        
        np_gamma = np.frompyfunc(mpmath.gamma, 1, 1)
        mp_gamma = np_gamma((np.ones(len(w_1))-0.5*w_1*1j))

        np_exp = np.frompyfunc(mpmath.exp, 1, 1)
        mp_exp = np_exp(np.pi*w_1/4.+0.5*w_1*1j*(np.log(w_1/2.)-2*phi_m))

        np_gamma_exp = np.array(mp_gamma*mp_exp.tolist(), dtype='complex64')
        
        F_w = np_gamma_exp*hyp_fn
        
        mu_p = np.sqrt(abs(1/2. + (y**2+2)/(2*y*np.sqrt(y**2+4))))
        mu_n = np.sqrt(abs(1/2. - (y**2+2)/(2*y*np.sqrt(y**2+4))))
        del_t = y*np.sqrt(y**2+4)/2. + np.log((np.sqrt(y**2+4)+y)/(np.sqrt(y**2+4)-y))
        F_g = mu_p - 1j*mu_n*np.exp(1j*w_2*del_t)

        F_f = np.concatenate((F_w,F_g))
        
        return F_f        

path = '/home1/soummyadip.basak/bilby/lens_amp_fun/'

frequency = np.loadtxt(path+'GW190814_PSD_H1.dat', usecols=(0))
PSD_H = np.loadtxt(path+'GW190814_PSD_H1.dat', usecols=(1))
PSD_L = np.loadtxt(path+'GW190814_PSD_L1.dat', usecols=(1))
PSD_V = np.loadtxt(path+'GW190814_PSD_V1.dat', usecols=(1))
Det = [Detector('H1'), Detector('L1'), Detector('V1')]
PSD = np.array([PSD_H,PSD_L,PSD_V])

hyp_fn_intp_hybrid = '/home1/ajith/working/cbc/lensing/DM_macho_constrain/data/intrp_cubic_hyp1f1_data_Ngrid3000x10000.pkl'

open_file = open(hyp_fn_intp_hybrid, "rb")
intrp_hyp1f1_hybrid = pickle.load(open_file)
open_file.close()


def lens_ampl_function_wave_optics(f, M_lz, y, intrp_hyp1f1=None):
    """ frequency dependent lensing magnification """
        
    w = 8*np.pi*M_lz*lal.MTSUN_SI*f
    x_m = (y + np.sqrt(y**2 +4)) / 2.
    phi_m = (x_m - y)**2/2. - np.log(x_m)
    
    if intrp_hyp1f1==None:
        # evaluate the hypergeometric fun using mpmath package 
        hyp_fn = np.vectorize(mpmath.hyp1f1)(0.5*w*1j, 1., 0.5*w*y**2*1j, maxterms=1e7)
        hyp_fn = np.array(hyp_fn.tolist(), dtype=complex)  # convert to numpy array from mpmath format 
    else: 
        # evaluate the interpolated function 
        hyp_fn = 10**intrp_hyp1f1['log_abs'](w,y)*np.exp(1j*intrp_hyp1f1['arg'](w,y))

    np_gamma = np.frompyfunc(mpmath.gamma, 1, 1)
    mp_gamma = np_gamma((np.ones(len(w))-0.5*w*1j))

    np_exp = np.frompyfunc(mpmath.exp, 1, 1)
    mp_exp = np_exp(np.pi*w/4.+0.5*w*1j*(np.log(w/2.)-2*phi_m))

    np_gamma_exp = np.array(mp_gamma*mp_exp.tolist(), dtype='complex64')
    
    F_f = np_gamma_exp*hyp_fn
        
    return F_f

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

def lens_ampl_function_hybrid(f, M_lz, y, intrp_hyp1f1=None):
    """ frequency dependent lensing magnification: hybrid using wave optics and geometric optics """
    
    w = 8*np.pi*M_lz*lal.MTSUN_SI*f
    
    # if y >= 2 geometric optics provides a good approximation 
    if y >= 2: 
        F_f = lens_ampl_factor_geom_optics(f, M_lz, y)
    # if y < 2, use a combination of wave optics (low freq) and geom optics (high freq)
    
    else:
        w_max = 100
        w_1 = w[np.where(w <= w_max)[0]]
        w_2 = w[np.where(w > w_max)[0]]
        f_1 = w_1/(8*np.pi*M_lz*lal.MTSUN_SI)
        f_2 = w_2/(8*np.pi*M_lz*lal.MTSUN_SI)
        
        if len(f_1) != 0:
            F_w = lens_ampl_function_wave_optics(f_1, M_lz, y, intrp_hyp1f1=intrp_hyp1f1)
            F_g = lens_ampl_factor_geom_optics(f_2, M_lz, y)
            F_f = np.concatenate((F_w,F_g))
            
        else:
            F_f = lens_ampl_factor_geom_optics(f_2, M_lz, y)
            
    return F_f

def lensed_fd_waveform_mine(**params):  
    
    det = params['det']
    time = 1249852257.0
    h_p, h_c = get_fd_waveform(approximant = params['approx'],
                               mass1 = params['mass1'],
                               mass2 = params['mass2'],
                               spin1z = params['spin1z'],
                               spin2z = params['spin2z'],
                               distance = params['dist'], #in Mpc
                               inclination = params['inc'],
                               coa_phase = params['coa_ph'],
                               delta_f = 0.1,
                               f_lower = 20.)

    frequency = h_p.get_sample_frequencies().data
    pos = np.where((frequency>=20) & (frequency<=1024))[0]
    f = frequency[pos]

    hp = h_p[pos]
    hc = h_c[pos]
    
    fp, fc = det.antenna_pattern(params['ra'], params['dec'], params['pol'], time)
    h_u = fp*hp + fc*hc
    
    M_lz = params['M_lz']
    y = params['y']

    shift = FrequencySeries(lens_ampl_function(f, M_lz, y), dtype=np.complex128, delta_f=h_p.delta_f, copy=False)

    hp_l = shift*hp
    hc_l = shift*hc

    h_l = fp*hp_l + fc*hc_l

    return f, h_l

def lensed_fd_waveform_hybrid(**params):   
    
    det = params['det']
    time = 1249852257.0
    h_p, h_c = get_fd_waveform(approximant = params['approx'],
                               mass1 = params['mass1'],
                               mass2 = params['mass2'],
                               spin1z = params['spin1z'],
                               spin2z = params['spin2z'],
                               distance = params['dist'], #in Mpc
                               inclination = params['inc'],
                               coa_phase = params['coa_ph'],
                               delta_f = 0.1,
                               f_lower = 20.)

    frequency = h_p.get_sample_frequencies().data
    pos = np.where((frequency>=20) & (frequency<=1024))[0]
    f = frequency[pos]
    hp = h_p[pos]
    hc = h_c[pos]

    
    fp, fc = det.antenna_pattern(params['ra'], params['dec'], params['pol'], time)
    h_u = fp*hp + fc*hc
    
    M_lz = params['M_lz']
    y = params['y']
    
    shift = FrequencySeries(lens_ampl_function_hybrid(f, M_lz, y, intrp_hyp1f1 = intrp_hyp1f1_hybrid),\
                                                      dtype=np.complex128, delta_f=h_p.delta_f, copy=False)

    hp_l = shift*hp
    hc_l = shift*hc

    h_l = fp*hp_l + fc*hc_l

    return f, h_l

# calculating mismatch; this is a special way of defining mismatch function to calculate fitting factor later:
    
def mismatch_fun(m1_s, m2_s, a1z_s, a2z_s, dL_s, inc_s, ra_s, dec_s, pol_s, coa_ph_s, M_lz, y, detr, f, h):
    
    if m1_s>150:
        return np.exp((m1_s-150)**2)
    elif m1_s<5:
        return np.exp((m1_s-5)**2)
    elif m2_s>150:
        return np.exp((m2_s-150)**2)
    elif m2_s<5:
        return np.exp((m2_s-5)**2)
    
    elif a1z_s>1:
        return np.exp((a1z_s-1)**2)
    elif a1z_s<-1:
        return np.exp((a1z_s+1)**2)
    elif a2z_s>1:
        return np.exp((a2z_s-1)**2)
    elif a2z_s<-1:
        return np.exp((a2z_s+1)**2)
    
    else:
        f_l,h_l = lensed_fd_waveform_mine(approx = 'IMRPhenomPv2', 
                                          mass1 = m1_s,
                                          mass2 = m2_s,
                                          spin1z = a1z_s,
                                          spin2z = a2z_s,
                                          dist = dL_s,
                                          inc = inc_s,
                                          ra = ra_s,
                                          dec = dec_s,
                                          pol = pol_s,
                                          coa_ph = coa_ph_s,
                                          M_lz = M_lz,
                                          y = y,
                                          det = detr)
  
        flen = len(h)
        h_l.resize(flen)
        h.resize(flen)
        
        if detr == Det[0]:
            ligo_psd = np.interp(f, frequency, PSD[0])
            LIGO_PSD = FrequencySeries(ligo_psd, dtype=np.float, delta_f=h_l.delta_f)
            m_H = match(h_l,h,psd=LIGO_PSD,low_frequency_cutoff = 20.)[0] # match b/n un/lensed waveforms
            mm_H = 1.0-m_H 
            return mm_H

        elif detr == Det[1]:
            ligo_psd = np.interp(f, frequency, PSD[1])
            LIGO_PSD = FrequencySeries(ligo_psd, dtype=np.float, delta_f=h_l.delta_f)
            m_L = match(h_l,h,psd=LIGO_PSD,low_frequency_cutoff = 20.)[0] # match b/n un/lensed waveforms
            mm_L = 1.0-m_L 
            return mm_L

        else:
            ligo_psd = np.interp(f, frequency, PSD[2])
            LIGO_PSD = FrequencySeries(ligo_psd, dtype=np.float, delta_f=h_l.delta_f)
            m_V = match(h_l,h,psd=LIGO_PSD,low_frequency_cutoff = 20.)[0] # match b/n un/lensed waveforms
            mm_V = 1.0-m_V 
            return mm_V


M_lz = np.loadtxt(path+'Mlz_y.dat', usecols=(0))
y = np.loadtxt(path+'Mlz_y.dat', usecols=(1))

Ml_y_grid = []
for i in M_lz:
    for j in y:
        Ml_y_grid.append([i,j])

Start = datetime.datetime.now()
print ('starting time is = {}'.format(Start))

m1, m2, a1z, a2z, dL, iota, ra, dec, pol, phi0 = 40, 30, 0.5, 0.5, 500, 0, np.pi/2, 0, 0, 0

M_l = []
y_l = []
mm = []

for i in Ml_y_grid:

    f_l, h_l = lensed_fd_waveform_hybrid(approx ='IMRPhenomPv2',
                                         mass1 = m1,
                                         mass2 = m2,
                                         spin1z = a1z,
                                         spin2z = a2z,
                                         dist = dL,
                                         inc = iota,
                                         ra = ra,
                                         dec = dec,
                                         pol = pol,
                                         coa_ph = phi0,
                                         M_lz = i[0],
                                         y = i[1],
                                         det = Det[0])

    mismatch = mismatch_fun(m1, m2, a1z, a2z, dL, iota, ra, dec, pol, phi0, i[0], i[1], Det[0], f_l, h_l)

    print('{}. M_{} = {}'.format(Ml_y_grid.index(i), np.where(i[0]==M_lz)[0][0], i[0]), ', y_{} = {}'.format(np.where(i[1]==y)[0][0],\
    i[1]), 't_clock = {}'.format(datetime.datetime.now()))
    print('mm = {}'.format(mismatch))

    M_l.append(i[0])
    y_l.append(i[1])
    mm.append(mismatch)

    np.savetxt('mismatch_H.dat', np.transpose((M_l, y_l, mm)), header = 'M_lz   \t\t    y    \t\t    mismatch')

End = datetime.datetime.now()
print ('ending time is = {}'.format(End))

