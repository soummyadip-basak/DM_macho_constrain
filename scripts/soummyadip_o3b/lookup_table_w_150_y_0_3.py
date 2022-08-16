#!/usr/bin/env python

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

#w0 = lal.MTSUN_SI*20*8*np.pi     # the minimum value of w corresponding to M_lz = 1 M_solar at frequency 20 Hz
w0 = 0.1
w1 = 150

w = np.linspace(w0, w1, int((w1-w0)/0.1))
y = np.linspace(0.01,3,1500)

def calc_hyp1f1(y):
    """ evaluate the hypergeomtric fun used in the lensing magnification
    w needs to be globally defined """

    hyp_fn = np.vectorize(mpmath.hyp1f1)(0.5*w*1j, 1., 0.5*w*y**2*1j, maxterms=1e7)
    hyp_fn = np.array(hyp_fn.tolist(), dtype=complex)
    print('hyp_fn = {}'.format(hyp_fn))
    return hyp_fn

N_cores = 10
t0 = time.time()
p = Pool(N_cores)
hyp_fn = p.map(calc_hyp1f1, y)
p.close()
t1 = time.time()
print('time taken %f s' %(t1-t0))

# reshape to a 2,2 array  

N_grid_y, N_grid_w = len(y), len(w)
hyp_fn = np.reshape(np.ravel(hyp_fn), (N_grid_y, N_grid_w))
outfile = 'hyp1f1_data_y_3_w_150_Ngrid_%dx%d.npz'%(N_grid_y, N_grid_w)
np.savez(outfile, w=w, y=y, hyp_fn=hyp_fn)

D = np.load(outfile, allow_pickle=True)
w = D['w']
y = D['y']
hyp_fn = D['hyp_fn']

# interpolate the data 
log_abs_hyp_fn = interp2d(w, y, np.log10(abs(hyp_fn)), fill_value=0)
arg_hyp_fn = interp2d(w, y, np.unwrap(np.angle(hyp_fn)), fill_value=0)

# pickle and save the interpolation object 
hyp_fn_intrp = {'log_abs':log_abs_hyp_fn, 'arg':arg_hyp_fn}

file_name = 'intrp_hyp1f1_data_y_3_w_150_Ngrid_%dx%d.pkl'%(N_grid_y, N_grid_w)
open_file = open(file_name, "wb")
pickle.dump(hyp_fn_intrp, open_file)
open_file.close()

