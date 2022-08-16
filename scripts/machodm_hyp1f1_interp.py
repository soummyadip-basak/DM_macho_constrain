
# coding: utf-8

# # Create an interpolant of the confluent hypergeometric function used in the lensing magnification 

# ## Preamble

# In[65]:

from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import scipy
from  scipy.integrate import quad
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


# In[2]:


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


# In[3]:


def calc_hyp1f1(y_l):
    """ evaluate the hypergeomtric fun used in the lensing magnification
    w needs to be globally defined """
    
    hyp_fn = np.vectorize(mpmath.hyp1f1)(0.5*w*1j, 1., 0.5*w*y_l**2*1j, maxterms=1e6)
    return np.array(hyp_fn.tolist(),dtype=complex)


# # Evaluate the hypergeometric function over a grid  

# In[4]:


N_cores = 24 

# create grid of w and y. Note that w is a global variable 
N_grid_w, N_grid_y = int(1e4), int(3e3)

w = np.linspace(0.1, 1200, N_grid_w)
y = np.linspace(0.01, 6, N_grid_y)

t0 = time.time()
p = Pool(N_cores)
hyp_fn = p.map(calc_hyp1f1, y) 
p.close()  
t1 = time.time()
print('time taken %f s' %(t1-t0))

# reshape to a 2,2 array  
hyp_fn = np.reshape(np.ravel(hyp_fn), (N_grid_y, N_grid_w))


# In[12]:


plt.figure(figsize=(13,5))
plt.subplot(121)
plt.pcolormesh(w, y, np.log10(abs(hyp_fn)), cmap='Reds')
plt.xlabel('$w$'); plt.ylabel('$y$')
plt.title('abs(1h1)')
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(w, y, np.unwrap(np.angle(hyp_fn)), cmap='Reds')
plt.xlabel('$w$'); plt.ylabel('$y$')
plt.title('arg(1h1)')
plt.colorbar()
plt.savefig('hyp1f1_data_Ngrid%dx%d.png' %(N_grid_y, N_grid_w), dpi=600)
plt.show()


# In[ ]:


# save the data 
outfile = 'hyp1f1_data_Ngrid%dx%d.npz' %(N_grid_y, N_grid_w)
np.savez(outfile, w=w, y=y, hyp_fn=hyp_fn)


# In[88]:


# load the saved data 
D = np.load(outfile, allow_pickle=True)
w = D['w']
y = D['y']
hyp_fn = D['hyp_fn']

# interpolate the data 
log_abs_hyp_fn = interp2d(w, y, np.log10(abs(hyp_fn)), fill_value=0)
arg_hyp_fn = interp2d(w, y, np.unwrap(np.angle(hyp_fn)), fill_value=0)

# pickle and save the interpolation object 
hyp_fn_intrp = {'log_abs':log_abs_hyp_fn, 'arg':arg_hyp_fn}

file_name = 'intrp_hyp1f1_data_Ngrid%dx%d_python2.pkl' %(N_grid_y, N_grid_w)
open_file = open(file_name, "wb")
pickle.dump(hyp_fn_intrp, open_file)
open_file.close()

# evaluate the interpolated function 
open_file = open(file_name, "rb")
intrp_hyp1f1 = pickle.load(open_file)
open_file.close()
hyp_fn_intp = 10**intrp_hyp1f1['log_abs'](w, y)*np.exp(1j*intrp_hyp1f1['arg'](w,y))


# In[71]:


plt.figure(figsize=(13,5))
plt.subplot(121)
plt.pcolormesh(w, y, np.log10(abs(hyp_fn_intp)), cmap='Reds')
plt.xlabel('$w$'); plt.ylabel('$y$')
plt.title('abs(1h1) intp')
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(w, y, np.unwrap(np.angle(hyp_fn_intp)), cmap='Reds')
plt.xlabel('$w$'); plt.ylabel('$y$')
plt.title('arg(1h1) intp')
plt.colorbar()
plt.savefig('hyp1f1_data_Ngrid%dx%d.png' %(N_grid_y, N_grid_w), dpi=600)
plt.show()


# In[77]:


# check the max error in the interpolation 
print (np.max(abs(hyp_fn-hyp_fn_intp)))


# In[81]:





# In[85]:





# In[ ]:




