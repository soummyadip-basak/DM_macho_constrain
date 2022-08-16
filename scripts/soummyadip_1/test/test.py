#!/home1/soummyadip.basak/bilby/ve3/bilby_som/bin/python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams

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

rc('text.latex', preamble='\\usepackage{txfonts}')
rc('text', usetex=True)
rc('font', family='serif')
rc('font', serif='times')
rc('mathtext', default='sf')
rc("lines", markeredgewidth=1)
rc("lines", linewidth=2)

x = np.linspace(-10,10,100)
y = x**2
plt.figure(figsize=(8,6))
plt.plot(x,y, label='test figure')
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('test/test.png')
plt.show()
