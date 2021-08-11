import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

import generate_plankton as gp

n_prec = 15
var = 2**np.linspace(-2,2,n_prec)

n_coms = 1000
for i in range(n_prec):
    traits = gp.generate_plankton(3, n_coms, diff_std={"s_zp": var[i]})
    
    
fig, ax = plt.subplots(3,2, figsize = (7,7))
ax0 = ax[0,0].twiny()
ax1 = ax[1,0].twiny()
ax2 = ax[2,0].twiny()

for fac in [0.25, 1, 4]:
    traits = gp.generate_plankton(3, n_coms, diff_std={"s_zp": fac,
                                                       "h_zp": fac})
    ax[0,0].hist(traits["s_zp"].flatten(), bins = np.linspace(0,1,30),
                 label = fac, alpha = 0.3)
    ax0.plot(fac, np.sum(traits["s_zp"]>0.95), 'ro')
    
    ax[0,1].hist(np.log(traits["h_zp"].flatten()), bins = 30,
                 label = fac, alpha = 0.3)
    
    traits = gp.generate_plankton(3, n_coms, diff_std={"size_P": fac})
    ax[1,0].hist(traits["s_zp"].flatten(), bins = np.linspace(0,1,30),
                 label = fac, alpha = 0.3)
    ax1.plot(fac, np.sum(traits["s_zp"]>0.95), 'ro')
    ax[1,1].hist(np.log(traits["h_zp"].flatten()), bins = 30,
                 label = fac, alpha = 0.3)
    
    traits = gp.generate_plankton(3, n_coms, diff_std={"size_Z": fac})
    ax[2,0].hist(traits["s_zp"].flatten(), bins = np.linspace(0,1,30),
                 label = fac, alpha = 0.3)
    ax2.plot(fac, np.sum(traits["s_zp"]>0.95), 'ro')
    ax[2,1].hist(np.log(traits["h_zp"].flatten()), bins = 30,
                 label = fac, alpha = 0.3)
ax[0,0].legend()
ax[0,1].legend()

fig.tight_layout()

fig.savefig("Figure_understanding_combined_traits.pdf")