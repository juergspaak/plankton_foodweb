import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer

n_prec = 11

n_coms = 100


const_traits = np.append(gp.pt.phyto_traits, gp.zt.zoop_traits)
const_traits = np.append(const_traits, ["s_zp", "h_zp"])

fac = 4**np.linspace(-1,1,n_prec)
rich_all = np.empty((len(const_traits), n_prec, n_coms, 2))
for i in range(n_prec):
    for j, trait in enumerate(const_traits):
        start = timer()
        traits = gp.generate_plankton(5, n_coms, diff_std = {trait: fac[i]})
        env = gp.generate_env(n_coms)
        traits = gp.phytoplankton_equilibrium(traits, env)
        
        rich_all[j,i] = assembly_richness(traits, env, plot_until = 0)
        print(i,j,fac[i], trait, np.mean(rich_all[j,i], axis = 0), timer()-start)
        
#"""
        
"""
rich_ref = np.empty((5, n_coms,2))
for i in range(len(rich_ref)):
    start = timer()
    traits = gp.generate_plankton(5, n_coms)
    env = gp.generate_env(n_coms)
    traits = gp.phytoplankton_equilibrium(traits, env)
    
    rich_ref[i] = assembly_richness(traits, env)
    print(i, np.mean(rich_ref[i], axis = 0), timer()-start)"""
    
rich_ref = rich_all[:,5]
fig, ax = plt.subplots(3,2, sharex = True, sharey = True, figsize = (13,13))

for i, trait in enumerate(gp.pt.phyto_traits):
    ax[0,0].plot(fac, np.mean(rich_all[i], axis = 1)[:,0], '.-')
    ax[0,1].plot(fac, np.mean(rich_all[i], axis = 1)[:,1], '.-', label = trait)
    ax[0,0].semilogx()
ax[0,1].legend(ncol = 3)
    
for i, trait in enumerate(gp.zt.zoop_traits):
    ax[1,0].plot(fac, np.mean(rich_all[i + len(gp.pt.phyto_traits)], axis = 1)[:,0], '.-')
    ax[1,1].plot(fac, np.mean(rich_all[i + len(gp.pt.phyto_traits)], axis = 1)[:,1], '.-',
                 label = trait)
    ax[1,0].semilogx()
ax[1,1].legend(ncol = 3)

for i, trait in enumerate(const_traits[-2:]):
    ax[2,0].plot(fac, np.mean(rich_all[i -2], axis = 1)[:,0], '.-')
    ax[2,1].plot(fac, np.mean(rich_all[i -2], axis = 1)[:,1], '.-',
                 label = trait)
    ax[2,0].semilogx()
ax[2,1].legend()

ax[2,1].set_xticks([0.5,1,2])
ax[2,1].set_xticklabels([0.5, 1, 2])

perc = np.nanpercentile(np.mean(rich_ref, axis = 1), [5, 50, 95], axis = 0)
for i in range(3):
    for j in range(2):
        ax[i,j].axhline(perc[0,j], color = "red", linestyle = '--')
        ax[i,j].axhline(perc[2,j], color = "red", linestyle = '--')
        ax[i,j].axhline(perc[1,j], color = "red")
        
    ax[i,0].set_ylabel("Species richness")
ax[0,0].set_title("Phytoplankton")
ax[0,1].set_title("Zooplankton")

ax[-1,0].set_xlabel("Variance factor")
ax[-1,1].set_xlabel("Variance factor")

fig.savefig("Figure_assembly_variance.pdf")