import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer

"""
n_prec = 11

n_coms = 100


const_traits = np.append(gp.pt.phyto_traits, gp.zt.zoop_traits)
const_traits = np.append(const_traits, ["h_zp"])

fac = 4**np.linspace(-1,1,n_prec)
rich_all = np.empty((len(const_traits), n_prec, n_coms, 2))
add_var = np.linspace(-1,1, n_prec)
for j, trait in enumerate(const_traits):
    if trait in gp.pt.std_phyto.columns:
        add_mean = add_var*gp.pt.std_phyto[trait].values**2
    elif trait in gp.zt.std_zoop.columns:
        add_mean = np.linspace(-1,1, n_prec)*gp.zt.std_zoop[trait].values**2
    elif trait == "h_zp":
        traits = gp.generate_plankton(5,n_coms, evolved_zoop=True)
        std_h_zp = np.std(np.log(traits["h_zp"]))
        add_mean = np.linspace(-1,1, n_prec)*std_h_zp**2
        
    for i in range(n_prec):
        start = timer()
        traits = gp.generate_plankton(5, n_coms, diff_mean = {trait: add_mean[i]})
        env = gp.generate_env(n_coms)
        traits = gp.phytoplankton_equilibrium(traits, env)
        
        rich_all[j,i] = assembly_richness(traits, env, plot_until = 0)
        print(i,j,fac[i], trait, np.mean(rich_all[j,i], axis = 0), timer()-start)

np.save("Data_assembly_mean", rich_all)  
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
    ax[0,0].plot(add_var, np.mean(rich_all[i], axis = 1)[:,0], '.-')
    ax[0,1].plot(add_var, np.mean(rich_all[i], axis = 1)[:,1], '.-', label = trait)
ax[0,1].legend(ncol = 3)
    
for i, trait in enumerate(gp.zt.zoop_traits):
    ax[1,0].plot(add_var, np.mean(rich_all[i + len(gp.pt.phyto_traits)], axis = 1)[:,0], '.-')
    ax[1,1].plot(add_var, np.mean(rich_all[i + len(gp.pt.phyto_traits)], axis = 1)[:,1], '.-',
                 label = trait)
ax[1,1].legend(ncol = 3)


ax[2,0].plot(add_var, np.mean(rich_all[-1], axis = 1)[:,0], '.-')
ax[2,1].plot(add_var, np.mean(rich_all[-1], axis = 1)[:,1], '.-',
             label = "h_zp")
ax[2,1].legend()

ax[2,0].set_xticks([-1,0,1])
ax[2,1].set_xticks([-1,0,1])

perc = np.nanpercentile(np.mean(rich_ref, axis = 1), [5, 50, 95], axis = 0)
for i in range(3):
    for j in range(2):
        ax[i,j].axhline(perc[0,j], color = "red", linestyle = '--')
        ax[i,j].axhline(perc[2,j], color = "red", linestyle = '--')
        ax[i,j].axhline(perc[1,j], color = "red")
        ax[i,j].axvline(0, color = "k")
        
    ax[i,0].set_ylabel("Species richness")
ax[0,0].set_title("Phytoplankton")
ax[0,1].set_title("Zooplankton")

ax[-1,0].set_xlabel("Change in mean")
ax[-1,1].set_xlabel("Change in mean")

for i,a in enumerate(ax.flatten()):
    a.set_title("ABCDEF"[i], loc = "left")

fig.savefig("Figure_assembly_mean.pdf")