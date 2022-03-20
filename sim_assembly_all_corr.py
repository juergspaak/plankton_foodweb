import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer

n_prec = 11

n_coms = 500


exp = 10**np.linspace(-1,1,n_prec)
corr_phyto = gp.pt.corr_phyto.values
corr_phyto = np.sign(corr_phyto)*np.abs(corr_phyto)**exp.reshape(-1,1,1)
corr_phyto[:, np.arange(len(gp.pt.corr_phyto)), np.arange(len(gp.pt.corr_phyto))] = 1


rich_phyto = np.full((n_prec, n_coms, 2), np.nan)
res_phyto = np.full((n_prec, n_coms, 3), np.nan)
dens_phyto = np.full((n_prec, n_coms, 2), np.nan)
tot_start = timer()
for i in range(n_prec):
    start = timer()
    traits = gp.generate_plankton(5, n_coms, corr_phyto = corr_phyto[i])
    env = gp.generate_env(n_coms)
    traits = gp.phytoplankton_equilibrium(traits, env)
    
    # simulate densities
    richness, id_survive, res_equi, dens_equi = assembly_richness(
                    traits, env, plot_until = 0, ret_all = True)
    rich_phyto[i] = richness
    res_phyto[i] = res_equi
    dens_phyto[i] = np.sum(dens_equi, axis = -1)
    print(i,np.mean(rich_phyto[i], axis = 0), timer()-start, timer()-tot_start)
    dens_phyto[dens_phyto == 0] = np.nan
np.savez("Data_assembly_all_corr_phyto", rich = rich_phyto,
         P = res_phyto[...,0], N = res_phyto[...,1], L = res_phyto[...,2],
         dens = np.log(dens_phyto), change_var = corr_phyto, exp = exp)


exp = 10**np.linspace(-0.1,1.5,n_prec)
corr_zoo = gp.zt.corr_zoop.values
corr_zoo = np.sign(corr_zoo)*np.abs(corr_zoo)**exp.reshape(-1,1,1)
corr_zoo[:, np.arange(len(gp.zt.corr_zoop)), np.arange(len(gp.zt.corr_zoop))] = 1

rich_zoo = np.full((n_prec, n_coms, 2), np.nan)
res_zoo = np.full((n_prec, n_coms, 3), np.nan)
dens_zoo = np.full((n_prec, n_coms, 2), np.nan)

for i in range(len(rich_zoo)):
    start = timer()
    traits = gp.generate_plankton(5, n_coms, corr_zoo = corr_zoo[i])
    env = gp.generate_env(n_coms)
    traits = gp.phytoplankton_equilibrium(traits, env)
    
    richness, id_survive, res_equi, dens_equi = assembly_richness(
                    traits, env, plot_until = 0, ret_all = True)
    rich_zoo[i] = richness
    res_zoo[i] = res_equi
    dens_zoo[i] = np.sum(dens_equi, axis = -1)
    print(i, np.mean(rich_zoo[i], axis = 0), timer()-start)

#"""
np.savez("Data_assembly_all_corr_zoo", rich = rich_zoo,
         P = res_zoo[...,0], N = res_zoo[...,1], L = res_zoo[...,2],
         dens = np.log(dens_zoo), change_var = corr_zoo, exp = exp)

"""
fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (13,13))

x_phyto = np.linalg.det(corr_phyto)
x_phyto = np.amax(np.linalg.eigvalsh(corr_phyto), axis = -1)/len(gp.pt.corr_phyto)

for i in range(5):
    ind = np.random.rand(n_coms) > 2/3
    ax[0,0].plot(x_phyto,
                 np.mean(rich_phyto[:,ind], axis = 1)[:,0], '.-')
    ax[1,0].plot(x_phyto,
                 np.mean(rich_zoo[:,ind], axis = 1)[:,1], '.-')
    
    x_zoo = np.linalg.det(corr_zoo)
    x_zoo = np.amax(np.linalg.eigvalsh(corr_zoo), axis = -1)/len(gp.zt.corr_zoop)
    ax[0,1].plot(x_zoo,
                 np.mean(rich_phyto[:,ind], axis = 1)[:,0], '.-')
    ax[1,1].plot(x_zoo,
                 np.mean(rich_zoo[:,ind], axis = 1)[:,1], '.-')
    
ax[0,0].axvline(np.amax(np.linalg.eigvalsh(gp.pt.corr_phyto), axis = -1)/len(gp.pt.corr_phyto), color = "k")
ax[1,0].axvline(np.amax(np.linalg.eigvalsh(gp.pt.corr_phyto), axis = -1)/len(gp.pt.corr_phyto), color = "k")

ax[0,1].axvline(np.amax(np.linalg.eigvalsh(gp.zt.corr_zoop), axis = -1)/len(gp.zt.corr_zoop), color = "k")
ax[1,1].axvline(np.amax(np.linalg.eigvalsh(gp.zt.corr_zoop), axis = -1)/len(gp.zt.corr_zoop), color = "k")


ax[1,0].set_xlabel("Max eigvalue")
ax[1,1].set_xlabel("max eigvalue")

ax[1,0].set_ylabel("Phyto richness")
ax[0,0].set_ylabel("Zoop richness")
ax[0,0].set_title("Phytoplankton corr")
ax[0,1].set_title("Zooplankton corr")

fig.savefig("Figure_assembly_all_corr.pdf")
"""