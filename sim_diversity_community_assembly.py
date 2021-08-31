import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis as cmap
import pandas as pd
from copy import deepcopy

from timeit import default_timer as timer

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import generate_plankton as gp
import plankton_growth as pg


n_coms = 100
r_phyto = 30
r_zoo = 20
colors = cmap(np.linspace(0,1,r_phyto))
traits = gp.generate_plankton(r_phyto, n_coms, r_zoo)
env = gp.generate_env(n_coms)

n_rerun = 5
species_arrival_order = np.argsort(np.random.rand(n_rerun, r_phyto))

def evolve_time(N, ti, envi, time, ret_var = "ind"):
    sol = solve_ivp(lambda t,logN: pg.convert_ode_to_log(logN, ti, envi),
                        time, np.log(N), method = "LSODA")
    # compute new r_spec, based on last 20 percent of time
    dens_phyto = np.exp(sol.y[:ti["r_phyto"],sol.t>0.8*time[-1]])
    dens_zoop = np.exp(sol.y[ti["r_phyto"]:, sol.t>0.8*time[-1]])
    
    # high relative frequency and absolute frequency
    ind_phyto = ((np.sum(dens_phyto, axis = 1)/np.sum(dens_phyto) > 1e-3)
                  & (np.mean(dens_phyto, axis = 1) > 1e-2))
    ind_phyto = ((np.sum(dens_phyto, axis = 1)/np.sum(dens_phyto) > 1e-3))
    ind_zoo = ((np.sum(dens_zoop, axis = 1)/np.sum(dens_zoop) > 1e-3)
               & (np.mean(dens_zoop, axis = 1) > 1e-2))
    
    
    return ind_phyto, ind_zoo,sol
    
def plot_densities(sol, ax, id_phyto, id_zoo):
    for i, color in enumerate(id_phyto):
        ax[0].semilogy(sol.t, np.exp(sol.y[i]), color = colors[color])
    for i, color in enumerate(id_zoo):
        ax[1].semilogy(sol.t, np.exp(sol.y[len(id_phyto) + i]),
                       color = colors[color])
                
def select_present(ti, ind_phyto, ind_zoo):
    t_year = {}
    for trait in gp.pt.phyto_traits:
                t_year[trait] = ti[trait][ind_phyto]
            
    for trait in gp.zt.zoop_traits:
        t_year[trait] = ti[trait][ind_zoo]
        
    t_year["h_zp"] = ti["h_zp"][ind_zoo]
    t_year["h_zp"] = t_year["h_zp"][:,ind_phyto]
    t_year["s_zp"] = ti["s_zp"][ind_zoo]
    t_year["s_zp"] = t_year["s_zp"][:,ind_phyto]
    
    t_year["r_phyto"] = len(ind_phyto)
    t_year["r_zoo"] = len(ind_zoo)
    return t_year

start = timer()
plot_until = 3
time_length = 365 # simulate one year
n_initial = 1

diversity = np.empty((n_coms, n_rerun, r_phyto,2))

start = timer()
for i in range(n_coms):
    ti, envi, i = gp.select_i(traits, env, i)
    
    for j in range(n_rerun):
        
        # initial species density
        N = np.log(np.append(np.full(n_initial, 100),
                             np.full(n_initial, 1)))
        id_phyto = id_zoo = species_arrival_order[j,[0]]
        #fig, ax = plt.subplots(2,1)
        for year in range(r_phyto):
            
            
            t_year = select_present(ti, id_phyto, id_zoo)
            
            time = year*time_length + np.array([0,time_length])
            
                
            # evolve species
            ind_phyto, ind_zoo, sol = evolve_time(np.exp(N), t_year, envi,
                                                     time)
            #plot_densities(sol, ax, id_phyto, id_zoo)
            diversity[i,j,year] = sum(ind_phyto), sum(ind_zoo)
            try:
                id_phyto = np.append(id_phyto[ind_phyto],
                                     species_arrival_order[j,year + 1])
                id_zoo = np.append(id_zoo[ind_zoo],
                                     species_arrival_order[j,year + 1])
            except IndexError:
                pass # last iteration year+1 index does not exist
            N = sol.y[:,-1]
            N[~np.append(ind_phyto, ind_zoo)] = np.nan
            N = np.insert(sol.y[:,-1], [t_year["r_phyto"],
                              t_year["r_phyto"]+ t_year["r_zoo"]],
                          np.log([1, 1e-2]))
            N = N[np.isfinite(N)]
        print(i,j, timer()-start, (timer()-start)/(1+j+i*(n_rerun)))
        #ax[0].set_ylim([1e-5, None])
        #ax[1].set_ylim([1e-5, None])


fig = plt.figure(figsize = (9,9))

ax = fig.add_subplot(2,1,1)
plt.plot(np.mean(diversity[...,0], axis = (0,1)), 'b')
sigmas = 1
plt.plot(np.mean(diversity[...,0], axis = (0,1))
         + sigmas*np.var(diversity[...,0], axis = (0,1)), 'b--')
plt.plot(np.mean(diversity[...,0], axis = (0,1)), 'b')
plt.plot(np.mean(diversity[...,0], axis = (0,1))
         - sigmas*np.var(diversity[...,0], axis = (0,1)), 'b--')


sigmas = 1
plt.plot(np.mean(diversity[...,1], axis = (0,1))
         + sigmas*np.var(diversity[...,1], axis = (0,1)), 'r--')
plt.plot(np.mean(diversity[...,1], axis = (0,1)), 'r')
plt.plot(np.mean(diversity[...,1], axis = (0,1))
         - sigmas*np.var(diversity[...,1], axis = (0,1)), 'r--')
plt.xlabel("Year")
plt.ylabel("species richness")

ax = fig.add_subplot(2,2,3)
years = [1,5,15,-1]
ax.hist(diversity[:,:,years,0].reshape(-1, len(years)))

ax = fig.add_subplot(2,2,4)
ax.hist(diversity[:,:,years,1].reshape(-1, len(years)))

fig.tight_layout()
fig.savefig("Figure_community_assembly.pdf")