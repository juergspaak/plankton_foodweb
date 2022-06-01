import numpy as np
import warnings
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import generate_plankton as gp
from scipy.integrate import solve_ivp
import plankton_growth as pg

import assembly_time_fun as atf


path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/fluctuations/"
save = path + "data_predator_fluctuations.npz"
"""
n_coms = 1000
n_spec = 20

traits = gp.generate_plankton(n_spec, n_coms)
env = gp.generate_env(n_coms)
traits = gp.phytoplankton_equilibrium(traits, env)

try:
    file = np.load(save)
    
except FileNotFoundError:
    richness, present, res, dens = atf.assembly_richness(
                                    traits, env, ret_all = True, save = save)
    np.savez(save, i = n_coms, present = present, res = res,
                     dens = dens, **traits, **env)

time = np.linspace(0,360, 500)

cases = np.zeros((2,2,2,3))
cases[1,:,:,0] = 1
cases[:,1,:,1] = 1
cases[:,:,1,2] = 1
cases.shape = (-1,3)

r_is = np.empty((n_coms, len(cases), 2))
resident_growth = np.empty(r_is.shape)
fluctuations = np.empty((n_coms,3)) # record amount of fluctuations

t_invest = -3
for i in range(n_coms):
    print("second", i)
    # select current community 
    ti, envi, i = gp.select_i(traits, env, i)
    # select traits of species present in second to last step
    tc = atf.select_survivors(ti,
                              np.where(dens[i,0,:,t_invest]>0)[0],
                              np.where(dens[i,1,:,t_invest]>0)[0])
    
    # starting densities
    N_start = np.append(res[i,:2,t_invest],
                        dens[i,...,t_invest][np.isfinite(dens[i,...,t_invest])])
    # simulate densities for another 2 years
    sol = solve_ivp(lambda t,logN: pg.convert_ode_to_log(logN, tc, envi, t),
                        time[[0,-1]], np.log(N_start), method = "LSODA",
                        t_eval = time)
    # convert to normal densities instead of log densities
    sol.y = np.exp(sol.y)
    
    # remove burn in phase]
    densities = np.insert(sol.y,
                          [2+tc["r_phyto"], 2+tc["r_phyto"] + tc["r_zoo"]],
                          np.full((2, len(time)), 0.1), axis = 0)
    
    # check whether last species could invade under different conditions
    tc2 = atf.select_survivors(ti,
                              np.where(present[i,0,:,t_invest])[0],
                              np.where(present[i,1,:,t_invest])[0])
    
    growth = np.empty((8,) + densities.shape)
    
    for i_case, case in enumerate(cases):
        dens_c = densities.copy()
        if case[0]:
            dens_c[:2] = np.mean(dens_c[:2], axis = 1, keepdims=True)
        if case[1]:
            dens_c[2:2+tc2["r_phyto"]] = np.mean(dens_c[2:2+tc2["r_phyto"]]
                                                 , axis = 1, keepdims=True)
        if case[2]:
            dens_c[-tc2["r_zoo"]:] = np.mean(dens_c[-tc2["r_zoo"]:]
                                             , axis = 1, keepdims=True)
        dens_c = np.log(dens_c)
        for step in range(len(time)):
            growth[i_case, :, step] = pg.convert_ode_to_log(dens_c[:,step],
                                                 tc2, envi, time[step])
    # average growth rate of invading species
    r_is[i] = np.mean(growth[:,[1 + tc2["r_phyto"], 1 + tc2["r_phyto"] + tc2["r_zoo"]]]
                      , axis = -1)
    # average growth rate of resident species
    resident_growth[i] = np.mean(growth[:,[2,-2]], axis = -1)
    
    # amount of fluctuations
    flucts = np.std(sol.y, axis = 1)/np.mean(sol.y, axis = 1)
    fluctuations[i] = [np.amax(flucts[:2]),
                       np.amax(flucts[2:2+tc["r_phyto"]]),
                       np.amax(flucts[-tc["r_zoo"]:])]

np.savez(save, i = n_coms, present = present, res = res,
                     dens = dens, **traits, **env, fluctuations = fluctuations,
                     r_is = r_is, resident_growth = resident_growth, cases = cases)
    
#"""
    
