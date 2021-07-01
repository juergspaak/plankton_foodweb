import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy

from timeit import default_timer as timer

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import generate_plankton as gp
import plankton_growth as pg


trait_combs = np.array([[],["k_Z"], ["s_zp"], ["h_zp"],["c_p", "c_n", "a", "c_Z"],
               ["k_p", "k_n"], ["mu_P", "mu_Z"]], dtype = "object")
n_coms = 2000*len(trait_combs)
r_phyto = 5
r_zoo = 5
traits = gp.generate_plankton(r_phyto, n_coms, r_zoo)
env = gp.generate_env(n_coms)

rand_trait = np.random.randint(len(trait_combs), size = n_coms)

data = pd.DataFrame(data = np.nan, columns=["trait_comb", "r_phyto", "r_zoo",
                                            "stable", "feas"],
                    index = np.arange(n_coms))
data["trait_comb"] = ["-".join(i) for i in trait_combs[rand_trait]]


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
    
    
    return ind_phyto, ind_zoo, sol.y[np.append(ind_phyto, ind_zoo),-1],sol
    
def plot_densities(sol, ti):
    try:
        plt.semilogy(sol.t, np.exp(sol.y[:ti["r_phyto"]]).T, '-')
        plt.semilogy(sol.t, np.exp(sol.y[ti["r_phyto"]:]).T, '--')
        plt.ylim([1e-5, None])
    except ZeroDivisionError:
                    pass
                
def select_survivors(ti, ind_phyto, ind_zoo):
    for trait in gp.pt.phyto_traits:
                ti[trait] = ti[trait][ind_phyto]
            
    for trait in gp.zt.zoop_traits:
        ti[trait] = ti[trait][ind_zoo]
        
    ti["h_zp"] = ti["h_zp"][ind_zoo]
    ti["h_zp"] = ti["h_zp"][:,ind_phyto]
    ti["s_zp"] = ti["s_zp"][ind_zoo]
    ti["s_zp"] = ti["s_zp"][:,ind_phyto]
    
    ti["r_phyto"] = sum(ind_phyto)
    ti["r_zoo"] = sum(ind_zoo)
    return ti

start = timer()
plot_until = 3
for i in range(n_coms):
    
    ti, envi, i = gp.select_i(traits, env, i)
    for t_name in trait_combs[rand_trait[i]]:
        if t_name == "s_zp":
            traits["s_zp"][:] = 1
        else:
            # take geometric mean
            ti[t_name][:] = np.exp(np.mean(np.log(ti[t_name])
                                           , keepdims = True))
            
    ti_copy = deepcopy(ti)
    envi_copy = deepcopy(envi)
    # initial species density
    N = np.log(np.append(np.full(r_phyto, 100), np.full(r_zoo, 1)))
    
    time = [0]
    
    r_spec_old = 2*[np.inf]
    time_step = 100
    if i < plot_until:
        plt.figure()
    while (ti["r_phyto"] + ti["r_zoo"]<r_spec_old[-2]) or time_step<400:
        time = time[-1] + np.array([0, time_step])
        r_spec_old.append(ti["r_phyto"] + ti["r_zoo"])
        
        # evolve species
        ind_phyto, ind_zoo, N, sol = evolve_time(np.exp(N), ti, envi, time)
        if i < plot_until:
            plot_densities(sol, ti)
        # select species that survived
        ti = select_survivors(ti, ind_phyto, ind_zoo)
        time_step *= 2
    
    point_equi,info,a ,b = fsolve(pg.per_cap_plankton_growth,
                        np.exp(np.mean(sol.y[:,sol.t>0.8*time[-1]], axis = 1)),
                        args = (ti, envi), full_output=True)
    
        
    # Check stability of equilibrium
    # Jacobian of system at equilibrium
    r = np.zeros((r_spec_old[-1], r_spec_old[-1]))
    r[np.triu_indices(r_spec_old[-1])] = info["r"].copy()
    jac = np.diag(point_equi).dot(info["fjac"].T).dot(r)
    try:
        eigval = np.amax(np.real(np.linalg.eigvals(jac)))
    except ValueError:
        eigval = np.nan
        point_equi = np.nan
    
    if i< plot_until:
        plt.plot(np.repeat(time[-1], len(point_equi)), point_equi, 'ko')
    
    # test, have we found a stable community
    print(i, np.round([timer()-start, (timer()-start)/(i+1)],2), data.shape)
    
    
    
    data.loc[i, ["r_phyto", "r_zoo"]] = ti["r_phyto"], ti["r_zoo"]
    data.loc[i, ["stable", "feas"]] = [eigval,
                                       np.amin(point_equi)]
    
    if i %1000 == 999: 
        data.to_csv("data_diversity_timeseries.csv")
        
    """if (ti["r_phyto"] == 1) and (ti["r_zoo"] == 2):
        
        N = np.append(np.full(ti["r_phyto"], 1000),
                      np.full(ti["r_zoo"], 0.1))
        sol = solve_ivp(save_fun,[0,50000], np.log(N),
                                method = "LSODA")
        plt.figure()
        plt.semilogy(sol.t, np.exp(sol.y[:ti["r_phyto"]]).T, '-')
        plt.semilogy(sol.t, np.exp(sol.y[ti["r_phyto"]:]).T, '--')
        plt.ylim([1e-5, None])
        raise
        print(trait_combs[rand_trait[i]])
        ti = deepcopy(ti_copy)
        envi = deepcopy(envi_copy)
        #i = 0
        N = np.append(np.full(r_phyto, 10000), np.full(r_zoo, 0.1))
    
        time = [0]
        i = 0
        raise"""
        #"""

    