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



n_spec_max = 4
n_specs = np.arange(1, n_spec_max+1)
n_coms = int(1e6)

# first generate sufficient communities
traits, env = gp.generate_communities(4, int(n_coms))


trait_combs = np.array([[],["k_Z"], ["s_zp"], ["h_zp"],["c_p", "c_n", "a"],
               ["k_p", "k_n",  "k_l"],["mu_P"],  ["mu_Z"], ["c_Z"],
               ["c_Z",  "k_Z"]], dtype = "object")
rand_trait = np.random.randint(len(trait_combs), size = n_coms)

data = pd.DataFrame(data = np.nan, columns=["trait_comb", "r_phyto", "r_zoo"],
                    index = np.arange(traits["n_coms"]))
data["trait_comb"] = ["-".join(i) for i in trait_combs[rand_trait]]

rel_tresh = 1e-5
def evolve_time(N, ti, envi, time, ret_var = "ind"):
    sol = solve_ivp(lambda t,logN: pg.convert_ode_to_log(logN, ti, envi),
                        time, np.log(N), method = "LSODA")
    # compute new r_spec, based on last 20 percent of time
    dens_phyto = np.exp(sol.y[:ti["r_phyto"],sol.t>0.8*time[-1]])
    dens_zoop = np.exp(sol.y[ti["r_phyto"]:, sol.t>0.8*time[-1]])
    
    # high relative frequency and absolute frequency
    ind_phyto = ((np.sum(dens_phyto, axis = 1)/np.sum(dens_phyto) > rel_tresh)
                  & (np.mean(dens_phyto, axis = 1) > 1e-2))
    ind_phyto = ((np.sum(dens_phyto, axis = 1)/np.sum(dens_phyto) > rel_tresh))
    ind_zoo = ((np.sum(dens_zoop, axis = 1)/np.sum(dens_zoop) > rel_tresh)
               & (np.mean(dens_zoop, axis = 1) > 1e-6))
    
    
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

plot_until = 3

start = timer()
for i in range(traits["n_coms"]):
    ti_org, envi, i = gp.select_i(traits, env, i)
    
    for t_name in trait_combs[rand_trait[i]]:
        if t_name == "s_zp":
            ti_org["s_zp"][:] = 1
        else:
            # take geometric mean
            ti_org[t_name][:] = np.exp(np.mean(np.log(ti_org[t_name])
                                           , keepdims = True))
    
    ind_phyto = ind_zoo = np.arange(ti_org["r_phyto"])
    ti = select_present(ti_org, ind_phyto, ind_zoo)
    r_spec_old = 2*[np.inf]
    time_step = 100
    time = [0]
    N = np.log(ti["r_phyto"]*[100] + ti["r_zoo"]*[1])
    if i < plot_until:
        plt.figure()
    while (ti["r_phyto"] + ti["r_zoo"]<r_spec_old[-2]) or time_step<400:
        time = time[-1] + np.array([0, time_step])
        r_spec_old.append(ti["r_phyto"] + ti["r_zoo"])
        
        # evolve species
        bool_phyto, bool_zoo, sol = evolve_time(np.exp(N), ti, envi, time)
        ind_phyto = ind_phyto[bool_phyto]
        ind_zoo = ind_zoo[bool_zoo]
        if i < plot_until:
            #plot_densities(sol, ti)
            plt.semilogy(sol.t, np.exp(sol.y.T))
        # select species that survived
        ti = select_present(ti_org, ind_phyto, ind_zoo)
        time_step *= 2
        N = sol.y[np.append(bool_phyto, bool_zoo), -1]
    if i<plot_until:
        plt.ylim([1e-5, None])
        plt.show()
    data.loc[i,["r_phyto", "r_zoo"]] = ti["r_phyto"], ti["r_zoo"]
    print(i, np.round([timer()-start, (timer()-start)/(i+1)],2), ti["r_phyto"], ti["r_zoo"])
    if i % 100 == 99:
        data.to_csv("Data_sim_preselect4.csv")
    
data.to_csv("Data_sim_preselect4.csv")
    