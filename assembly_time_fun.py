import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import warnings

import generate_plankton as gp
from scipy.integrate import solve_ivp
import plankton_growth as pg

def evolve_time(N, ti, envi, time):
    sol = solve_ivp(lambda t,logN: pg.convert_ode_to_log(logN, ti, envi),
                        time, np.log(N), method = "LSODA")
    # compute new r_spec, based on last 20 percent of time
    p = 0.5
    t_cutoff = p*time[0] + (1-p)*time[-1]
    with warnings.catch_warnings(record = True):
        dens_phyto = np.exp(sol.y[2:(2+ti["r_phyto"]),sol.t>t_cutoff])
        dens_zoop = np.exp(sol.y[2+ti["r_phyto"]:, sol.t>t_cutoff])
    
    # both phytoplankton and zooplankton might be died out, avoid zero division
    with warnings.catch_warnings(record = True):
        # high relative frequency and absolute frequency
        ind_phyto = ((np.sum(dens_phyto, axis = 1)/np.sum(dens_phyto) > 1e-3)
                      & (np.median(dens_phyto, axis = 1) > 1e-2))
        ind_zoo = ((np.sum(dens_zoop, axis = 1)/np.sum(dens_zoop) > 1e-3)
                   & (np.median(dens_zoop, axis = 1) > 1e-2))
    
    
    return (ind_phyto, ind_zoo,
                sol.y[np.concatenate(([True, True], ind_phyto, ind_zoo)),-1],sol)

def select_survivors(ti, ind_phyto, ind_zoo):
    tc = {}
    for trait in gp.pt.phyto_traits:
        tc[trait] = ti[trait][ind_phyto]
            
    for trait in gp.zt.zoop_traits:
        tc[trait] = ti[trait][ind_zoo]
        
    tc["h_zp"] = ti["h_zp"][ind_zoo]
    tc["h_zp"] = tc["h_zp"][:,ind_phyto]
    tc["s_zp"] = ti["s_zp"][ind_zoo]
    tc["s_zp"] = tc["s_zp"][:,ind_phyto]
    
    tc["r_phyto"] = len(ind_phyto)
    tc["r_zoo"] = len(ind_zoo)
    return tc


def assembly_richness(traits, env, time_org = np.array([0, 365]),
                      plot_until = 0, ret_all = True):
    richness = np.empty((traits["n_coms"],2))
    colors = viridis(np.linspace(0,1,traits["r_phyto"]))
    id_survive = np.full((traits["n_coms"], 2, traits["r_phyto"]),
                         False, dtype = bool)
    dens_equi = np.full(id_survive.shape, np.nan)
    res_equi = np.full((traits["n_coms"], 3), np.nan)
    for i in range(traits["n_coms"]):
        if i < plot_until:
            fig, ax = plt.subplots(3, sharex = True)
            ax[0].set_title(i)
        ti, envi, i = gp.select_i(traits, env, i)
        species_order = np.argsort(np.random.rand(ti["r_phyto"]))
        #species_order = np.sort(species_order)
        ind_phyto = species_order[[0]]
        ind_zoo = species_order[[0]]
        N_start = np.array([envi["P"], envi["N"],
                            ti["N_star_P_res"][species_order[0]], 1])
        N_start[np.isnan(N_start)] = 1
        time = time_org.copy()
        
        for j in range(ti["r_phyto"]):
            
            tc = select_survivors(ti, ind_phyto, ind_zoo)
            s_phyto, s_zoo, N_start, sol = evolve_time(N_start, tc,
                                                       envi, time)
            
            if i < plot_until:
                # plot resources
                ax[0].semilogy(sol.t, np.exp(sol.y[0].T), 'r')
                ax[0].semilogy(sol.t, np.exp(sol.y[1].T), 'b')
                
                for k, ind in enumerate(ind_phyto):
                    ax[1].semilogy(sol.t, np.exp(sol.y[2+k]),
                                   '-' if s_phyto[k] else '--',
                                   alpha = 0.3+s_phyto[k]/2,
                                   color = colors[species_order[ind]])
                for k, ind in enumerate(ind_zoo):
                    ax[2].semilogy(sol.t, np.exp(sol.y[2 + len(ind_phyto) + k]),
                                   '-' if s_zoo[k] else '--',
                                   alpha = 0.3+s_zoo[k]/2,
                                   color = colors[species_order[ind]])
                
            ind_phyto = ind_phyto[s_phyto]
            ind_zoo = ind_zoo[s_zoo]
            if j < ti["r_phyto"]-1:
                N_start = np.insert(np.exp(N_start),
                                    [2+sum(s_phyto), 2+sum(s_phyto)+sum(s_zoo)],
                                    [10,0.1])
                
                ind_phyto = np.append(ind_phyto, species_order[j+1])
                ind_zoo = np.append(ind_zoo, species_order[j+1])
                time = time[-1] + time_org
                

        # save results of assembly
        richness[i] = len(ind_phyto), len(ind_zoo)
        id_survive[i,0,ind_phyto] = True
        id_survive[i,1,ind_zoo] = True
        
        p = .5
        t_cut = sol.t>p*time[0] + (1-p)*time[-1]
        sol.y = sol.y[np.concatenate(([True, True], s_phyto, s_zoo))]
        sol.y = sol.y[:, t_cut]
        
        # all species might have died out, mean of empty slice
        with warnings.catch_warnings(record = True):
            mean_dens = np.mean(np.exp(sol.y), axis = -1)
        res_equi[i,:2] = mean_dens[:2]
        dens_equi[i,0, ind_phyto] = mean_dens[2:(2 + len(ind_phyto))]
        dens_equi[i,1, ind_zoo] = mean_dens[(2 + len(ind_phyto)):]
        
        I_out = envi["I_in"]*np.exp(-envi["zm"]*
               np.sum(ti["a"][ind_phyto,np.newaxis]*
                      np.exp(sol.y[2:(2+len(ind_phyto))]), axis = 0))
        res_equi[i,-1] = np.mean(I_out)
        if i < plot_until:
            
            ax[1].set_ylim([1e-5, min(1e10, ax[1].get_ylim()[1])])
            ax[2].set_ylim([1e-5, min(1e10, ax[1].get_ylim()[1])])
            ax[1].set_title(str(ind_phyto))
            ax[2].set_title(str(ind_zoo))
            plt.show()
    if ret_all:
        return richness, id_survive, res_equi, dens_equi
        
    return richness

if __name__ == "__main__":     
    n_com = 100
    traits = gp.generate_plankton(5, n_com)
    env = gp.generate_env(n_com)
    traits = gp.phytoplankton_equilibrium(traits, env)
    
    richness, id_survive, res_equi, dens_equi = assembly_richness(traits, env,
                        time_org = np.array([0,360]))
    print(np.mean(richness, axis = 0))

           