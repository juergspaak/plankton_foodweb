import numpy as np
import matplotlib.pyplot as plt


import generate_plankton as gp
from scipy.integrate import solve_ivp
import plankton_growth as pg

def evolve_time(N, ti, envi, time):
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
                      plot_until = 3):
    richness = np.empty((traits["n_coms"],2))
    for i in range(traits["n_coms"]):
        
        if i < plot_until:
            fig, ax = plt.subplots(2, sharex = True)
        ti, envi, i = gp.select_i(traits, env, i)
        ind_phyto = np.array([0])
        ind_zoo = np.array([0])
        N_start = np.array([ti["N_star_P_res"][0], 1])
        time = time_org.copy()
        for j in range(ti["r_phyto"]):
            
            tc = select_survivors(ti, ind_phyto, ind_zoo)
            s_phyto, s_zoo, N_start, sol = evolve_time(N_start, tc,
                                                       envi, time)
            if i < plot_until:
                ax[0].semilogy(sol.t, np.exp(sol.y[:len(s_phyto)].T))
                ax[1].semilogy(sol.t, np.exp(sol.y[len(s_phyto):].T))
                ax[0].set_ylim([1e-5, None])
                ax[1].set_ylim([1e-5, None])
             
            if j < ti["r_phyto"]-1:
                N_start = np.insert(np.exp(N_start),
                                    [sum(s_phyto), sum(s_phyto)+sum(s_zoo)],
                                    [10,0.1])
                
                ind_phyto = np.append(ind_phyto[s_phyto], j+1)
                ind_zoo = np.append(ind_zoo[s_zoo], j+1)
                time = time[-1] + time_org

        
        richness[i] = len(ind_phyto), len(ind_zoo)
    return richness

if __name__ == "__main__":     
    traits, env = gp.generate_communities(4, int(1e4))
    traits = gp.phytoplankton_equilibrium(traits, env)
    print(traits["n_coms"])
    
    richness = assembly_richness(traits, env)

           