import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from timeit import default_timer as timer

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import generate_plankton as gp
import plankton_growth as pg


trait_combs = np.array([[],["k_Z"], ["s_zp"], ["h_zp"],["c_p", "c_n", "a", "c_Z"],
               ["k_p", "k_n"], ["mu_P", "mu_Z"]])
n_coms = 1000*len(trait_combs)
r_phyto = 5
r_zoo = 5
traits = gp.generate_plankton(r_phyto, n_coms, r_zoo)
env = gp.generate_env(n_coms)

richness = np.empty((n_coms, 2))
start = timer()



rand_trait = np.random.randint(len(trait_combs), size = n_coms)

data = pd.DataFrame(data = np.nan, columns=["trait_comb", "r_phyto", "r_zoo"],
                    index =np.arange(n_coms))

for i in range(n_coms):
    
    ti, envi, i = gp.select_i(traits, env, i)
    for t_name in trait_combs[rand_trait[i]]:
        if t_name == "s_zp":
            traits["s_zp"][:] = 1
        else:
            # take geometric mean
            ti[t_name][:] = np.exp(np.mean(np.log(ti[t_name])
                                           , keepdims = True))
    # initial species density
    N = np.append(np.full(r_phyto, 100), np.full(r_zoo, 1))
    
    time = [0]
    if i<3:
        fig = plt.figure()
    for time_step in [100, 1000]:
        r_spec_old = np.inf
        while ti["r_phyto"] + ti["r_zoo"]<r_spec_old:
            time = time[-1] + np.array([0, time_step])
            r_spec_old = ti["r_phyto"] + ti["r_zoo"]
            def save_fun(t, logN, start = timer()):
                #if timer()-start> 2:
                #    raise RuntimeError
                return pg.convert_ode_to_log(logN, ti, envi)
            
            try:
                sol = solve_ivp(save_fun,time, np.log(N),
                                method = "LSODA")
            except RuntimeError:
                print("runtime")
                sol.t = np.full(2, np.nan)
                sol.y = np.full((len(N),2), np.nan)
            
            if i<3:
                try:
                    plt.semilogy(sol.t, np.exp(sol.y[:ti["r_phyto"]]).T, '-')
                    plt.semilogy(sol.t, np.exp(sol.y[ti["r_phyto"]:]).T, '--')
                except ZeroDivisionError:
                    pass
            
            # compute new r_spec, based on last 20 percent of time
            dens_phyto = np.exp(sol.y[:ti["r_phyto"],sol.t>0.8*time[-1]])
            dens_zoop = np.exp(sol.y[ti["r_phyto"]:, sol.t>0.8*time[-1]])
            
            # high relative frequency and absolute frequency
            ind_phyto = ((np.sum(dens_phyto, axis = 1)/np.sum(dens_phyto) > 1e-3)
                          & (np.mean(dens_phyto, axis = 1) > 1e-2))
            ind_zoo = ((np.sum(dens_zoop, axis = 1)/np.sum(dens_zoop) > 1e-3)
                       & (np.mean(dens_zoop, axis = 1) > 1e-2))
            
            for trait in gp.pt.trait_names:
                ti[trait] = ti[trait][ind_phyto]
            
            for trait in gp.zt.zoop_traits:
                ti[trait] = ti[trait][ind_zoo]
                
            ti["h_zp"] = ti["h_zp"][ind_zoo]
            ti["h_zp"] = ti["h_zp"][:,ind_phyto]
            ti["s_zp"] = ti["s_zp"][ind_zoo]
            ti["s_zp"] = ti["s_zp"][:,ind_phyto]
            
            ti["r_phyto"] = sum(ind_phyto)
            ti["r_zoo"] = sum(ind_zoo)
            
            r_spec = ti["r_phyto"] + ti["r_zoo"]
            N = np.exp(sol.y[:,-1][np.append(ind_phyto, ind_zoo)])
    plt.ylim([1e-5, None])
    print(i, np.round([timer()-start, (timer()-start)/(i+1)],2))
    
    richness[i] = ti["r_phyto"], ti["r_zoo"]
    
richenss = np.array(richness)    
    
      
    