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
const_traits = np.append(const_traits, ["h_zp"])

rich_all = np.empty((len(const_traits), n_prec, n_coms, 2))
res_all = np.full((len(const_traits), n_prec, n_coms, 3), np.nan)
dens_all = np.full((len(const_traits), n_prec, n_coms, 2), np.nan)

add_var = np.linspace(-1,1, n_prec)
all_start = timer()
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
        
        # simulate densities
        richness, id_survive, res_equi, dens_equi = assembly_richness(
                        traits, env, plot_until = 0, ret_all = True)
        rich_all[j,i] = richness
        res_all[j,i] = res_equi
        dens_all[j,i] = np.nansum(dens_equi, axis = -1)
        
        print(i,j,add_var[i], trait, np.mean(rich_all[j,i], axis = 0),
              timer()-start, timer()-all_start)

    np.savez("Data_assembly_mean2", rich = rich_all, res = res_all,
             dens = np.log(dens_all),
         traits = const_traits, change_traits = add_var)
