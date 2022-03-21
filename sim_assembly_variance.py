"""
This file runs simulations with altered variance for phytoplankton and zooplankton
traits
"""


import numpy as np
from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer


const_traits = np.append(gp.pt.phyto_traits[1:], gp.zt.zoop_traits[1:])
const_traits = np.append(const_traits, ["h_zp", "s_zp"])
n_prec = 9
n_coms = 1000
n_spec = 20

    

all_start = timer()
fac = 2**np.linspace(-1,1,n_prec)

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"

for i in range(n_prec):
    for trait in const_traits:
        print(i, trait)
        # check whether computation has already been done
        try:
            save = "assebly_var_{}_{}_{}_{}.npz".format(n_spec, n_coms, trait, i)
            
            data = np.load(path + save)
            
            continue
        except FileNotFoundError:
            pass
        save = "data/" + save
        
        # is file already being created?
        try:
            data = np.load(save)
            continue
        except FileNotFoundError:
            # file is not already being created, start process
            np.savez(save, prelim = 1)
        
    
        # generate altered traits
        traits = gp.generate_plankton(n_spec, n_coms, diff_std = {trait: fac[i]})
        env = gp.generate_env(n_coms)
        traits = gp.phytoplankton_equilibrium(traits, env)

        # simulate densities
        start = timer()
        richness, present, res, dens = assembly_richness(
                        traits, env, plot_until = 0, ret_all = True,
                        save = save)
        
        np.savez(save, change_var = fac[i], i = n_coms, present = present, res = res,
                 dens = dens, time = timer()-start, **traits, **env )