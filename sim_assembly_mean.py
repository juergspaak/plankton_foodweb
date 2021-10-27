import numpy as np

import sys

from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer


const_traits = np.append(gp.pt.phyto_traits[1:], gp.zt.zoop_traits[1:])
const_traits = np.append(const_traits, ["h_zp"])
n_prec = 9
n_coms = 1000
n_spec = 20

    
add_var = np.linspace(-1,1, n_prec)
all_start = timer()

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"

for i in range(n_prec):
    for trait in const_traits:
        print(i, trait)
        # check whether computation has already been done
        try:
            save = "assebly_mean_{}_{}_{}_{}.npz".format(n_spec, n_coms, trait, i)
            
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
        
        
        # do actual computation
        if trait in gp.pt.std_phyto.columns:
            add_mean = add_var*gp.pt.std_phyto[trait].values
        elif trait in gp.zt.std_zoop.columns:
            add_mean = add_var*gp.zt.std_zoop[trait].values
        elif trait == "h_zp":
            traits = gp.generate_plankton(5,n_coms, evolved_zoop=True)
            std_h_zp = np.std(np.log(traits["h_zp"]))
            add_mean = add_var*std_h_zp
    
        # generate altered traits
        traits = gp.generate_plankton(n_spec, n_coms, diff_mean = {trait: add_mean[i]})
        env = gp.generate_env(n_coms)
        traits = gp.phytoplankton_equilibrium(traits, env)

        # simulate densities
        start = timer()
        richness, present, res, dens = assembly_richness(
                        traits, env, plot_until = 0, ret_all = True,
                        save = save)
        
        np.savez(save, change_mean = add_mean[i], i = n_coms, present = present, res = res,
                 dens = dens, time = timer()-start, **traits, **env )