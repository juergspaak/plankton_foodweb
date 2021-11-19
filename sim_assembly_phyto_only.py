import numpy as np

import sys

from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer


n_rep = 6
n_coms = 1000
n_spec = 20

    

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"

for i in range(n_rep):
    print(i, "new")
    # check whether computation has already been done
    try:
        save = "assembly_phyto_only_{}_{}_{}.npz".format(n_spec, n_coms, i)
        
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
    traits = gp.generate_plankton(n_spec, n_coms,
                                  diff_mean = {"mu_Z": -10*gp.zt.std_zoop["mu_Z"]})
    env = gp.generate_env(n_coms)
    traits = gp.phytoplankton_equilibrium(traits, env)

    # simulate densities
    start = timer()
    richness, present, res, dens = assembly_richness(
                    traits, env, plot_until = 2, ret_all = True,
                    save = save)
    
    np.savez(save, i = n_coms, present = present, res = res,
             dens = dens, time = timer()-start, **traits, **env )