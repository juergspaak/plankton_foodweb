"""
This file runs simulations where not all resources are limiting or
where zooplankton are not present.
This is achieved by changing the resource requirements or the mortality
rate of zooplankton
"""

import numpy as np

from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer
from itertools import combinations


n_coms = 1000
n_spec = 20


# cases: Each of the three resources can be limiting
# predators can be present or not

cases = ["N", # nitrogen not limiting
         "P", # phosphorus not limiting
         "L", # light not limiting
         "Z"] # zooplankton not present

combs = [""] + list(combinations(cases, 1)) + list(combinations(cases, 2)) + list(combinations(cases,3))
combs.remove(("N", "P", "L")) # remove case where no resource is limiting

    

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"
path = "data/"
for comb in combs:
    print(comb, "new")
    # check whether computation has already been done
    try:
        save = "assembly_non_lim{}_{}_{}.npz".format((len(comb)*"_{}").format(*comb), n_spec, n_coms)
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
    diff_mean = {}
    if "N" in comb:
        diff_mean["c_n"] = -10
        diff_mean["k_n"] = -10
    if "P" in comb:
        diff_mean["c_p"] = -10
        diff_mean["k_p"] = -10
    if "L" in comb:
        diff_mean["a"] = -10
        diff_mean["k_l"] = -10
    if "Z" in comb:
        diff_mean["mu_Z"] = -10
    traits = gp.generate_plankton(n_spec, n_coms, diff_mean = diff_mean)
    env = gp.generate_env(n_coms)
    traits = gp.phytoplankton_equilibrium(traits, env)

    # simulate densities
    start = timer()
    richness, present, res, dens = assembly_richness(
                    traits, env, plot_until = 0, ret_all = True,
                    save = save)
    
    np.savez(save, i = n_coms, present = present, res = res,
             dens = dens, time = timer()-start, **traits, **env )