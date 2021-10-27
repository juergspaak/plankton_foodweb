import numpy as np
import pandas as pd
from scipy.optimize import brentq
from assembly_time_fun import assembly_richness
from timeit import default_timer as timer

import generate_plankton as gp

n_coms = int(1000)
n_spec = 20

def fun(val,tri,trj,A):
    A.loc[tri, trj] = val
    A.loc[trj, tri] = val
    return np.amin(np.linalg.eigvalsh(A))

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"

for i, key in enumerate(gp.pt.phyto_traits[1:]):
    
    for j,keyj in enumerate(gp.pt.phyto_traits[1:]):
        print(i,j, key, keyj)
        if key == keyj:
            continue
        
        # is file already existent in save location?
        try:
            save = "assembly_corr_{}_{}_{}_{}_{}.npz".format(n_spec, n_coms,
                                                        *sorted([key, keyj]), 0 if i>j else 8)
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
        
        
        corr = brentq(fun, gp.pt.corr_phyto.loc[key,keyj],
                                            -1 if i>j else 1,
                                    args = (key, keyj, gp.pt.corr_phyto.copy()))
        tradeoff = {key+":"+keyj: corr}
        print(key, keyj, corr)

        traits = gp.generate_plankton(n_spec,n_coms,
                                      evolved_zoop=True, tradeoffs = tradeoff)
        
        env = gp.generate_env(n_coms)
        traits = gp.phytoplankton_equilibrium(traits, env)
        
        start = timer()
        richness, present, res, dens = assembly_richness(
                traits, env, plot_until = 0, ret_all = True,
                save = save)
        
        np.savez(save, corr = corr, i = n_coms, present = present, res = res,
         dens = dens, time = timer()-start, **traits, **env)