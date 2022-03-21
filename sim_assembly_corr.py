"""
This file runs simulations with altered correlations for the phytoplankton
and zooplankton traits
"""


import numpy as np
import warnings
from scipy.optimize import brentq

from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer




zoop = ["c_Z:k_Z", "c_Z:mu_Z", "c_Z:m_Z", "k_Z:mu_Z", "m_Z:mu_Z"]
gleaner = ["c_n:k_n", "c_p:k_p", ]
resources = ["c_n:c_p", "k_n:k_p"]
growth_defense = ["R_p:e_P", "R_P:mu_P", "e_P:mu_P"]
weird = ["c_n:mu_P", "R_P:k_n", "R_P:a"]
corrs = zoop + gleaner + resources + weird + growth_defense

n_prec = 9
n_coms = int(1000)
n_spec = 20

def fun(x, A, tri, trj):
    A.loc[tri, trj] = x
    A.loc[trj,tri] = x
    return np.amin(np.linalg.eigvalsh(A.values))

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"

for i in range(n_prec):
    for j,tradeoff in enumerate(corrs):
        tri, trj = tradeoff.split(":")
        print(i,j, tri, trj)
        
        try:
            save = "assembly_corr_{}_{}_{}_{}_{}.npz".format(n_spec, n_coms, tri, trj, i)
            
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
        
        A = (gp.pt.corr_phyto.copy() if tri in gp.pt.phyto_traits
             else gp.zt.corr_zoop.copy())
        
        
        corr_bounds = [brentq(fun, A.loc[tri,trj], -1,
                                    args = (A.copy(), tri, trj))+0.001,
                       brentq(fun, A.loc[tri,trj], 1,
                                    args = (A.copy(), tri, trj))-0.001]
        
        corr = np.linspace(*corr_bounds, n_prec)[i]
        tradeoff = {tri+":"+trj: corr}
        print(tri, trj, i, corr, tradeoff)
        with warnings.catch_warnings(record=True) as w:
            traits = gp.generate_plankton(n_spec,n_coms,
                                          evolved_zoop=True, tradeoffs = tradeoff)
            
            env = gp.generate_env(n_coms)
            traits = gp.phytoplankton_equilibrium(traits, env)
        
            
        if len(w):
            raise
        
        
        start = timer()
        richness, present, res, dens = assembly_richness(
                    traits, env, plot_until = 0, ret_all = True,
                    save = save)
        np.savez(save, corr = corr, i = n_coms, present = present, res = res,
             dens = dens, time = timer()-start, **traits, **env )
    