import numpy as np
from scipy.optimize import brentq
from timeit import default_timer as timer
import warnings

import generate_plankton as gp
from NFD_equilibrium import fast_NFD

def fun(x, A, tri, trj):
    A.loc[tri, trj] = x
    A.loc[trj,tri] = x
    return np.amin(np.linalg.eigvalsh(A.values))

n_prec = 9

n_coms = 1000

gleaner = ["c_Z:m_Z", "c_Z:k_Z", "c_Z:mu_Z", "c_n:k_n", "c_p:k_p"]
defense = ["R_P:mu_P", "R_P:k_n"]
super_resource = ["R_P:a", "R_P:e_P"]

corrs = gleaner + defense + super_resource

add_var = np.linspace(-1,1, n_prec)
all_start = timer()

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"

for k in range(10):
    for i in range(n_prec):
    
        for j,tradeoff in enumerate(corrs):
            tri, trj = tradeoff.split(":")
            print(tri, trj, i)
            try:
                save = "assembly_corr_{}_{}_{}_{}_{}.npz".format(n_coms, tri, trj, i,k)
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
                traits = gp.generate_plankton(2,n_coms,
                                              evolved_zoop=True, tradeoffs = tradeoff)
                
                env = gp.generate_env(n_coms)
                traits = gp.phytoplankton_equilibrium(traits, env)
            
                
            if len(w):
                raise
    
            start = timer()
            
            # simulate densities
            results, issues = fast_NFD(traits, env, save)
            
            np.savez(save, i = n_coms, **results, issue = issues, **traits, **env,
                     time = timer() - start, corr = corr)