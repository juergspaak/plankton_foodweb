import numpy as np
from scipy.optimize import brentq
import warnings

from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer

n_prec = 9
n_coms = 1000
n_spec = 20

# find correlation matrices for maximal and minimal correlation of affinities
A = gp.pt.corr_phyto.copy()

def fun(x, A, tri, trj):
    A.loc[tri, trj] = x
    A.loc[trj,tri] = x
    return np.amin(np.linalg.eigvalsh(A.values))

corr_bounds = {"mu_P:k_n": 0, "mu_P:k_p":0, "k_n:k_p": 0}
for key in corr_bounds.keys():
    tri, trj = key.split(":")
    corr_bounds[key] = [brentq(fun, A.loc[tri,trj], -1,
                                args = (A.copy(), tri, trj))+0.001,
                   brentq(fun, A.loc[tri,trj], 1,
                                args = (A.copy(), tri, trj))-0.001]
corr_bounds["mu_P:k_n"] = corr_bounds["mu_P:k_n"][::-1]
corr_bounds["mu_P:k_p"] = corr_bounds["mu_P:k_p"][::-1]


corr_bounds = {key: np.linspace(*corr_bounds[key], n_prec)
               for key in corr_bounds.keys()}
# first two correlations are too extreme
corr_bounds = {key: np.linspace(*corr_bounds[key][[2,-1]], n_prec)
               for key in corr_bounds.keys()}

diff_std = 2**np.linspace(-1,1,n_prec)
diff_std[:] = 1

path = "data"
for i in range(n_prec):
    print(i)
    try:
        save = "assembly_R_star_{}_{}_{}.npz".format(n_spec, n_coms, i)
        
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
    
    with warnings.catch_warnings(record=True) as w:
        traits = gp.generate_plankton(n_spec,n_coms, evolved_zoop=True,
                    tradeoffs = {key: corr_bounds[key][i]
                                 for key in corr_bounds.keys()},
                    diff_std={"mu_P": diff_std[i]})
        
        env = gp.generate_env(n_coms)
        traits = gp.phytoplankton_equilibrium(traits, env)
        
    if len(w):
        raise
    with warnings.catch_warnings(record = True) as w:
        R_star = np.log([traits["R_star_n"].flatten(),
                             traits["R_star_p"].flatten()])
        R_star = R_star[:,np.all(np.isfinite(R_star), axis = 0)]
        corr = np.corrcoef(R_star)[0,1]
    
    start = timer()
    richness, present, res, dens = assembly_richness(
                traits, env, plot_until = 0, ret_all = True,
                save = save)
    np.savez(save, corr = corr, i = n_coms, present = present, res = res,
         dens = dens, time = timer()-start, **traits, **env )