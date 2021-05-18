import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import generate_plankton as gp


try:
    t_coex
except:
    start = timer()
    n_coms = int(10e5)
    r_phyto = 4

    traits = gp.generate_plankton(r_phyto, n_coms, evolved_zoop=True)
    env = {"I_in": np.random.uniform(50,200, (n_coms,1)),
                   "P": np.random.uniform(5,20, (n_coms, 1)),
                   "N": np.random.uniform(50,500, (n_coms,1)),
                   "d": np.random.uniform(0.01,0.2, (n_coms,1)),
                   "zm": np.random.uniform(1,100, (n_coms,1))}
    traits = gp.community_equilibrium(traits, env)
    
    
    ind1 = np.isfinite(traits["N_star_P"]).all(axis = -1)
    ind2 = np.isfinite(traits["N_star_Z"]).all(axis = -1)
    ind = ind1 & ind2
    n_com_new = np.sum(ind)
    print(n_com_new/n_coms, n_com_new)
    t_coex = {key:traits[key][ind] for key in traits.keys()}
    env_coex = {key: env[key][ind] for key in env.keys()}
    t_all = gp.generate_plankton(r_phyto, n_com_new, evolved_zoop=True)
    print("time used: ", timer()-start)
    
errors = 10.0**np.linspace(-4,0, 13)
itera = 30
frac = np.empty((len(errors), itera))
for i in range(len(errors)):
    for j in range(itera):
        error = errors[i]
        t_err = {key:t_coex[key]*(1+np.random.uniform(-error, error, t_coex[key].shape))
                 for key in t_coex.keys()}
        env_err = {key: env_coex[key]*(1+np.random.uniform(-error, error, env_coex[key].shape))
                   for key in env_coex.keys()}
        t_err = gp.community_equilibrium(t_err, env_err)
        ind1 = np.isfinite(t_err["N_star_P"]).all(axis = -1)
        ind2 = np.isfinite(t_err["N_star_Z"]).all(axis = -1)
        ind = ind1 & ind2
        n_com_new = np.sum(ind)
        frac[i,j] = n_com_new/len(ind)
    print(error, np.round(np.mean(frac[i]),3))


plt.boxplot(frac.T)
plt.gca().set_xticklabels(np.round(errors,2))
plt.xlabel("relative error")
plt.ylabel("Fraction of coexisting communities")