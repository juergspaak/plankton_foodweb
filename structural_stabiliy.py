import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import generate_plankton as gp



t_coex, env_coex = gp.generate_communities(3, int(10e5))
    
errors = 10.0**np.linspace(-4,0, 13)
itera = 30
frac = np.empty((len(errors), itera))
for i in range(len(errors)):
    for j in range(itera):
        error = errors[i]
        t_err = {key:t_coex[key]*(1+np.random.uniform(-error, error, t_coex[key].shape))
                 for key in gp.select_keys(t_coex)}
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