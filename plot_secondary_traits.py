import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.stats import linregress

import generate_plankton as gp
import plankton_growth as pg


def scatter_plot(data, key1, key2, axc):
    axc.scatter(data[key1][:1000], data[key2][:1000], s = 1,
                color = "orange")
    s,i,r,p,std = linregress(data[key1], data[key2])
    
    linreg_range = np.percentile(data[key1], [1,50,99])
    axc.plot(linreg_range, i + s*linreg_range, 'r-', linewidth = 3,
             label = "s = {}".format(np.round(s, 3)))
    axc.plot(linreg_range, np.mean(data[key2])+
        np.sign(s)*(linreg_range-linreg_range[1]), "b--", linewidth = 3)
    axc.legend()
    axc.set_xlabel(key1)
    axc.set_ylabel(key2)

r_phyto = 1
r_zoo = 3
n_coms = int(1e5)
traits = gp.generate_plankton(r_phyto,n_coms)
env = {"I_in": np.random.uniform(50,200, (n_coms,1)),
       "P": np.random.uniform(5,20, (n_coms, 1)),
       "N": np.random.uniform(25,100, (n_coms,1)),
       "d": np.random.uniform(0.01,0.2, (n_coms,1)),
       "zm": np.random.uniform(1,100, (n_coms,1))}
traits = gp.phytoplankton_equilibrium(traits, env)
traits = gp.community_equilibrium(traits, env)
traits["size_Z"] *= 1e6
ind1 = np.isfinite(traits["N_star_P"]).all(axis = -1)
ind2 = np.isfinite(traits["N_star_Z"]).all(axis = -1)
ind = ind1 & ind2

traits = {key: traits[key][ind] for key in traits.keys()}
print(np.round([sum(ind1)/len(ind1), sum(ind2)/len(ind2), sum(ind)/len(ind)],3))

env = {key: env[key][ind] for key in env.keys()}
n_coms = len(traits["mu_P"])

##############################################################################
# test that equilibriums have been computed correctly
# check whether equilibrium was computed correctly
i = np.random.randint(n_coms)
t_i = {key: traits[key][i] for key in traits.keys()}
env_i = {key: env[key][i,0] for key in env.keys()}

# check whether computation has been done correctly
N = np.append(t_i["N_star_P"], t_i["N_star_Z"])
if np.amax(np.abs(pg.plankton_growth(N, t_i, env_i)))>1e-8:
    raise
# check phytoplankton monoculture
for i in range(r_phyto):
    N = np.zeros(r_phyto)
    N[i] = t_i["N_star_P_res"][i]
    if np.abs(N[i]*(pg.phyto_growth(N, t_i, env_i)[i]-env_i["d"]))>1e-8:
        raise


###############################################################################


with warnings.catch_warnings(record = True):
    tf_simple = {key: np.log(traits[key].flatten()) for key in traits.keys()}
    tf_zp = {key: np.log(traits[key]) for key in traits.keys()}
    tf_zp["size_P"] = np.repeat(tf_zp["size_P"][:,np.newaxis], r_zoo, axis = 1)
    tf_zp["size_Z"] = np.repeat(tf_zp["size_Z"][...,np.newaxis], r_phyto,
                                     axis = 1)


fig, ax = plt.subplots(3,2,figsize = (9,9), sharex = True, sharey = True)
ax = ax.flatten()

keys = ["N_star_P" + case for case in ["_n", "_p", "_l", "_res", ""]] + ["N_star_Z"]
size = 5*["size_P"] + ["size_Z"]

for i,key in enumerate(keys):
    scatter_plot(tf_simple, size[i], key, ax[i])

fig, ax = plt.subplots(2,1, figsize = (7,7))
traits_all = gp.generate_plankton(r_phyto,100000)
traits_all["size_Z"] *= 1e6
bins = 30

for i in range(r_phyto):
    ax[0].hist(np.log(traits_all["size_P"][:,i]), bins = bins, density = True,
               alpha = 0.9)
    ax[0].hist(np.log(traits["size_P"][:,i]), bins = bins, density = True,
               alpha = 0.5, zorder = 3)
    
    ax[1].hist(np.log(traits_all["size_Z"][:,i]), bins = bins, density = True,
               alpha = 0.9)
    ax[1].hist(np.log(traits["size_Z"][:,i]), bins = bins, density = True,
               alpha = 0.5, zorder = 3)

fig.savefig("Figure_equilibrium_size_dependence.pdf")

