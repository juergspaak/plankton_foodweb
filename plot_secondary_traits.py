import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.stats import linregress

import generate_plankton as gp


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
traits = gp.generate_plankton(r_phyto,100000)
traits = gp.phytoplankton_equilibrium(traits)
traits = gp.community_equilibrium(traits)
traits["size_Z"] *= 1e6
ind = ((np.all(traits["N_star_P"]>0, axis = -1))
       & np.all(traits["N_star_Z"]>0, axis = -1)
       & np.all(traits["N_star_P_res"]>0, axis = -1))
traits = {key: traits[key][ind] for key in traits.keys()}
with warnings.catch_warnings(record = True):
    tf_simple = {key: np.log(traits[key].flatten()) for key in traits.keys()}
    tf_zp = {key: np.log(traits[key]) for key in traits.keys()}
    tf_zp["size_P"] = np.repeat(tf_zp["size_P"][:,np.newaxis], r_zoo, axis = 1)
    tf_zp["size_Z"] = np.repeat(tf_zp["size_Z"][...,np.newaxis], r_phyto,
                                     axis = 1)
tf_zp = {key: tf_zp[key].flatten() for key in tf_zp.keys()}


fig, ax = plt.subplots(3,2,figsize = (9,9), sharex = True, sharey = True)
ax = ax.flatten()

keys = ["N_star_P" + case for case in ["_n", "_p", "_l", "_res", ""]] + ["N_star_Z"]
size = 5*["size_P"] + ["size_Z"]

for i,key in enumerate(keys):
    scatter_plot(tf_simple, size[i], key, ax[i])
    
fig, ax = plt.subplots(2,1, figsize = (7,7))
traits_all = gp.generate_plankton(r_phyto,100000)
traits_all["size_Z"] *= 1e6
traits_all = {key: np.log(traits_all[key].flatten()) for key in traits_all.keys()}
bins = 30
ax[0].hist(traits_all["size_P"], bins = bins, density = True)
ax[0].hist(tf_simple["size_P"], bins = bins, density = True, alpha = 0.5)
ax[1].hist(traits_all["size_Z"], bins = bins, density = True)
ax[1].hist(tf_simple["size_Z"], bins = bins, density = True, alpha = 0.5)


