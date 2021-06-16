import numpy as np
import matplotlib.pyplot as plt

import generate_plankton as gp
import plankton_growth as pg

traits, env = gp.generate_communities(1,int(1e4), evolved_zoop=True)

# find the interesting phyto and zooplankton species of correct size
n_plots = 5
size_P = np.nanpercentile(traits["size_P"], np.linspace(5,95,n_plots))
inds_P = np.argmin(np.abs(traits["size_P"]-size_P), axis = 0)

size_Z = np.nanpercentile(traits["size_Z"], np.linspace(5,95,n_plots))
inds_Z = np.argmin(np.abs(traits["size_Z"]-size_Z), axis = 0)

fig, ax = plt.subplots(n_plots, n_plots, figsize = (2.5*n_plots, 2.5*n_plots),
                       sharex = True, sharey = True)

n_precision = 100
for ip, ind_P in enumerate(inds_P):
    for iz, ind_Z in enumerate(inds_Z):
        # first copy all traits for the phytoplankton species
        tij, envi, i = gp.select_i(traits, env, ind_P)
        
        # select zooplankton traits
        for trait_z in ["size_Z", "c_Z", "mu_Z", "k_Z"]:
            tij[trait_z] = traits[trait_z][ind_Z]
        
        # selectivity is one, only one phytoplankton species
        tij["s_zp"] = np.ones((1,1,1))
        tij["h_zp"] = np.exp(np.log(0.001/24)
                -2.11*(0.2878*np.log(tij["size_Z"])+3.75)
                -np.log(np.pi/6) + 1*np.log(tij["size_P"]))
        tij["h_zp"].shape = (1,1,1)
        
        growth = np.empty((n_precision,2))
        N = np.exp(np.linspace(-5, np.log(10*tij["N_star_P_res"][0]),
                               n_precision))
        for i in range(n_precision):
            growth[i] = pg.plankton_growth(np.array([N[i],1]), tij, envi)
        
        ax[ip, iz].semilogx(N, growth[:,1])
        ax[ip, iz].axhline(0)
        ax[ip, iz].axhline(tij["mu_Z"][0])
        ax[ip, iz].axvline(tij["N_star_P_res"][0])