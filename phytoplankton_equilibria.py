import numpy as np
import matplotlib.pyplot as plt
import warnings

from scipy.stats import linregress

import generate_plankton as gp
import plankton_growth as pg

n_coms = int(1e4)
r_phyto = 1

traits = gp.generate_plankton(r_phyto, n_coms, evolved_zoop=True)
env = gp.generate_env(n_coms)
traits = gp.community_equilibrium(traits, env)
traits = gp.phytoplankton_equilibrium(traits, env)

# change zooplankton size to \mum^3
traits["size_Z"] /= gp.uc["mum3_mg"]

with warnings.catch_warnings(record = True):
    traits_log = {key: np.log(traits[key])
              for key in gp.select_keys(traits)}




def scatter_plot(data, key1, key2, axc):
    
    ind = np.isfinite(data[key1]*data[key2])
    x = data[key1][ind]
    y = data[key2][ind]
    axc.scatter(x[:1000], y[:1000], s = 1, color = "orange")
    s,i,r,p,std = linregress(x,y)
    
    linreg_range = np.percentile(x, [1,50,99])
    axc.plot(linreg_range, i + s*linreg_range, 'r-', linewidth = 3,
             label = "s = {}".format(np.round(s, 3)))
    axc.plot(linreg_range, np.mean(y)+
        np.sign(s)*(linreg_range-linreg_range[1]), "b--", linewidth = 3)
    axc.legend()
    axc.set_xlabel(key1)
    axc.set_ylabel(key2)


# different equilibria of the phytoplankton communities
equis = ["_n", "_p", "_l", "_res", ""]
itera = 1000
bins = 30
N_stars = ["N_star_P" + equi for equi in equis] + ["N_star_Z"]
sizes = len(equis)*["size_P"] + ["size_Z"]
fig, ax = plt.subplots(len(equis)+1,2, figsize = (11,11), sharex = "col",
                       sharey = "col")

for i,equi in enumerate(N_stars):
    scatter_plot(traits_log, sizes[i], equi, axc = ax[i,0])

    ax[i,1].hist(np.log(traits[sizes[i]].flatten()), bins = bins,
                 density = True)
    ax[i,1].hist(np.log(traits[sizes[i]][traits[equi]>0]),
                 bins = bins, alpha = 0.5, density = True)
    ax[i,1].set_xlabel(sizes[i])
    ax[i,1].set_ylabel("frequency")
    
    ax[i,1].text(ax[i,1].get_xlim()[0], np.mean(ax[i,1].get_ylim()),
                "N = {}".format(np.sum(traits[equi]>0)/n_coms/r_phyto))
    
fig.savefig("Figure_N_star_values.pdf")
