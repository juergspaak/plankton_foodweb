import numpy as np
import matplotlib.pyplot as plt

import generate_plankton as gp
add_mean = [-1,0,1]
change_var = [0.5, 1, 2]

itera = int(1e5)
trait = "mu_P"
mean_traits = np.array([gp.generate_plankton(1, itera,
                            diff_mean={trait:i*gp.pt.std_phyto[trait]})[trait]
               for i in add_mean])
mean_traits = np.log(mean_traits)

fig, ax = plt.subplots(2,2, figsize = (9,9), sharex = "col")
label = ["Decreased", "Reference", "Increased"]
perc = [0.1, 99.9]
for i in range(len(add_mean)):
    ax[0,0].hist(mean_traits[i], alpha = 0.5,
                 bins = np.linspace(*np.percentile(mean_traits, perc), 30),
                 label = label[i])
    ax[0,1].hist(np.exp(mean_traits[i]), alpha = 0.5,
                 bins = np.linspace(*np.percentile(np.exp(mean_traits), perc), 50),
                 label = label[i])
    
var_traits = np.array([gp.generate_plankton(1, itera,
                            diff_std={trait:i})[trait] for i in change_var])
var_traits = np.log(var_traits)

for i in range(len(add_mean)):
    ax[1,0].hist(var_traits[i], alpha = 0.5,
                 bins = np.linspace(*np.percentile(var_traits, perc), 30),
                 label = label[i])
    ax[1,1].hist(np.exp(var_traits[i]), alpha = 0.5,
                 bins = np.linspace(*np.percentile(np.exp(var_traits), perc), 50),
                 label = label[i])
    
# add layout
for i, a in enumerate(ax.flatten()):
    a.legend()
    a.set_title("ABCD"[i], loc = "left")
    
ax[0,0].set_title("Log trait distribution")
ax[0,1].set_title("Trait distribution")

ax[1,0].set_xlabel("Maximal growth rate $\mu_i^P$ (log)")
ax[1,1].set_xlabel("Maximal growth rate $\mu_i^P$")

ax[0,0].set_ylabel("Frequency")
ax[1,0].set_ylabel("Frequency")

fig.savefig("Figure_ap_change_traits.pdf")
    
    