import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer


n_coms = 100


const_traits = np.append(gp.pt.phyto_traits, gp.zt.zoop_traits)
const_traits = np.append(const_traits, ["h_zp"])




traits = gp.generate_plankton(5, n_coms)
env = gp.generate_env(n_coms)
traits = gp.phytoplankton_equilibrium(traits, env)
        
rich_all, id_survive, res_equi, dens_equi = assembly_richness(traits, env, plot_until = 0)


fig, ax = plt.subplots(3,2, figsize = (9,9))

prob_richness = np.empty((traits["r_phyto"], traits["r_zoo"]))
for i in range(traits["r_phyto"]):
    for j in range(traits["r_phyto"]):
        prob_richness[i,j] = np.sum(np.all(rich_all == [i,j], axis = -1))

prob_richness /= np.sum(prob_richness)
#prob_richness[prob_richness == 0] = np.nan

cmap = ax[0,0].imshow(prob_richness, origin = "lower")
fig.colorbar(cmap, ax = ax[0,0])
r_phyto, r_zoo = np.meshgrid(np.arange(traits["r_phyto"]),
                             np.arange(traits["r_zoo"]))
ax[0,0].scatter(r_phyto, r_zoo, s = 10000*prob_richness**2, color = "red")

ax[0,0].set_xlabel("Phytoplankton richness")
ax[0,0].set_ylabel("Zooplankton richness")

# traits survivor
tr_surv_phyto = np.array([traits[key][id_survive[:,0]] for key in gp.pt.phyto_traits])
tr_surv_zoo = np.array([traits[key][id_survive[:,1]] for key in gp.zt.zoop_traits])
tr_surv_phyto = np.log(tr_surv_phyto)
tr_surv_zoo = np.log(tr_surv_zoo)


# plot changes in mean and variance of each trait
diff_mean_phyto = (np.mean(tr_surv_phyto, axis = -1) - gp.pt.mean_phyto)/gp.pt.std_phyto**2
diff_var_phyto = (np.std(tr_surv_phyto, axis = -1) - gp.pt.std_phyto)/gp.pt.std_phyto
ax[0,1].plot(diff_mean_phyto, diff_var_phyto, 'r.')
for key in diff_mean_phyto.columns:
    ax[0,1].text(diff_mean_phyto[key], diff_var_phyto[key], key)

diff_mean_zoo = (np.mean(tr_surv_zoo, axis = -1) - gp.zt.mean_zoop)/gp.zt.std_zoop**2
diff_var_zoo = (np.std(tr_surv_zoo, axis = -1) - gp.zt.std_zoop)/gp.zt.std_zoop
ax[0,1].plot(diff_mean_zoo, diff_var_zoo, 'b.')
for key in diff_mean_zoo.columns:
    ax[0,1].text(diff_mean_zoo[key], diff_var_zoo[key], key)
    
ax[0,1].set_xlabel("Change in mean trait value")
ax[0,1].set_ylabel("Change in variance trait value")

from matplotlib import colors

corr_surv = np.corrcoef(tr_surv_phyto)
divnorm=colors.TwoSlopeNorm(vmin=-0.15,
                            vcenter=0,
                            vmax=0.15)
ax[1,0].set_title("Change in phytoplankton\ncorrelation")
cmap = ax[1,0].imshow(corr_surv - gp.pt.corr_phyto, norm = divnorm, cmap = "RdBu")
ax[1,0].set_xticks(np.arange(len(gp.pt.phyto_traits)))
ax[1,0].set_yticks(np.arange(len(gp.pt.phyto_traits)))
ax[1,0].set_xticklabels(gp.pt.phyto_traits, rotation = 90)
ax[1,0].set_yticklabels(gp.pt.phyto_traits)
fig.colorbar(cmap, ax = ax[1,0])

corr_surv = np.corrcoef(tr_surv_zoo)

ax[1,1].set_title("Change in zooplankton\ncorrelation")
cmap = ax[1,1].imshow(corr_surv - gp.zt.corr_zoop, norm = divnorm, cmap = "RdBu")
fig.colorbar(cmap, ax = ax[1,1])
ax[1,1].set_xticks(np.arange(len(gp.zt.zoop_traits)))
ax[1,1].set_yticks(np.arange(len(gp.zt.zoop_traits)))
ax[1,1].set_xticklabels(gp.zt.zoop_traits, rotation = 90)
ax[1,1].set_yticklabels(gp.zt.zoop_traits)

# plot distribution of surviving species
ax[2,0].hist(tr_surv_phyto[0], bins = 30, density = True, label = "Survivors")
ax[2,0].hist(np.log(traits["size_P"]).flatten(), bins = 30, density = True,
             alpha = 0.5, label = "reference")
ax[2,0].legend()
ax[2,0].set_xlabel("Phytoplankton size")
ax[2,0].set_ylabel("Probability")

ax[2,1].hist(tr_surv_zoo[0], bins = 30, density = True, label = "Survivors")
ax[2,1].hist(np.log(traits["size_Z"]).flatten(), bins = 30, density = True,
             alpha = 0.5, label = "reference")
ax[2,1].legend()
ax[2,1].set_xlabel("Zooplankton size")
ax[2,1].set_ylabel("Probability")

for i,a in enumerate(ax.flatten()):
    a.set_title("ABCDEF"[i], loc = "left")
fig.tight_layout()

#fig.savefig("Figure_ap_base_case_analysis.pdf")
