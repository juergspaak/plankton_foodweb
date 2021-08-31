import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

import generate_plankton as gp
"""
n_spec_max = 4
n_specs = np.arange(1, n_spec_max + 1)
n_coms = int(1e4)



n_prec = 41
div_phyto = pd.DataFrame(np.nan, columns = ["exp", "det", "eigv", "mean"]+
                       ["r_spec_{}".format(2*i) for i in n_specs],
                       index = np.arange(n_prec))

exp = 10**np.linspace(-2,2,n_prec)
corr_phyto = gp.pt.corr_phyto.values
corr_phyto = np.sign(corr_phyto)*np.abs(corr_phyto)**exp.reshape(-1,1,1)
corr_phyto[:, np.arange(len(gp.pt.corr_phyto)), np.arange(len(gp.pt.corr_phyto))] = 1

div_phyto["det"] = np.linalg.det(corr_phyto)
div_phyto["eigv"] = np.amax(np.linalg.eigvalsh(corr_phyto), axis = 1)
div_phyto["exp"] = exp
div_phyto["eigmin"] = np.amin(np.linalg.eigvalsh(corr_phyto), axis = 1)

div = []


for n_spec in n_specs:
    # generate phytoplankton and 
    traits, env = gp.generate_communities(n_spec,n_coms,
                                  evolved_zoop=True)
    div.append(traits["n_coms"])
div.append(np.sum(n_specs*div)/sum(div))
print(div[-1])

for i in range(n_prec):
    div = []
    if div_phyto.loc[i, "eigmin"] <0:
        continue
    for n_spec in n_specs:
        # generate phytoplankton and 
        traits = gp.generate_plankton(n_spec,n_coms,
                                      evolved_zoop=True
                                      , corr_phyto = corr_phyto[i])

        env = gp.generate_env(n_coms)
        
        # remove communities in which not all phytoplankton can survive
        traits = gp.phytoplankton_equilibrium(traits, env)
        ind = np.all(np.isfinite(traits["N_star_P_res"]), axis = 1)
        traits, env, ind = gp.select_i(traits, env, ind)
        
        # compute equilibrium conditions
        traits = gp.community_equilibrium(traits, env)
        div.append(np.sum(np.all(traits["N_star_Z"]>0, axis = 1) &
                          np.all(traits["N_star_P"]>0, axis = 1))
                   /len(traits["mu_P"]))

    div.append(np.sum(n_specs*div)/sum(div))
    div_phyto.loc[i, ["r_spec_{}".format(2*i) for i in n_specs] + ["mean"]] = div
    print(i, np.linalg.det(corr_phyto[i]))

div_zoo = pd.DataFrame(np.nan, columns = ["exp", "det", "eigv", "mean"]+
                       ["r_spec_{}".format(2*i) for i in n_specs],
                       index = np.arange(n_prec))


exp = 10**np.linspace(-0.1,2,n_prec)
corr_zoo = gp.zt.corr_zoop.values
corr_zoo = np.sign(corr_zoo)*np.abs(corr_zoo)**exp.reshape(-1,1,1)
corr_zoo[:, np.arange(len(gp.zt.corr_zoop)), np.arange(len(gp.zt.corr_zoop))] = 1

div_zoo["det"] = np.linalg.det(corr_zoo)
div_zoo["eigv"] = np.amax(np.linalg.eigvalsh(corr_zoo), axis = 1)
div_zoo["exp"] = exp
div_zoo["eigmin"] = np.amin(np.linalg.eigvalsh(corr_zoo), axis = 1)

for i in range(n_prec):
    div = []
    if div_zoo.loc[i, "eigmin"] <0:
        continue
    for n_spec in n_specs:
        # generate zooplankton and 
        traits = gp.generate_plankton(n_spec,n_coms,
                                      evolved_zoop=True
                                      , corr_zoo = corr_zoo[i])
        env = gp.generate_env(n_coms)
        
        # remove communities in which not all phytoplankton can survive
        traits = gp.phytoplankton_equilibrium(traits, env)
        ind = np.all(np.isfinite(traits["N_star_P_res"]), axis = 1)
        traits, env, ind = gp.select_i(traits, env, ind)
        
        # compute equilibrium conditions
        traits = gp.community_equilibrium(traits, env)
        div.append(np.sum(np.all(traits["N_star_Z"]>0, axis = 1) &
                          np.all(traits["N_star_P"]>0, axis = 1))
                   /len(traits["mu_P"]))
    div.append(np.sum(n_specs*div)/sum(div))
    div_zoo.loc[i, ["r_spec_{}".format(2*i) for i in n_specs] + ["mean"]] = div
    print(i, np.linalg.det(corr_zoo[i]))
"""

fig, ax = plt.subplots(2,3, sharey = True)
mean = div_phyto.loc[div_phyto.exp == 1, "mean"]

ax[0,0].set_ylabel("Species richness")
ax[1,0].set_ylabel("Species richness")

ax[0,0].semilogx(div_phyto.det, div_phyto["mean"], '.')
ax[0,0].axvline(np.linalg.det(gp.pt.corr_phyto), color = "k")
ax[1,0].semilogx(div_zoo.det, div_zoo["mean"], '.')
ax[1,0].axvline(np.linalg.det(gp.zt.corr_zoop), color = "k")
ax[1,0].set_xlabel("determinant")

ax[0,1].plot(div_phyto.eigv, div_phyto["mean"], '.')
ax[0,1].axvline(np.amax(np.linalg.eigvalsh(gp.pt.corr_phyto)), color = "k")
ax[1,1].plot(div_zoo.eigv, div_zoo["mean"], '.')
ax[1,1].axvline(np.amax(np.linalg.eigvalsh(gp.zt.corr_zoop)), color = "k")
ax[1,1].set_xlabel("Maximum eigenvalue")

ax[0,2].semilogx(div_phyto.exp, div_phyto["mean"], '.')
ax[0,2].axvline(1, color = "k")
ax[1,2].semilogx(div_zoo.exp, div_zoo["mean"], '.')
ax[1,2].axvline(1, color = "k")
ax[1,2].set_xlabel("Exponent")



for a in ax.flatten():
    a.axhline(mean.values, color = "red")

fig.savefig("Figure_all_tradeoffs.pdf")

###############################################################################
fig, ax = plt.subplots(2,1)
ax[0].plot(div_phyto.eigv/len(gp.pt.phyto_traits), div_phyto["mean"], '.')
ax[0].axvline(np.amax(np.linalg.eigvalsh(gp.pt.corr_phyto))/len(gp.pt.phyto_traits), color = "k")
ax[1].plot(div_zoo.eigv/len(gp.zt.zoop_traits), div_zoo["mean"], '.')
ax[1].axvline(np.amax(np.linalg.eigvalsh(gp.zt.corr_zoop))/len(gp.zt.zoop_traits), color = "k")
ax[1].set_xlabel("Maximum eigenvalue")

ax[0].set_title("Change of phytoplankton correlation")
ax[1].set_title("Change of zooplankton correlation")
ax[0].set_ylabel("Species richeness")
ax[1].set_ylabel("Species richness")
fig.tight_layout()
fig.savefig("Figure_ap_all_correlations.pdf")