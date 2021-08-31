import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

import generate_plankton as gp
"""
n_spec_max = 4
n_specs = np.arange(1, n_spec_max + 1)
n_coms = int(1e4)

trait_combs = np.array(["h_zp"])

n_prec = 9
delta_mean = np.linspace(-1,1,n_prec)
div_phyto = pd.DataFrame(np.nan, index = np.arange(n_prec),
                         columns = gp.pt.phyto_traits)
corr_phy_mm = div_phyto.copy()
div_zoop = pd.DataFrame(np.nan, index = np.arange(n_prec),
                        columns = gp.zt.zoop_traits)

div_combined = pd.DataFrame(np.nan, index = np.arange(n_prec),
                        columns = trait_combs)

div_all = pd.DataFrame(np.nan, index = ["r_spec_{}".format(2*i)
                                for i in n_specs], columns = [])

ref_id = div_all.columns

var = 2**np.linspace(-2,2,n_prec)
var[0] = 0.01
for i, key in enumerate(div_phyto.columns):
    print(key)
    add_mean = delta_mean*gp.pt.std_phyto[key].values**2
    for j, v in enumerate(var):
        # increased variance
        div = []
        for n_spec in n_specs:
            if n_spec <=0:
                continue
            # generate phytoplankton and environment           
            traits = gp.generate_plankton(n_spec,n_coms, evolved_zoop=True,
                                          diff_mean = {key:add_mean[j]})
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
        div_all[key + str(j)] = div
        div_phyto.loc[j,key] = div[-1]
        
for i, key in enumerate(div_zoop.columns):
    print(key)
    add_mean = delta_mean*gp.zt.std_zoop[key].values**2
    for j, v in enumerate(var):
        # increased variance
        div = []
        for n_spec in n_specs:
            if n_spec <=0:
                continue
            # generate phytoplankton and environment           
            traits = gp.generate_plankton(n_spec,n_coms, evolved_zoop=True,
                                          diff_mean = {key:add_mean[j]})
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
        div_all[key + str(j)] = div
        div_zoop.loc[j,key] = div[-1]


##### combined traits
for i, key in enumerate(trait_combs):
    traits = gp.generate_plankton(n_spec,n_coms, evolved_zoop=True)
    std_h_zp = np.std(np.log(traits["h_zp"]))
    add_mean = delta_mean*std_h_zp**2
    print(key)
    for j, v in enumerate(var):
        # increased variance
        div = []
        for n_spec in n_specs:
            if n_spec <=0:
                continue
            # generate phytoplankton and environment           
            traits = gp.generate_plankton(n_spec,n_coms, evolved_zoop=True,
                                          diff_mean = {"h_zp": add_mean[j]})
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
        div_all[key + str(j)] = div
        div_combined.loc[j,key] = div[-1]
"""

ref_id = div_all.columns[[key[-1] == str(n_prec//2) for key in div_all.columns]]
m_ref = np.nanmean(div_all.loc["mean", ref_id])
std_ref = np.nanstd(div_all.loc["mean", ref_id])
fac = 1.96
fig, ax = plt.subplots(3,1,sharex = True, sharey = True, figsize = (9,9))

ax[0].plot(delta_mean, div_phyto, label = div_phyto.columns)
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(div_phyto.columns.values, ncol = 3)
ax[0].axhline(m_ref-fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[0].axhline(m_ref+fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[0].axvline(0, color = "k")

ax[1].plot(delta_mean, div_zoop, label = div_zoop.columns)
ax[1].legend(div_zoop.columns.values, ncol = 2)
ax[1].axhline(m_ref-fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[1].axhline(m_ref+fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[1].axvline(0, color = "k")

ax[2].plot(delta_mean, div_combined, label = div_combined.columns)
ax[2].legend(div_combined.columns.values, ncol = 2)
ax[2].axhline(m_ref-fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[2].axhline(m_ref+fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[2].axvline(0, color = "k")

ax[-1].set_xlabel("Change in mean")

for i,a in enumerate(ax.flatten()):
    a.set_title("ABCDEF"[i], loc = "left")
    a.set_ylabel("Species richness")

fig.savefig("Figure_effect_of_trait_mean.pdf")