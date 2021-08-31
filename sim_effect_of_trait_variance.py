import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

import generate_plankton as gp
"""
n_spec_max = 4
n_specs = np.arange(1, n_spec_max + 1)
n_coms = int(1e4)

trait_combs = np.array(["s_zp", "h_zp"])

n_prec = 15
div_phyto = pd.DataFrame(np.nan, index = np.arange(n_prec),
                         columns = gp.pt.phyto_traits)
corr_phy_mm = div_phyto.copy()
div_zoop = pd.DataFrame(np.nan, index = np.arange(n_prec),
                        columns = gp.zt.zoop_traits)

div_combined = pd.DataFrame(np.nan, index = np.arange(n_prec),
                        columns = trait_combs)

corr_zoo_mm = div_zoop.copy()
div_all = pd.DataFrame(np.nan, index = ["r_spec_{}".format(2*i)
                                for i in n_specs] + ["mean"], columns = [])

ref_id = div_all.columns

var = 2**np.linspace(-2,2,n_prec)
var[0] = 0.01
for i, key in enumerate(div_phyto.columns):
    print(key)
    for j, v in enumerate(var):
        # increased variance
        div = []
        for n_spec in n_specs:
            if n_spec <=0:
                continue
            # generate phytoplankton and environment           
            traits = gp.generate_plankton(n_spec,n_coms,
                                          evolved_zoop=True, diff_std = {key:v})
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
    for j, v in enumerate(var):
        # increased variance
        div = []
        for n_spec in n_specs:
            if n_spec <=0:
                continue
            # generate phytoplankton and environment           
            traits = gp.generate_plankton(n_spec,n_coms,
                                          evolved_zoop=True, diff_std = {key:v})
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
    print(key)
    for j, v in enumerate(var):
        # increased variance
        div = []
        for n_spec in n_specs:
            if n_spec <=0:
                continue
            # generate phytoplankton and environment           
            traits = gp.generate_plankton(n_spec,n_coms,
                                          evolved_zoop=True, diff_std = {key:v})
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
m_ref = np.mean(div_all.loc["mean", ref_id])
std_ref = np.std(div_all.loc["mean", ref_id])
var2 = var.copy()
var2[0] = 0.25
fac = 1.96
fig, ax = plt.subplots(3,1,sharex = True, sharey = True, figsize = (9,9))

ax[0].semilogx(var2, div_phyto, label = div_phyto.columns)
handles, labels = ax[0].get_legend_handles_labels()
ax[0].legend(div_phyto.columns.values, ncol = 3)
ax[0].axhline(m_ref-fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[0].axhline(m_ref+fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[0].axvline(1, color = "k")

ax[1].semilogx(var2, div_zoop, label = div_zoop.columns)
ax[1].legend(div_zoop.columns.values, ncol = 2)
ax[1].axhline(m_ref-fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[1].axhline(m_ref+fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[1].axvline(1, color = "k")

ax[2].semilogx(var2, div_combined, label = div_combined.columns)
ax[2].legend(div_combined.columns.values, ncol = 2)
ax[2].axhline(m_ref-fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[2].axhline(m_ref+fac*std_ref, color = "r", linestyle = "--", alpha = 0.5)
ax[2].axvline(1, color = "k")

ax[2].set_xticks([0.5,1,2])
ax[2].set_xticklabels([0.5, 1,2])

fig.savefig("Figure_effect_of_trait_variance.pdf")

