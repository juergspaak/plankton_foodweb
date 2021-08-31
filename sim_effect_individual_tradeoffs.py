import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

import generate_plankton as gp

n_spec_max = 4
n_specs = np.arange(1, n_spec_max + 1)
n_coms = int(1e4)

trait_combs = np.append(gp.pt.phyto_traits, gp.zt.zoop_traits)
trait_combs = [["Ref"], ["s_zp"], ["h_zp"]] + [[trait] for trait in trait_combs]
trait_combs = np.array(trait_combs, dtype = "object")

div_phyto = pd.DataFrame(np.nan, index = gp.pt.phyto_traits,
                         columns = gp.pt.phyto_traits)
corr_phy_mm = div_phyto.copy()
div_zoop = pd.DataFrame(np.nan, index = gp.zt.zoop_traits,
                        columns = gp.zt.zoop_traits)
corr_zoo_mm = div_zoop.copy()
div_all = pd.DataFrame(np.nan, index = ["r_spec_{}".format(2*i)
                                for i in n_specs] + ["mean"], columns = [])

def fun(val,tri,trj,A):
    A.loc[tri, trj] = val
    A.loc[trj, tri] = val
    return np.amin(np.linalg.eigvalsh(A))

##############################################################################
# generate a reference list
for i in range(20):
    div = []
    for n_spec in n_specs:
        if n_spec <=0:
            continue
        # generate phytoplankton and 
        traits = gp.generate_plankton(n_spec,n_coms,
                                      evolved_zoop=True)

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
    div_all["ref" + str(i)] = div
    print(i)
    
ref_id = div_all.columns

for i, key in enumerate(div_phyto.columns):
    
    for j,keyj in enumerate(div_phyto.columns):
        if key == keyj:
            continue
        div = []
        corr_phy_mm.loc[key, keyj] = brentq(fun, gp.pt.corr_phyto.loc[key,keyj],
                                            -1 if i>j else 1,
                                    args = (key, keyj, gp.pt.corr_phyto.copy()))
        tradeoff = {key+":"+keyj: corr_phy_mm.loc[key, keyj]}
        for n_spec in n_specs:
            if n_spec <=0:
                continue
            # generate phytoplankton and 
            traits = gp.generate_plankton(n_spec,n_coms,
                                      evolved_zoop=True, tradeoffs = tradeoff)

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
        div_all[key+":"+keyj] = div
        div_phyto.loc[key,keyj] = div[-1]
        print(key, keyj, div[-1])

###############################################################################
# zooplankton traits

for i, key in enumerate(div_zoop.columns):
    
    for j,keyj in enumerate(div_zoop.columns):
        if key == keyj:
            continue
        div = []
        corr_zoo_mm.loc[key, keyj] = brentq(fun, gp.zt.corr_zoop.loc[key,keyj],
                                            -1 if i>j else 1,
                                    args = (key, keyj, gp.zt.corr_zoop.copy()))
        tradeoff = {key+":"+keyj: corr_zoo_mm.loc[key, keyj]}
        for n_spec in n_specs:
            if n_spec <=0:
                continue
            # generate phytoplankton and 
            traits = gp.generate_plankton(n_spec,n_coms,
                                      evolved_zoop=True, tradeoffs = tradeoff)

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
        div_all[key+":"+keyj] = div
        div_zoop.loc[key,keyj] = div[-1]
        print(key, keyj, div[-1])

non_ref = div_all.columns[len(ref_id):]


div_all2 = div_all.sort_values("mean", axis = 1)

ref_m = np.mean(div_all.loc["mean", ref_id])
ref_std = np.std(div_all.loc["mean", ref_id])

plt.figure()
plt.hist(div_all.loc["mean", ref_id], bins = 5, alpha = 0.5, density = True)
plt.hist(div_all.loc["mean", non_ref], bins = 15, alpha = 0.5,
         density = True)
plt.axvline(ref_m, color = "r")
fac = 2.5
plt.axvline(ref_m + ref_std*fac, color = "r", linestyle = "--")
plt.axvline(ref_m - ref_std*fac, color = "r", linestyle = "--")


fig, ax = plt.subplots(2,2, figsize = (12,12), sharey = "row", sharex = "row")
from matplotlib import colors
divnorm=colors.TwoSlopeNorm(vmin=min(div_all.loc["mean"]),
                            vcenter=ref_m,
                            vmax=max(div_all.loc["mean"]))
cmap = ax[0,0].imshow(div_phyto, origin = "lower", norm=divnorm, cmap = "RdBu",)
fig.colorbar(cmap, ax = ax[0,0])

cmap = ax[0,1].imshow(corr_phy_mm, origin = "lower", cmap = "RdBu",
                      vmin = -1, vmax = 1)
fig.colorbar(cmap, ax = ax[0,1])
ax[0,0].set_xticks(range(len(div_phyto.columns.values)))
ax[0,0].set_xticklabels(div_phyto.columns.values)
ax[0,0].set_yticks(range(len(div_phyto.columns.values)))
ax[0,0].set_yticklabels(div_phyto.columns.values)

#### zooplankton traits
cmap = ax[1,0].imshow(div_zoop, origin = "lower", norm=divnorm, cmap = "RdBu",)
fig.colorbar(cmap, ax = ax[1,0])

cmap = ax[1,1].imshow(corr_zoo_mm, origin = "lower", cmap = "RdBu",
                      vmin = -1, vmax = 1)
fig.colorbar(cmap, ax = ax[1,1])
ax[1,0].set_xticks(range(len(div_zoop.columns.values)))
ax[1,0].set_xticklabels(div_zoop.columns.values)
ax[1,0].set_yticks(range(len(div_zoop.columns.values)))
ax[1,0].set_yticklabels(div_zoop.columns.values)

fig.savefig("Figure_effect_individual_tradeoffs.pdf")