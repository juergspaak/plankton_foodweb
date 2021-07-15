import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy

import generate_plankton as gp

n_spec_max = 4
n_specs = np.arange(1, n_spec_max+1)
n_coms = int(1e5)

# first generate sufficient communities
traits, env = gp.generate_communities(4, int(n_coms))
print(traits["n_coms"])
while traits["n_coms"]<1e4:
    traits_new, env_new = gp.generate_communities(4, int(n_coms))
    traits = {key: np.append(traits[key], traits_new[key], axis = 0)
              for key in gp.select_keys(traits)}
    traits["n_coms"] = len(traits["size_P"])
    print(traits["n_coms"])
    env = {key: np.append(env[key], env_new[key]) for key in env.keys()}

traits_org = copy.deepcopy(traits)
env_org = copy.deepcopy(env)

trait_combs = np.append(gp.pt.phyto_traits, gp.zt.zoop_traits)
trait_combs = [[], ["s_zp"], ["h_zp"]] + [[trait] for trait in trait_combs]
trait_combs += [["c_p", "c_n", "a"],["k_p", "k_n", "k_l"], ["c_Z", "k_Z"]]
trait_combs = np.array(trait_combs, dtype = "object")

columns = ["comb", "n_comb"] + ["r_spec_{}".format(2*i)
                                for i in n_specs]



diversity = pd.DataFrame(data = np.nan, columns = columns,
                         index = np.arange(len(trait_combs)))
traits_org = traits_org
trait_combs = trait_combs


for ic, comb in enumerate(trait_combs):
    diversity.loc[ic, "comb"] = "-".join(comb)
    diversity.loc[ic, "n_comb"] = len(comb)
    print(ic, comb)
    
    # copy tratits
    traits_changed = {key: traits_org[key].copy()
                      for key in gp.select_keys(traits_org)}
    env = {key: env_org[key] for key in env.keys()}
        
    # set traits to constant values
    for t_name in comb:
        if t_name == "s_zp":
            traits_changed["s_zp"][:] = 1
        else:
            # take geometric mean
            axis = tuple(np.arange(traits_changed[t_name].ndim)[1:])
            traits_changed[t_name][:] = np.exp(np.mean(np.log(traits_changed[t_name])
                                , axis = axis, keepdims = True))
                
    ind_c = np.arange(traits_org["n_coms"]).reshape(-1,1)
    stay_phyto = stay_zoo = np.repeat(np.arange(n_spec_max).reshape(1,-1),
                                      traits_org["n_coms"], axis = 0)
    for i in range(n_spec_max):
        if len(ind_c) == 0:
            continue
        
        traits = {"r_phyto": stay_phyto.shape[-1],
                  "r_zoo": stay_phyto.shape[-1], "n_coms": len(stay_phyto)}
        for trait in gp.pt.phyto_traits:
            traits[trait] = traits_changed[trait][ind_c, stay_phyto]
        for trait in gp.zt.zoop_traits:
            traits[trait] = traits_changed[trait][ind_c, stay_zoo]
        for trait in ["h_zp", "s_zp"]:
            traits[trait] = traits_changed[trait][ind_c[:,np.newaxis],
                                       stay_zoo[..., np.newaxis]
                                       , stay_phyto[:,np.newaxis]]
        
        env = {key: env_org[key][ind_c] for key in env_org.keys()}
        
        
        # compute equilibria
        traits = gp.community_equilibrium(traits, env)
        
        ind = (np.isfinite(traits["N_star_P"]).all(axis = -1) &
               np.isfinite(traits["N_star_Z"]).all(axis = -1))
        ind_c = ind_c[~ind]
        ind_c.shape = -1,1
        diversity.loc[ic, "r_spec_{}".format((n_spec_max-i)*2)] = np.sum(ind)
        
        # remove species with lowest density
        traits["N_star_P"][np.isnan(traits["N_star_P"])] = -np.inf
        traits["N_star_Z"][np.isnan(traits["N_star_Z"])] = -np.inf
        
        stay_phyto = np.argsort(traits["N_star_P"])[~ind,1:]
        stay_zoo = np.argsort(traits["N_star_P"])[~ind,1:]
        


diversity["mean"] = np.nansum(n_specs*diversity[["r_spec_{}".format(2*i)
                                for i in n_specs]], axis = 1)
diversity["tot_prob"] = np.nansum(diversity[["r_spec_{}".format(2*i)
                                for i in n_specs]], axis = 1)

for i in n_specs:
    diversity["r_spec_rel_{}".format(2*i)] = (
                diversity["r_spec_{}".format(2*i)]/diversity["tot_prob"])
    
diversity["rel_mean"] = np.nansum(n_specs*diversity[["r_spec_rel_{}".format(2*i)
                                for i in n_specs]], axis = 1)

diversity = diversity.loc[np.argsort(diversity["mean"])]

div_arr = diversity[["r_spec_{}".format(2*i)
                                for i in n_specs]].values

fig, ax = plt.subplots(2,1,sharex = True, sharey = False)
loc = np.arange(len(trait_combs))
for i in range(n_spec_max):
    ax[0].bar(loc, div_arr[:,i], bottom = np.nansum(div_arr[:,:i], axis = 1))
    
ax_mean = ax[0].twinx()
ax_mean.plot(loc, diversity["mean"], 'ko')

ax[1].set_xticks(loc)
ax[1].set_xticklabels(diversity["comb"], rotation = 90)

div_arr = diversity[["r_spec_rel_{}".format(2*i)
                                for i in n_specs]].values
for i in range(n_spec_max):
    ax[1].bar(loc, div_arr[:,i], bottom = np.nansum(div_arr[:,:i], axis = 1))
    
ax_mean = ax[1].twinx()
ax_mean.plot(loc, diversity["rel_mean"], 'ko')


fig.tight_layout()
fig.savefig("Figure_diversity_perturb.pdf")