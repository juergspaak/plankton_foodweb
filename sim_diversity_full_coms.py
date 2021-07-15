import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import generate_plankton as gp

n_spec_max = 4
n_specs = np.arange(1, n_spec_max + 1)
n_coms = int(1e4)

trait_combs = np.append(gp.pt.phyto_traits, gp.zt.zoop_traits)
trait_combs = [[], ["s_zp"], ["h_zp"]] + [[trait] for trait in trait_combs]
trait_combs += [["c_p", "c_n", "a"],["k_p", "k_n", "k_l"], ["c_Z", "k_Z"]]
trait_combs = np.array(trait_combs, dtype = "object")

columns = ["comb", "n_comb"] + ["r_spec_{}".format(2*i)
                                for i in n_specs]

diversity = pd.DataFrame(data = np.nan, columns = columns,
                         index = np.arange(len(trait_combs)))
full_names = ["Unperturbed", "Selectivity", "Handling\ntime",
            "Size\nphyto", "Gorwth\nphyto", "Halfsaturation\nN",
            "Halfsaturation\nP","Halfsaturation\nlight", "Uptake\nN",
            "Uptake\nP", "Absorption", "Edibility", "Resource\nContent",
            "Size\nzoo", "Growth\nzoo", "Clearance", "Mortality\nzoo",
            "Halfsaturation\nzoo"]
full_names = full_names + (len(diversity)-len(full_names))*[""]
diversity["full_names"] = full_names
for ic, comb in enumerate(trait_combs):
    diversity.loc[ic, "comb"] = "-".join(comb)
    diversity.loc[ic, "n_comb"] = len(comb)
    print(ic, comb)
    for n_spec in n_specs:
        if n_spec <=0:
            continue
        # generate phytoplankton and environment
        traits = gp.generate_plankton(n_spec,n_coms, n_spec,
                                      evolved_zoop=True)
        
        # set traits to constant values
        for t_name in comb:
            if t_name == "s_zp":
                traits["s_zp"][:] = 1
            else:
                # take geometric mean
                axis = tuple(np.arange(traits[t_name].ndim)[1:])
                traits[t_name][:] = np.exp(np.mean(np.log(traits[t_name])
                                    , axis = axis, keepdims = True))
        env = gp.generate_env(n_coms)
        
        # compute equilibria
        traits = gp.community_equilibrium(traits, env)            
        
        ind = (np.isfinite(traits["N_star_P"]).all(axis = -1) &
               np.isfinite(traits["N_star_Z"]).all(axis = -1))
        diversity.loc[ic, "r_spec_{}".format(n_spec*2)] = np.sum(ind)/len(ind)


diversity["mean"] = np.sum(n_specs*diversity[["r_spec_{}".format(2*i)
                                for i in n_specs]], axis = 1)
diversity["tot_prob"] = np.sum(diversity[["r_spec_{}".format(2*i)
                                for i in n_specs]], axis = 1)

for i in n_specs:
    diversity["r_spec_rel_{}".format(2*i)] = (
                diversity["r_spec_{}".format(2*i)]/diversity["tot_prob"])
    
diversity["rel_mean"] = np.sum(n_specs*diversity[["r_spec_rel_{}".format(2*i)
                                for i in n_specs]], axis = 1)

diversity = diversity.loc[np.argsort(diversity["mean"])]

div_arr = diversity[["r_spec_{}".format(2*i)
                                for i in n_specs]].values

if __name__ == "__main__":
    fig, ax = plt.subplots(2,1,sharex = True, sharey = True)
    loc = np.arange(len(trait_combs))
    for i in range(n_spec_max):
        ax[0].bar(loc, div_arr[:,i], bottom = np.sum(div_arr[:,:i], axis = 1))
        
    ax_mean = ax[0].twinx()
    ax_mean.plot(loc, diversity["mean"], 'ko')
    
    ax[1].set_xticks(loc)
    ax[1].set_xticklabels(diversity["comb"], rotation = 90)
    
    div_arr = diversity[["r_spec_rel_{}".format(2*i)
                                    for i in n_specs]].values
    for i in range(n_spec_max):
        ax[1].bar(loc, div_arr[:,i], bottom = np.sum(div_arr[:,:i], axis = 1))
        
    ax_mean = ax[1].twinx()
    ax_mean.plot(loc, diversity["rel_mean"], 'ko')
    
    
    fig.tight_layout()
    fig.savefig("Figure_diversity_full_coms.pdf")