import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import generate_plankton as gp

n_spec_max = 4
n_specs = np.arange(1, n_spec_max + 1)
n_coms = int(1e4)

trait_combs = np.append(gp.pt.trait_names, gp.zt.zoop_traits)
trait_combs = [[], ["s_zp"], ["h_zp"]] + [[trait] for trait in trait_combs]
trait_combs += [["c_p", "c_n", "a", "c_Z"],["k_p", "k_n"],
                ["mu_P", "mu_Z"]]
trait_combs = np.array(trait_combs)

columns = ["comb", "n_comb"] + ["r_spec_{}".format(2*i)
                                for i in n_specs]

diversity = pd.DataFrame(data = np.nan, columns = columns,
                         index = np.arange(len(trait_combs)))

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

diversity = diversity.loc[np.argsort(diversity["mean"])]

div_arr = diversity[["r_spec_{}".format(2*i)
                                for i in n_specs]].values

fig = plt.figure()
loc = np.arange(len(trait_combs))
for i in range(n_spec_max):
    plt.bar(loc, div_arr[:,i], bottom = np.sum(div_arr[:,:i], axis = 1))
    
plt.plot(loc, diversity["mean"], 'ko')

plt.xticks(loc, diversity["comb"], rotation = 90)

fig.savefig("Figure_diversity_full_coms.pdf")