import numpy as np

from timeit import default_timer as timer


import generate_plankton as gp
from NFD_equilibrium import fast_NFD

n_prec = 9

n_coms = 1000


const_traits = ["R_P",  "c_Z", "k_Z", "h_zp", "e_P"]

add_var = np.linspace(-1,1, n_prec)
all_start = timer()
for k in range(20):
    for j, trait in enumerate(const_traits):
        if trait in gp.pt.std_phyto.columns:
            add_mean = add_var*gp.pt.std_phyto[trait].values
        elif trait in gp.zt.std_zoop.columns:
            add_mean = np.linspace(-1,1, n_prec)*gp.zt.std_zoop[trait].values
        elif trait == "h_zp":
            traits = gp.generate_plankton(2,n_coms, evolved_zoop=True)
            std_h_zp = np.std(np.log(traits["h_zp"]))
            add_mean = np.linspace(-1,1, n_prec)*std_h_zp
            
        for i in range(n_prec):
            save = "data/sim_NFD_mean_{}_{}_{}_{}.npz".format(n_coms, trait, i, k)
            try:
                data = np.load(save)
                continue
            except FileNotFoundError:
                np.savez(save, prelim = 1)
            
            print(trait, i, k)
            start = timer()
            traits = gp.generate_plankton(2, n_coms, diff_mean = {trait: add_mean[i]})
            env = gp.generate_env(n_coms)
            
            # simulate densities
            results, issues = fast_NFD(traits, env, save)
            
            np.savez(save, i = i, **results, issue = issues, **traits, **env,
                     time = timer() - start)