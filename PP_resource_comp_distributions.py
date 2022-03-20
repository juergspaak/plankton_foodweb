import numpy as np
import matplotlib.pyplot as plt

import generate_plankton as gp
import plankton_growth as pg

n_coms = 5000
r_phyto = 2

traits = gp.generate_plankton(r_phyto, n_coms, r_zoop = 1)
traits["a"] /= 1000
traits["k_l"] /= 100000
env = gp.generate_env(n_coms)
env["I_in"][:] = 10000
traits = gp.phytoplankton_equilibrium(traits, env)

env_com = {key: env[key][:,np.newaxis] for key in env.keys()}
# compute how often can species invade
invasion = np.empty((n_coms, r_phyto))
growth = np.empty((2, n_coms))
for i in range(2):
    N_star = np.array([traits["R_star_n_res"][:, i],
                       traits["R_star_p_res"][:, i],
                       *traits["N_star_P_res"].T]).T
    N_star[:, 2+i] = 0
    growth[i] = (pg.phyto_growth(N_star[:,:2], N_star[:,2:], traits, env_com)
                         - env_com["d"])[:, 1-i]

print("invasion possible: ", np.mean(np.all(growth>0, axis = 0)))



traits = {key: np.log(traits[key]) for key in traits.keys()}




fig = plt.figure()
plt.scatter(traits["R_star_p"], traits["R_star_n"], s = 1)
plt.xlabel("$R^*_N$", fontsize = 14)
plt.ylabel("$R^*_P$", fontsize = 14)
ind = np.all(np.isfinite(traits["R_star_p"]*traits["R_star_n"]), axis = 1)


# is coexistence potentially possible?
# i.e. species have a tradeoff in R^*
pot_coex = (np.nanargmax(traits["R_star_p"], axis = 1)
            != np.nanargmax(traits["R_star_n"], axis = 1))

print("Competitive dominance: ", np.round(1 - np.mean(pot_coex),2))

# for the species that potentially coexist, how many of them have priority
# effects?
keys = ["c_n", "c_p", "R_star_n", "R_star_p"]
traits_coex = {key: traits[key][pot_coex] for key in keys}

traits_coex["c"] = np.exp([traits_coex["c_n"], traits_coex["c_p"]])
traits_coex["c"] = traits_coex["c"]/np.linalg.norm(traits_coex["c"], axis = 0)

priority = np.argmax(traits_coex["R_star_n"], axis = 1) == np.argmax(traits_coex["c"][0], axis = -1)
print("Priority: ", np.round(np.mean(priority), 2))

# how often can we invade?

