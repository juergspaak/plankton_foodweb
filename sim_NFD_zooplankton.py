import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from NFD_definitions.numerical_NFD import NFD_model, InputError
import generate_plankton as gp
import plankton_growth as pg
from NFD_zooplankton import growth_zooplankton
from scipy.integrate import solve_ivp

from timeit import default_timer as timer

n_coms = int(1e4)
r_phyto = 2
traits = gp.generate_plankton(r_phyto, n_coms)
env = gp.generate_env(n_coms)
traits = gp.phytoplankton_equilibrium(traits, env)

traits["ND"] = np.full(traits["size_Z"].shape, np.nan)
traits["FD"] = np.full(traits["size_Z"].shape, np.nan)
traits["eta"] = np.full(traits["size_Z"].shape, np.nan)
traits["mu"] = np.full(traits["size_Z"].shape, np.nan)
traits["r_i"] = np.full(traits["size_Z"].shape, np.nan)
traits["c"] = np.full(traits["size_Z"].shape, np.nan)

trait_comb = np.random.choice(["", "k_n", "s_zp"], traits["n_coms"])
species_info = gp.zt.zoop_traits + ["ND", "FD", "mu", "eta", "r_i"]

def find_equi(ind_present, ti, envi):
    
    N_fix = np.zeros(ti["r_phyto"] + ti["r_zoo"])
    def internal(t,N):
        N_fix[ind_present] = N
        return pg.plankton_growth(N_fix, ti, envi)[ind_present]
    N_start = ti["r_phyto"]*[10] + (ti["r_zoo"]-1)*[1]
    time = [0,5*365]
    sol = solve_ivp(internal, time, N_start, method = "LSODA")
    plt.figure()
    plt.plot(sol.t, sol.y.T)
    return sol.y[ti["r_phyto"]:,-1]

save = 5
start = timer()
found = {"":0, "s_zp":0, "k_n":0}
for i in range(traits["n_coms"]):
    try:
        ti, envi, i = gp.select_i(traits, env, i)
        if trait_comb[i] == "s_zp":
            ti["s_zp"][:] = 1
        elif trait_comb[i] == "k_n":
            ti["k_n"][:] = np.exp(np.mean(np.log(ti["k_n"])))
        
        pars = {}
        pars = NFD_model(growth_zooplankton, pars = pars, args  = (ti, envi),
                         xtol = 1e-4)
        for key in ["ND", "FD", "eta", "mu", "r_i"]:
            traits[key][i] = pars[key].copy()
        traits["c"][i] = pars["c"][[0,1],[1,0]]
        found[trait_comb[i]] += 1
        print(i, found, timer()-start, traits["n_coms"])
    except InputError:
        continue
    
    if np.sum([found[key] for key in found.keys()]) % 100 == 99:
        ind = np.isfinite(traits["ND"][:,0])
        data = pd.DataFrame({key + str(i): traits[key][ind,i]
                             for key in species_info
                             for i in range(2)})
        data["trait_comb"] = trait_comb[ind]
        data.to_csv("NFD_data_{}.csv".format(save), index = False)
ind = np.isfinite(traits["ND"][:,0])
data = pd.DataFrame({key + str(i): traits[key][ind,i]
                     for key in species_info
                     for i in range(2)})
data["trait_comb"] = trait_comb[ind]
data.to_csv("NFD_data_{}.csv".format(save), index = False)
"""
fig = plt.figure()
ax = plt.gca()
c = np.all(traits["r_i"]>0, axis = 1, keepdims=True)*np.ones(ti["r_zoo"])
plt.scatter(traits["ND"], 1-1/(1-traits["FD"]), c =c,
            s = 1)
perc = 0
plt.xlim(np.nanpercentile(traits["ND"], [perc,100-perc]))
plt.ylim(np.nanpercentile(traits["FD"], [perc,100-perc]))
plt.xlim([-2,2])
plt.ylim([-3,2])
plt.plot(plt.gca().get_xlim(), plt.gca().get_xlim(), color = "k")
ax.axvline(0, color = 'grey')
ax.axvline(1, color = 'grey')
ax.axhline(1, color = 'grey')
ax.axhline(0, color = 'grey')
plt.xlabel("Niche differences")
plt.ylabel("Fitness differences")
title = "Reference"
title = "No selectivity"
title = "No halfsaturation zooplankton"
title = "No halfsaturation phytoplankton"
plt.title(title)


fig.savefig("NFD_distribution_{}.pdf".format(title))"""


   