import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

from timeit import default_timer as timer
import phyto_growth as pg

itera = 10000
lim_factors = np.random.choice(["NPL", "NP", "NL", "PL"], itera,
                               replace = True)
const_traits = np.random.choice(["None", "mu", "k", "c"], itera,
                                replace = True)
constant_traits = {"None": [],
                   "mu": ["mu_l", "mu_p", "mu_n"],
                   "k": ["k_n", "k_p", "alpha"],
                   "c": ["c_n", "c_p", "k"]}

P_supply = np.random.uniform(0,3, itera)
N_supply = np.random.uniform(0,30, itera)
I_in_supply = np.random.uniform(30,200, itera)
loss_rate = np.random.uniform(0.05, 0.3, itera)
zm = np.random.uniform(1,10, itera)


time = np.linspace(0,1000,1001)
start = timer()

r_start = np.random.randint(2,10, itera)
n_com = np.full(itera, 10)

df = pd.DataFrame({"lim_factors": lim_factors, "const_traits": const_traits,
                   "N": N_supply, "P": P_supply, "I_in": I_in_supply,
                   "loss_rate": loss_rate, "zm": zm,
                   "r_start": r_start, "n_com": n_com})
rich_values = ["richness", "prob_1", "prob_2", "prob_3", "prob_high"]
for key in rich_values:
    df[key] = np.nan
for i in range(itera):
    com_shape = (r_start[i],n_com[i])
    env = {"I_in": I_in_supply[i],
       "P": P_supply[i],
       "N": N_supply[i],
       "m": loss_rate[i],
       "zm": zm[i]}
    traits = pg.generate_phytoplankton(*com_shape)
    
    for trait in constant_traits[const_traits[i]]:
        traits[trait][:] = np.average(traits[trait], axis = 0)
    
    sol = odeint(lambda N, t: N*pg.phyto_growth(N, traits, env = env,
                           limiting_res = lim_factors[i]),
                 np.full(traits["mu_l"].size, 1e7),
                 time)
    sol.shape = (-1, ) + com_shape
    rel_abund = sol[-1]/np.sum(sol[-1], axis = 0)
    richness = (rel_abund>1e-2).sum(axis = 0)
    if max(richness>3):
        # run for longer time to reach exclusion
        sol = odeint(lambda N, t: N*pg.phyto_growth(N, traits, env = env,
                           limiting_res = lim_factors[i]),
                 np.full(traits["mu_l"].size, 1e7),
                 10*time)
    sol.shape = (-1, ) + com_shape
    rel_abund = sol[-1]/np.sum(sol[-1], axis = 0)
    richness = (rel_abund>1e-2).sum(axis = 0)
    df.loc[i, rich_values] = [np.average(richness),
          sum(richness == 1), sum(richness == 2), sum(richness == 3),
          sum(richness>3)]
    
    print(i, timer()-start, (timer()-start)/(i+1), (timer()-start)/(i+1)*(itera-i))
    
df.to_csv("simulation_result.csv", index = False)