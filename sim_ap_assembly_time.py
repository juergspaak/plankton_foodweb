import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer


n_coms = 1000
n_rich = [1,2,3,4,5,10,15,20,25]


rich_all = np.full((len(n_rich),  n_coms, 2), np.nan)
res_all = np.full((len(n_rich), n_coms, 3), np.nan)
dens_all = np.full((len(n_rich), n_coms, 2), np.nan)
time = np.full(len(n_rich), np.nan)

all_start = timer()

for i, rich in enumerate(n_rich):
    start = timer()
    traits = gp.generate_plankton(rich, n_coms)
    env = gp.generate_env(n_coms)
    traits = gp.phytoplankton_equilibrium(traits, env)
    
    # simulate densities
    richness, id_survive, res_equi, dens_equi = assembly_richness(
                    traits, env, plot_until = 0, ret_all = True)
    rich_all[i] = richness
    res_all[i] = res_equi
    dens_all[i] = np.nansum(dens_equi, axis = -1)
    dens_all[dens_all == 0] = np.nan
    time[i] = timer()-start
    print(i, rich, np.mean(rich_all[i], axis = 0),
          timer()-start, timer()-all_start)
    
    np.savez("Data_assembly_years", rich = rich_all, res = res_all,
             dens = np.log(dens_all), change_var = n_rich, time = time)
