# simulate example dynamics over time
import generate_plankton as gp
import copy

import numpy as np

from assembly_time_fun import assembly_richness

np.random.seed(1)

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/fluctuations/"

##############################################################################
# let each resource fluctuate on its own with different periods
periods = [5, 10, 25, 50, 100]

n_spec = 20
n_coms = 100
traits = gp.generate_plankton(n_spec, n_coms, fluct_temp = True)


for pred in ["", "_no_pred"]:
    if pred == "_no_pred":
        # remove predators to potentially amplify effects of fluctuations
        traits["mu_Z"] /= 1e5
        traits["m_Z"] += 10
    for period in periods:
        save = path + "data_fluctuations_temperature{}_{}.npz".format(pred, period)
        try:            
            data = np.load(save)
            continue
        except FileNotFoundError:
            np.savez(save, prelim = 1)
        
        print("\n\n", period)
        env = gp.generate_env(n_coms, fluct_env = ["T"])
        traits = gp.phytoplankton_equilibrium(traits, env)
        
        env["freq_T"][:] = period
        env["T"][:] = 20.0
        env["ampl_T"][:] = 0.25
        richness, present, res, dens = assembly_richness(
                        traits, env, plot_until = 3, ret_all = True,
                        save = save, pr=True)
        
        np.savez(save, i = n_coms, present = present, res = res,
                 dens = dens, **traits, **env )

# faster invasion of new species
n_spec = 30
traits = gp.generate_plankton(n_spec, n_coms, fluct_temp = True)      
        
for pred in ["", "_no_pred"]:
    if pred == "_no_pred":
        # remove predators to potentially amplify effects of fluctuations
        traits["mu_Z"] /= 1e5
        traits["m_Z"] += 10
    for period in periods:
        save = path + "data_fluctuations_temperature{}_{}_fast_invasion.npz".format(pred, period)
        try:            
            data = np.load(save)
            continue
        except FileNotFoundError:
            np.savez(save, prelim = 1)
        
        print("\n\n", period)
        env = gp.generate_env(n_coms, fluct_env = ["T"])
        traits = gp.phytoplankton_equilibrium(traits, env)
        
        env["freq_T"][:] = period
        env["T"][:] = 20.0
        env["ampl_T"][:] = 0.25
        richness, present, res, dens = assembly_richness(
                        traits, env, plot_until = 3, ret_all = True,
                        save = save, pr=True, time_org = np.array([0,90]))
        
        np.savez(save, i = n_coms, present = present, res = res,
                 dens = dens, **traits, **env )
        
# generate reference cases
save = path + "data_fluctuations_temperature_ref_fast_invasion.npz"
try:            
    data = np.load(save)
except FileNotFoundError:
    print("reference case")
    np.savez(save, prelim = 1)
    env["ampl_T"][:] = 0
    traits = gp.phytoplankton_equilibrium(traits, env)
    richness, present, res, dens = assembly_richness(
                        traits, env, plot_until = 3, ret_all = True,
                        save = save, pr=True, time_org = np.array([0,90]))