

# simulate example dynamics over time
import generate_plankton as gp
import plankton_growth as pg
import copy

import numpy as np

from assembly_time_fun import assembly_richness

np.random.seed(1)

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/fluctuations/"

variables = ["N", "P", "I_in", "d", "zm"]
periods = [1, 10, 25, 50, 100]

n_spec = 20
n_coms = 100
traits = gp.generate_plankton(n_spec, n_coms)

for variable in variables:
    for period in periods:
        save = path + "data_fluctuations_{}_{}.npz".format(variable, period)
        try:            
            data = np.load(save)
            continue
        except FileNotFoundError:
            np.savez(save, prelim = 1)
        
        print("\n\n", variable, period)
        env = gp.generate_env(n_coms, fluct_env = [variable])
        traits = gp.phytoplankton_equilibrium(traits, env)
        
        env["freq_" + variable][:] = period
        richness, present, res, dens = assembly_richness(
                        traits, env, plot_until = 0, ret_all = True,
                        save = save)
        
        np.savez(save, i = n_coms, present = present, res = res,
                 dens = dens, **traits, **env )

traits["mu_Z"] /= 1e5
traits["m_Z"] += 10
for variable in variables:
    for period in periods:
        save = path + "data_fluctuations_no_pred_{}_{}.npz".format(variable, period)
        try:            
            data = np.load(save)
            continue
        except FileNotFoundError:
            np.savez(save, prelim = 1)
        
        print("\n\n", variable, period)
        env = gp.generate_env(n_coms, fluct_env = [variable])
        traits = gp.phytoplankton_equilibrium(traits, env)
        
        env["freq_" + variable][:] = period
        richness, present, res, dens = assembly_richness(
                        traits, env, plot_until = 0, ret_all = True,
                        save = save)
        
        np.savez(save, i = n_coms, present = present, res = res,
                 dens = dens, **traits, **env )

env = gp.generate_env(n_coms, fluct_env = ["N", "P"])
# both must have same frequency
env["freq_N"] = env["freq_P"]

envs = {"in_phase": copy.deepcopy(env),
        "quarter_phase": copy.deepcopy(env),
        "off_phase": copy.deepcopy(env)}


# correct phase of different flucutations
envs["in_phase"]["phase_N"] = envs["in_phase"]["phase_P"]
envs["quarter_phase"]["phase_N"] = envs["quarter_phase"]["phase_P"] + np.pi/2
envs["off_phase"]["phase_N"] = envs["off_phase"]["phase_P"] + np.pi

traits = gp.generate_plankton(n_spec, n_coms)
traits = gp.phytoplankton_equilibrium(traits, env)

for key in envs.keys():
    save = path + "data_fluctuations_N_P_{}.npz".format(key)
    try:            
        data = np.load(save)
        continue
    except FileNotFoundError:
        np.savez(save, prelim = 1)
    richness, present, res, dens = assembly_richness(
                            traits, envs[key], plot_until = 0, ret_all = True,
                            save = save)
