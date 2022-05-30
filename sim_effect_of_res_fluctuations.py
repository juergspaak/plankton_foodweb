# simulate example dynamics over time
import generate_plankton as gp
import copy

import numpy as np

from assembly_time_fun import assembly_richness

np.random.seed(1)

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/fluctuations/"

##############################################################################
# let each resource fluctuate on its own with different periods
variables = ["N", "P", "I_in", "d", "zm"]
periods = [10, 25, 50, 100]

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

###############################################################################
# remove predators to potentially amplify effects of fluctuations
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
        
###############################################################################
# combined fluctuations of N and P, but different phase differences
env = gp.generate_env(n_coms, fluct_env = ["N", "P"])
# both must have same frequency
env["freq_N"] = env["freq_P"]

envs = {"in_phase": copy.deepcopy(env),
        "quarter_phase": copy.deepcopy(env),
        "off_phase": copy.deepcopy(env),
        "no_fluct": copy.deepcopy(env)}


# correct phase of different flucutations
envs["in_phase"]["phase_N"] = envs["in_phase"]["phase_P"]
envs["quarter_phase"]["phase_N"] = envs["quarter_phase"]["phase_P"] + np.pi/2
envs["off_phase"]["phase_N"] = envs["off_phase"]["phase_P"] + np.pi
envs["no_fluct"]["ampl_N"][:] = 0
envs["no_fluct"]["ampl_P"][:] = 0

traits = gp.generate_plankton(n_spec, n_coms)
traits = gp.phytoplankton_equilibrium(traits, env)

for key in envs.keys():
    save = path + "data_fluctuations_N_P_{}.npz".format(key)
    try:            
        data = np.load(save)
        continue
    except FileNotFoundError:
        np.savez(save, prelim = 1)
    print(save)
    richness, present, res, dens = assembly_richness(
                            traits, envs[key], plot_until = 0, ret_all = True,
                            save = save)
    
##############################################################################
# remove predator dependency
# same simulations but without predators
traits["mu_Z"] /= 1e5
traits["m_Z"] += 10

for key in envs.keys():
    save = path + "data_fluctuations_N_P_no_pred_{}.npz".format(key)
    try:            
        data = np.load(save)
        continue
    except FileNotFoundError:
        np.savez(save, prelim = 1)
    print(save)
    richness, present, res, dens = assembly_richness(
                            traits, envs[key], plot_until = 0, ret_all = True,
                            save = save)
