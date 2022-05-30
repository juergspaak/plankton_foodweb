# simulate example dynamics over time
import generate_plankton as gp
import copy

import numpy as np

from assembly_time_fun import assembly_richness

np.random.seed(1)

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/fluctuations/"

n_spec = 30
n_coms = 100

env = gp.generate_env(n_coms, fluct_env = ["N", "P"])
# both must have same frequency
env["freq_P"] = np.random.randint(50,300, len(env["freq_P"]))
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

for pred in ["", "_no_pred"]:
    if pred == "_no_pred":
        traits["m_Z"][:] = 10
        traits["mu_Z"][:] = 1e-5
    for key in envs.keys():
        save = path + "data_fluctuations_temp_richness{}_N_P_{}.npz".format(pred, key)
        try:            
            data = np.load(save)
            continue
        except FileNotFoundError:
            np.savez(save, prelim = 1)
        print(save)
        richness, present, res, dens = assembly_richness(
                                traits, envs[key], plot_until = 3, ret_all = True,
                                save = save, time_org = np.array([0,100]))
