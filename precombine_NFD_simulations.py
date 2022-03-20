import numpy as np
import pandas as pd
import generate_plankton as gp

n_prec = 9
NFD_keys = ["ND", "F", "r_i", "mu", "eta", "N_star", "issue"]
env_keys = ["I_in", "P", "N", "d", "zm"]
comb_traits = ["h_zp", "s_zp_raw", "s_zp"]
keys = (NFD_keys + env_keys + comb_traits
        + list(gp.pt.phyto_traits) + list(gp.zt.zoop_traits))
r_phyto = r_zoo = 20

##############################################################################
# changes in mean

traits = ["c_Z", "k_Z", "e_P", "R_P", "h_zp"]


counter = 0
path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"
file = "sim_NFD_mean_1000_{}_{}_{}.npz"
k_max = 0
for i in range(n_prec):
    for trait in traits:
        combined_data = {"r_phyto": r_phyto, "r_zoo": r_zoo}
        for k in range(20):
            try:
                data = np.load(path + file.format(trait, i, k), allow_pickle=True)
                id_comp = np.sum(data["issue"] != None)
                k_max = max(k, k_max)
                try:
                    combined_data["n_coms"] += id_comp
                    for key in keys:
                        combined_data[key] = np.append(combined_data[key], data[key][:id_comp], axis = 0)
                except KeyError:
                    combined_data["n_coms"] = id_comp
                    for key in keys:
                        combined_data[key] = data[key][:id_comp]
            except (FileNotFoundError, KeyError):
                continue

        np.savez(path + "sim_NFD_mean_combined_{}_{}.npz".format(trait, i), **combined_data)

#"""

##############################################################################
# changes in correlation

gleaner = ["c_Z:m_Z", "c_Z:k_Z", "c_Z:mu_Z", "c_n:k_n", "c_p:k_p", "k_Z:mu_Z"]
defense = ["R_P:mu_P", "R_P:k_n"]
super_resource = ["R_P:a", "R_P:e_P"]

corrs = gleaner + defense + super_resource
tradeoff = [gleaner, defense, super_resource]
file = "assembly_corr_1000_{}_{}_{}_{}.npz"
for corr in corrs:
    for i in range(n_prec):
        combined_data = {"r_phyto": r_phyto, "r_zoo": r_zoo}
        for k in range(10):
            try:
                data = np.load(path +
                               file.format(*corr.split(":"), i, k), allow_pickle = True)
                id_comp = np.sum(data["issue"] != None)
                try:
                    combined_data["n_coms"] += id_comp
                    try:
                        combined_data["corr"] = data["corr"]
                    except KeyError:
                        pass
                    for key in keys:
                        combined_data[key] = np.append(combined_data[key], data[key][:id_comp], axis = 0)
                except KeyError:
                    combined_data["n_coms"] = id_comp
                    try:
                        combined_data["corr"] = data["corr"]
                    except KeyError:
                        pass
                    for key in keys:
                        combined_data[key] = data[key][:id_comp]
            except (FileNotFoundError, KeyError):
                continue
        
        np.savez(path + "sim_NFD_corr_combined_{}_{}_{}.npz".format(*corr.split(":"), i), **combined_data)
#"""
