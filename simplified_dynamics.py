import numpy as np
import matplotlib.pyplot as plt

from zoop_traits import uc
import generate_plankton as gp
import plankton_growth as pg

def generate_communities(r_phyto, n_coms, evolved_zoop = True, r_zoo = None,
                         monoculture_equi = True, size_P = None, diff_mean = {},
                         tradeoffs = {}, size_Z = None, diff_std = {}, corr_phyto = None,
                         corr_zoo = None):
    traits = gp.generate_plankton(r_phyto, n_coms, r_zoo, evolved_zoop=evolved_zoop,
                               size_P = size_P, diff_mean = diff_mean, tradeoffs = tradeoffs,
                               size_Z = size_Z, diff_std = diff_std, corr_phyto = corr_phyto,
                               corr_zoo = corr_zoo)
    
    env = gp.generate_env(n_coms)
    
    traits = community_equilibrium(traits, env)
    ind1 = np.isfinite(traits["N_star_P"]).all(axis = -1)
    ind2 = np.isfinite(traits["N_star_Z"]).all(axis = -1)
    ind = ind1 & ind2
    
    traits = {key: traits[key][ind] for key in gp.select_keys(traits)}
    traits["r_phyto"] = r_phyto
    traits["r_zoo"] = traits["size_Z"].shape[-1]
    traits["n_coms"] = len(traits["size_P"])
    env = {key: env[key][ind] for key in env.keys()}
    n_coms = len(traits["mu_P"])
    
    if monoculture_equi:
        traits = gp.phytoplankton_equilibrium(traits, env)
    
    return traits, env

def community_equilibrium(tr, env):
    try:
        if env["zm"].ndim == 1:
            env = {key: np.expand_dims(env[key],-1) for key in env.keys()}
    except AttributeError:
        pass
    
    # equilibrium resource intake by zooplankton
    tr["R_star_Z"] = tr["k_Z"]*tr["m_Z"]/(tr["mu_Z"]-tr["m_Z"])
    
    c_Z = np.expand_dims(tr["c_Z"],-1)
    tr["A_zoop"] = c_Z*tr["s_zp"]*(-tr["h_zp"]*tr["R_star_Z"][...,np.newaxis]
                                   + (tr["R_P"]*tr["e_P"])[:,np.newaxis])
    
    # equilibrium density of phytoplankton species
    ind = np.linalg.cond(tr["A_zoop"]) < 1e10
    tr["N_star_P"] = np.full(tr["mu_P"].shape, np.nan)
    tr["N_star_P"][ind] = np.linalg.solve(tr["A_zoop"][ind],
                                          tr["R_star_Z"][ind])
    tr["N_star_P"][tr["N_star_P"]<0] = np.nan
   
    #####################
    # compute growth rates of phytoplankton
    #####################
    # compute resource concentrations for nitrogen
    tr["N_conc_n"] = (env["d"]*env["N"]
                    - uc["ml_L"]*np.sum(tr["N_star_P"]*tr["c_n"], axis = -1,
                                        keepdims=True))
    tr["N_conc_n"][tr["N_conc_n"]<0] = 0
    # compute resource concentrations for phosphorus    
    tr["N_conc_p"] = (env["d"]*env["P"]
                    - uc["ml_L"]*np.sum(tr["N_star_P"]*tr["c_p"], axis = -1,
                                        keepdims=True))
    tr["N_conc_p"][tr["N_conc_p"]<0] = 0
    
    # outcoming light concentration
    tr["tot_abs"] = env["zm"]*np.sum(tr["a"]*tr["N_star_P"], axis = -1,
                                     keepdims=True)
    
    # given resource concentrations, compute growth rates
    tr["growth_p"] = tr["N_conc_p"]/(tr["k_p"] + tr["N_conc_p"])
    tr["growth_n"] = tr["N_conc_n"]/(tr["k_n"] + tr["N_conc_n"])
    tr["growth_l"] = (1/tr["tot_abs"]
                    *np.log((tr["k_l"] + env["I_in"])/
                            (tr["k_l"] + env["I_in"]*np.exp(-tr["tot_abs"]))))
    
    # compute growth rate of phytoplankton
    tr["growth_P"] = tr["mu_P"]*np.amin([tr["growth_p"],
                                         tr["growth_n"],
                                         tr["growth_l"]], axis = 0)
    tr["limit_res_Z"] = np.argmin([tr["growth_p"],
                                         tr["growth_n"],
                                         tr["growth_l"]], axis = 0)
    

    numerator = uc["h_day"]/uc["ml_L"]*c_Z*tr["s_zp"]*tr["e_P"][:,np.newaxis,:]
    denom = 1 + tr["c_Z"]*np.einsum("...zp,...zp,...p->...z",
                                    tr["h_zp"],tr["s_zp"],tr["N_star_P"])

    
    tr["A_phyto"] = numerator/denom[...,np.newaxis]
    tr["A_phyto"] = np.moveaxis(tr["A_phyto"], -2,-1) # transposing
    
    # compute zooplankton equilibrium density
    # identify well behaved matrices
    ind = np.linalg.cond(tr["A_phyto"]) < 1e10
    tr["N_star_Z"] = np.full(tr["mu_Z"].shape, np.nan)
    tr["N_star_Z"][ind] = np.linalg.solve(tr["A_phyto"][ind],
                                     (tr["growth_P"] - env["d"])[ind])
    
    # set phytoplankton densities of inexistent zoop to nan
    tr["N_star_P_raw"] = tr["N_star_P"].copy()
    tr["N_star_P"][tr["N_star_Z"]<0] = np.nan
    tr["N_star_Z"][tr["N_star_Z"]<0] = np.nan
    return tr

def test_equilibrium(tr, env, i = None):
    i = np.argmax(np.isfinite(traits["N_star_Z"]).all(axis = 1))
    tr_i, envi, i = gp.select_i(traits, env, i)
    
    # is growth rates 0
    growth = pg.plankton_growth(np.concatenate((tr_i["N_conc_n"], tr_i["N_conc_p"],
                                      tr_i["N_star_P"], tr_i["N_star_Z"])),
                                      tr_i, envi)
    
    return np.allclose(growth[2:], 0, rtol = 1e-8)
    

n_coms = 100
traits = gp.generate_plankton(3, n_coms)
env = gp.generate_env(n_coms)

traits = community_equilibrium(traits, env)
print(test_equilibrium(traits, env))