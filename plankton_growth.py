import numpy as np
from zoop_traits import uc
import warnings
#warnings.simplefilter("error", RuntimeWarning)


def light_growth(N, t, env):
    # light limited growth at grtowth rate I_in
    tot_abs = np.sum(env["zm"]*N*t["a"], axis = -1, keepdims = True)
    # returns nan if tot_abs is zero, but this is handled correctly in next step
    return 1/tot_abs*np.log((t["k_l"] + env["I_in"])/
                (t["k_l"] + env["I_in"]*np.exp(-tot_abs)))


def phyto_growth(res, N_phyto, t, env):
    growth = np.array([res[...,[0]]/(res[...,[0]] + t["k_n"]),
                       res[...,[1]]/(res[...,[1]] + t["k_p"]),
                       light_growth(N_phyto, t, env)])
    # light growth might be nan if phyto densities are zero
    return t["mu_P"]*np.nanmin(growth, axis = 0)

def grazing(N_phyto,t):
    # how much zooplankton eat of each specific phytoplankton
    numerator = np.expand_dims(t["c_Z"],-1)*t["s_zp"]*np.expand_dims(t["e_P"],-2)
    
    denom = 1 + t["c_Z"]*np.einsum("...zp,...zp,...p->...z",
                                   t["h_zp"], t["s_zp"], N_phyto)
    return numerator/np.expand_dims(denom,-1)

def per_cap_plankton_growth(N, t, env):
    
    N = N.copy()
    N[~(1e-8<N)] = 1e-8 # species with too small densities cause numerical problems
    # separate densities into phytoplankton and zooplankton
    res = N[...,:2] # nitrogen and phosphorus concentration
    N_phyto = N[...,2:(t["r_phyto"] + 2)]
    N_zoo = N[...,2 + t["r_phyto"]:]
    
    # compute growth rate of zooplankton
    grazed = grazing(N_phyto,t)
    R_Z = np.einsum("...p,...zp->...z",t["R_P"]*N_phyto,grazed)
    dZ_dt = t["mu_Z"]*R_Z/(R_Z + t["k_Z"]) - t["m_Z"]
    
    # growth rate of phytoplankton
    growth_P = phyto_growth(res, N_phyto, t, env)
    dP_dt = (growth_P
             - uc["h_day"]/uc["ml_L"]*np.einsum("...z,...zp->...p", N_zoo, grazed)
             - env["d"])
    
    # change in resources
    dres_dt = np.array([env["d"]*(env["N"] - res[...,0]) -
                     uc["ml_L"]*np.sum(t["c_n"]*growth_P*N_phyto, axis = -1),
              env["d"]*(env["P"] - res[...,1])
                  - uc["ml_L"]*np.sum(t["c_p"]*growth_P*N_phyto, axis = -1)]).T
    
    return np.concatenate((dres_dt/res, dP_dt, dZ_dt), axis = -1)

def plankton_growth(N, t, env):
    return N*per_cap_plankton_growth(N, t, env)

def convert_ode_to_log(logN, t, env, time = 0):
    
    # find current environmental settings
    env_c = {
        R: env[R]*(1 + env["ampl_"+R]
        *np.sin(2*np.pi*time/env["freq_"+R] + env["phase_"+R])) 
        for R in ["N", "P", "I_in", "d", "zm"]}
    
    with warnings.catch_warnings(record = True):
        N = np.exp(logN)
        N[N>1e20] = 1e20 # prevent overflow
    # dlog(N)/dt = 1/N dN/dt
    return per_cap_plankton_growth(N, t, env_c)

###############################################################################
if __name__ == "__main__":
    # simulate example dynamics over time
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    import generate_plankton as gp
    
    from assembly_time_fun import assembly_richness
    
    n_spec = 20
    n_coms = 5
    traits = gp.generate_plankton(n_spec, n_coms)
    env = gp.generate_env(n_coms, fluct_env=["N", "P", "I_in"])
    traits = gp.phytoplankton_equilibrium(traits, env)

    # simulate densities
    richness, present, res, dens = assembly_richness(
                    traits, env, plot_until = 5, ret_all = True)