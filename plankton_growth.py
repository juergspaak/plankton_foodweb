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
    
    dres_dt = np.array([env["d"]*(env["N"] - res[...,0]) -
                     uc["ml_L"]*np.sum(t["c_n"]*growth_P*N_phyto, axis = -1),
              env["d"]*(env["P"] - res[...,1])
                  - uc["ml_L"]*np.sum(t["c_p"]*growth_P*N_phyto, axis = -1)]).T
    
    return np.concatenate((dres_dt/res, dP_dt, dZ_dt), axis = -1)

def plankton_growth(N, t, env):
    return N*per_cap_plankton_growth(N, t, env)

def convert_ode_to_log(logN, t, env):
    with warnings.catch_warnings(record = True):
        N = np.exp(logN)
        N[N>1e20] = 1e20 # prevent overflow
    # dlog(N)/dt = 1/N dN/dt
    return per_cap_plankton_growth(N, t, env)

###############################################################################
if __name__ == "__main__":
    # simulate example dynamics over time
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    import generate_plankton as gp
    r_phyto = 4
    r_zoo = 1
    n_coms = int(1e4)
    traits, env = gp.generate_communities(r_phyto, n_coms)
    
    """
    # test whether all equilibria are computed correctly
    N = np.append(traits["N_star_P"], traits["N_star_Z"], axis = 1)
    
    # multispecies equilibrium
    if np.amax(np.abs(plankton_growth(N, traits, env)))>1e-8:
        raise ValueError("Community equilibrium not computed correctly")
        
    if r_phyto == 1:    
        if np.amax(np.abs(nitrogen_growth(traits["N_star_P_n"], traits, env)
                          -env["d"]))>1e-8:
            raise ValueError("Nitrogen equilibrium not computed correctly")
            
        if np.amax(np.abs(phosphor_growth(traits["N_star_P_p"], traits, env)
                          -env["d"]))>1e-8:
            raise ValueError("Nitrogen equilibrium not computed correctly")
        
        # competition for light
        if np.amax(np.abs(light_growth(traits["N_star_P_l"], traits, env)
                          -env["d"]))>1e-8:
            pass # returns correct for most, some converge not fast enough
            #raise ValueError("Light equilibrium not computed correctly")
    """
    
    # select a random community and simulate it
    ti, envi, i = gp.select_i(traits, env)
    
    # add environmental noise
    N = np.concatenate(([envi["N"], envi["P"]],ti["N_star_P"], ti["N_star_Z"]), axis = 0)
    err = 1e-0
    N *= np.random.uniform(1-err,1+err,N.shape)
       
    
    time = [0,1000]
    
    fig, ax = plt.subplots(3, sharex = True)
    
    
    sol = solve_ivp(lambda t, logN: convert_ode_to_log(logN, ti, envi),
                         time, np.log(N), method = "LSODA")   
    
    ax[0].semilogy(sol.t, np.exp(sol.y[:2]).T, '-')
    ax[1].semilogy(sol.t, np.exp(sol.y[2:(2+r_phyto)]).T, '-')
    ax[2].semilogy(sol.t, np.exp(sol.y[2 + r_phyto:]).T, '--')
    
    ax[1].set_ylim([1e-5, None])
    ax[2].set_ylim([1e-3, None])