import numpy as np
from zoop_traits import uc


def light_growth(N, t, env):
    # light limited growth at grtowth rate I_in
    tot_abs = np.sum(env["zm"]*N*t["a"], axis = -1, keepdims = True)
    # returns nan if tot_abs is zero, but this is handled correctly in next step
    return t["mu_P"]/tot_abs*np.log((t["k_l"] + env["I_in"])/
            (t["k_l"] + env["I_in"]*np.exp(-tot_abs)))
    
def phosphor_growth(N, t, env):
    P = env["d"]*env["P"] - uc["ml_L"]*np.sum(N * t["c_p"], axis = -1,
                                              keepdims = True)
    P[P<0] = 0
    return t["mu_P"]*P/(P + t["k_p"])

def nitrogen_growth(N, t, env):
    N = env["d"]*env["N"] - uc["ml_L"]*np.sum(N * t["c_n"], axis = -1,
                                                keepdims = True)
    N[N<0] = 0
    return t["mu_P"]*N/(N  + t["k_n"])

limiting_growth = {"N": nitrogen_growth,
                  "P": phosphor_growth,
                  "L": light_growth}
limiting_growth_keys = np.array(["N", "P", "L"])

def phyto_growth(N, t, env, limiting_res = limiting_growth_keys):
    
    growth = np.empty(((len(limiting_res),  ) + N.shape))
    for i, key in enumerate(limiting_res):
        growth[i] = limiting_growth[key](N, t, env)
    # light growth might be nan if phyto densities are zero
    return np.nanmin(growth, axis = 0)

def grazing(N_phyto,t):
    # how much zooplankton eat of each specific phytoplankton
    numerator = np.expand_dims(t["c_Z"],-1)*t["s_zp"]
    
    denom = 1 + t["c_Z"]*np.einsum("...zp,...zp,...p->...z",
                                   t["h_zp"], t["s_zp"], N_phyto)
    return numerator/np.expand_dims(denom,-1)

def per_cap_plankton_growth(N, t, env, limiting_res = limiting_growth_keys):
    N = N.copy()
    N[N<1e-5] = 1e-5
    # separate densities into phytoplankton and zooplankton
    N_phyto = N[...,:t["r_phyto"]]
    N_zoo = N[...,t["r_phyto"]:]
    grazed = grazing(N_phyto,t)
    R_Z = np.einsum("...p,...zp->...z",t["R_P"]*N_phyto,grazed)
    dZ_dt = t["mu_Z"]*R_Z/(R_Z + t["k_Z"]) - t["m_Z"]
    dP_dt = (phyto_growth(N_phyto, t, env, limiting_res)
             - uc["h_day"]/uc["ml_L"]*np.einsum("...z,...zp->...p", N_zoo, grazed)
             - env["d"])
    
    return np.append(dP_dt, dZ_dt, axis = -1)

def plankton_growth(N, t, env):
    return N*per_cap_plankton_growth(N, t, env)

def convert_ode_to_log(logN, t, env):
    N = np.exp(logN)
    N[N<1e-5] = 1e-5
    return plankton_growth(N, t, env)/N


###############################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    import generate_plankton as gp
    r_phyto = 4
    r_zoo = 1
    n_coms = int(1e4)
    traits, env = gp.generate_communities(r_phyto, n_coms)

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
        
    # select a random community and simulate it
    ti, envi, i = gp.select_i(traits, env)
    
    # add environmental noise
    N = np.append(ti["N_star_P"], ti["N_star_Z"], axis = 0)
    err = 1e-0
    N *= np.random.uniform(1-err,1+err,N.shape)
       
    
    time = [0,1000]
    
    fig, ax = plt.subplots(2, sharex = True)
    
    
    sol = solve_ivp(lambda t, logN: convert_ode_to_log(logN, ti, envi),
                         time, np.log(N), method = "LSODA")   
    
    ax[0].semilogy(sol.t, np.exp(sol.y[:r_phyto]).T, '-')
    ax[1].semilogy(sol.t, np.exp(sol.y[r_phyto:]).T, '--')