import numpy as np
import generate_plankton as gp

"""
traits and their meaning

# environmental parameters
P: Resource supply of phosphorus [\mumol P L^-1]
N: Resource supply of nitrogen [\mumol N L^-1]
dil: dilution rate (same as mortality rate) [day^-1]
I: iradiance [\mumol quanta m^-2 s^-1]
zm: depth of epilimnion [m]


# Phytoplantkon specific traits
size_P: Biovolume [\mum^3]
mu_P: maximal growth rate of phytoplankton [day ^-1]
k_p: Halfsaturation constant wrt. phosphorus, [\mumol P L^-1]
k_n: Halfsaturation constant wrt. nitrogen, [\mumol N L^-1]
k_l: Halfsaturation constant wrt. Light [\mumol quanta m^-2 s^-1]
c_p: maximum uptake rate of phosphorus [µmol P cell^-1 day^-1]
c_n: maximum uptake rate, [µmol N cell^-1 day^-1]
a: absorption coefficient of light, [xxx]
m: mortality rate/dilution rate [day^-1]
N_P: phytoplankton density [cell \mul ^-1]
R_P: nutrient contents [mumol R cell^-1],

Zooplankton traits
size_Z: Zooplankton size [mg]
c_Z: clearance rate [ml/h]
N_Z: zooplankton density [mg/xxx]
mu_Z: maximum growth rate of Zooplankton [mg day^-1]

Joint variables
h_zp: handling time of phytoplankton by zoop, [day/cell]
s_zp: seectivity/preference of eating phytoplankton p by zooplankton z [1]
"""

def light_growth(N, t, env = gp.env):
    # light limited growth at grtowth rate I_in
    tot_abs = np.sum(env["zm"]*N*t["a"], axis = -1, keepdims = True)
    if tot_abs == 0:
        print(N)
    return t["mu_P"]/tot_abs*np.log((t["k_l"] + env["I_in"])/
            (t["k_l"] + env["I_in"]*np.exp(-tot_abs)))
    
def phosphor_growth(N, t, env = gp.env):
    P = env["d"]*env["P"] - gp.uc["ml_L"]*np.sum(N * t["c_p"], axis = -1,
                                              keepdims = True)
    P[P<0] = 0
    return t["mu_P"]*P/(P + t["k_p"])

def nitrogen_growth(N, t, env = gp.env):
    N = env["d"]*env["N"] - gp.uc["ml_L"]*np.sum(N * t["c_n"], axis = -1,
                                                keepdims = True)
    N[N<0] = 0
    return t["mu_P"]*N/(N  + t["k_n"])

limiting_growth = {"N": nitrogen_growth,
                  "P": phosphor_growth,
                  "L": light_growth}
limiting_growth_keys = np.array(["N", "P", "L"])

def phyto_growth(N, t, env = gp.env, limiting_res = limiting_growth_keys):
    
    growth = np.empty(((len(limiting_res),  ) + N.shape))
    for i, key in enumerate(limiting_res):
        growth[i] = limiting_growth[key](N, t, env)
    return np.nanmin(growth, axis = 0)

def grazing(N_phyto,t, pr = False):
    # how much zooplankton eat of each specific phytoplankton
    numerator = np.expand_dims(t["c_Z"],-1)*t["s_zp"]*N_phyto
    
    denom = 1 + t["c_Z"]*np.einsum("zp,zp,...p->...z",t["h_zp"],t["s_zp"],
                                   N_phyto)
    return numerator/np.expand_dims(denom,-1)

def plankton_growth(N, t, env, limiting_res = limiting_growth_keys):
    N[N<0] = 0
    # separate densities into phytoplankton and zooplankton
    N_phyto = N[...,:len(t["mu_P"])]
    N_zoo = N[...,len(t["mu_P"]):]
    grazed = grazing(N_phyto,t)
    
    dZ_dt = N_zoo*(t["mu_Z"]/t["alpha_Z"]*np.sum(t["R_P"]*grazed, axis = -1)
                   - t["m_Z"])
    dP_dt = (N_phyto*phyto_growth(N_phyto, t, env, limiting_res)
             - gp.uc["h_day"]/gp.uc["ml_L"]*np.einsum("...z,...zp->...p", N_zoo, grazed)
             - N_phyto*env["d"])
            
    
    return np.append(dP_dt, dZ_dt, axis = -1)

def evaluate_growth(N, t, env):
    phosphor = env["P"] - np.sum(N*t["c_p"], axis = -2)
    phosphor[phosphor<0] = 0
    
    nitrogen = env["N"] - np.sum(N*t["c_n"], axis = -2)
    nitrogen[nitrogen<0] = 0
    
    I_out = env["I_in"]*np.exp(-env["zm"]*np.sum(N*t["k"], axis = -2))
    
    growth_all = np.empty(((len(limiting_growth_keys),  ) + N.shape))
    for i, key in enumerate(limiting_growth_keys):
        growth_all[i] = limiting_growth[key](N, t, env)
    growth = np.nanmin(growth_all, axis = 0)
    limiting_factor = limiting_growth_keys[np.argmin(growth, axis = 0)]
    
    return phosphor, nitrogen, I_out, limiting_factor, growth, growth_all

if __name__ == "__main__":    
    r_phyto = 1
    r_zoo = 1
    traits = gp.generate_plankton(r_phyto, 1000, r_zoo, evolved_zoop=True)
    traits = gp.community_equilibrium(traits)
    ind1 = np.isfinite(traits["N_star_P"]).all(axis = -1)
    ind2 = np.isfinite(traits["N_star_Z"]).all(axis = -1)
    ind = (np.isfinite(traits["N_star_P"]).all(axis = -1)
           & np.isfinite(traits["N_star_Z"]).all(axis = -1))
    print(sum(ind))
    traits = {key:traits[key][ind] for key in traits.keys()}
    t2 = {}
    i = np.random.randint(len(traits["mu_P"]))
    for key in traits.keys():
        t2[key] = traits[key][i]
    """if r_phyto == 1 == r_zoo:
        print(i)
        # on-stable equilibrium :(
        N = np.array([t2["N_star_P_z"][0,0], t2["N_star_Z"][0,0]])
        print(plankton_growth(N, t2, gp.env))
        
        
        N *= 1 + np.random.uniform(-1e-4, 1e-4, 2)
    else:
        N = np.array(r_phyto*[500] + r_zoo*[10])"""
    
    N = np.append(t2["N_star_P"], t2["N_star_Z"])
    print("N_star", N)
    print("growth",plankton_growth(N, t2, gp.env))
    N = N*np.random.uniform(1-1e-3,1+1e-3,N.shape)
    print(np.round(t2["s_zp"],3))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    
    time = np.linspace(0,500, 100)
    env = {"I_in": 100,
       "P": 500,
       "N": 2500,
       "d": 0.05,
       "zm": 10}
    sol = odeint(lambda N,t: plankton_growth(N, t2, gp.env), N, time)
    
    fig, ax = plt.subplots(2, sharex = True)
    ax[0].plot(time, sol[:,:r_phyto], '-')
    ax[1].plot(time, sol[:,r_phyto:], '-')
    ax[0].semilogy()
    ax[0].set_ylim([1e-6,None])
    ax[1].semilogy()
    ax[1].set_ylim([1e-3,None])
    
    
    