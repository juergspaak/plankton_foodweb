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

h = 1 # hill exponent

def light_growth(N, t, env = gp.env):
    # light limited growth at grtowth rate I_in
    tot_abs = np.sum(env["zm"]*N*t["a"], axis = -1, keepdims = True)
    # returns nan if tot_abs is zero, but this is handled correctly in next step
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
    # light growth might be nan if phyto densities are zero
    return np.nanmin(growth, axis = 0)

def grazing(N_phyto,t, pr = False, h = h):
    # how much zooplankton eat of each specific phytoplankton
    numerator = np.expand_dims(t["c_Z"],-1)*t["s_zp"]*N_phyto**h
    
    denom = 1 + t["c_Z"]*np.einsum("zp,zp,...p->...z",t["h_zp"],t["s_zp"],
                                   N_phyto**h)
    return numerator/np.expand_dims(denom,-1)

def plankton_growth(N, t, env, limiting_res = limiting_growth_keys):
    N[N<1e-3] = 1e-3
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

def convert_ode_to_log(logN, t, env):
    N = np.exp(logN)
    return plankton_growth(N, t, env)/N