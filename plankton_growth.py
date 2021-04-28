import numpy as np
import generate_plankton as gp

"""
traits and their meaning

# environmental parameters
P: Resource supply of phosphorus [\mumol P L^-1]
N: Resource supply of nitrogen [\mumol N L^-1]
d: dilution rate (same as mortality rate) [day^-1]
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
xxx needs conversion from resource to zoop? needs to sum phyto P

Joint variables
h_zp: handling time of phytoplankton by zoop, [day/cell]
s_zp: seectivity/preference of eating phytoplankton p by zooplankton z [1]
"""

env = {"I_in": 100,
       "P": 5,
       "N": 25,
       "d": 0.2,
       "zm": 5}

# unit conversions used
uc = {"ml_L": 1000,
      "h_day": 24 # hours in a day
      }

def light_growth(N, t, env = env):
    # light limited growth at grtowth rate I_in
    tot_abs = np.sum(N * t["a"], axis = -1, keepdims = True)
    return t["mu_P"]/tot_abs*np.log((t["k_l"] + env["I_in"])/
            (t["k_l"] + env["I_in"]*np.exp(-env["zm"]*tot_abs)))
    
def phosphor_growth(N, t, env = env):
    P = env["d"]*env["P"] - uc["ml_L"]*np.sum(N * t["c_p"], axis = -1,
                                              keepdims = True)
    P[P<0] = 0
    return t["mu_P"]*P/(P + t["k_p"])

def nitrogen_growth(N, t, env = env):
    N = env["d"]*env["N"] - uc["ml_L"]*np.sum(N * t["c_n"], axis = -1,
                                                keepdims = True)
    N[N<0] = 0
    return t["mu_P"]*N/(N  + t["k_n"])

limiting_growth = {"N": nitrogen_growth,
                  "P": phosphor_growth,
                  "L": light_growth}
limiting_growth_keys = np.array(["N", "P", "L"])

def phyto_growth(N, t, env = env, limiting_res = limiting_growth_keys):
    
    N = N.reshape(t["mu_P"].shape)
    growth = np.empty(((len(limiting_res),  ) + N.shape))
    for i, key in enumerate(limiting_res):
        growth[i] = limiting_growth[key](N, t, env)

    growth = np.nanmin(growth, axis = 0)
    
    growth.shape = growth.size   
    return growth

def grazing(N_phyto,t):
    # how much zooplankton eat of each specific phytoplankton
    numerator = t["c_Z"][:,np.newaxis]*t["s_zp"]*N_phyto
    denom = 1 + t["c_Z"]*np.einsum("zp,zp,p->z",t["h_zp"],t["s_zp"],N_phyto)
    return numerator/denom[...,np.newaxis]

def zoop_growth(grazed, t):
    # growth on substitutional resources
    return t["mu_Z"]/t["alpha_Z"]*np.sum(t["R_P"]*grazed, axis = -1)


def plankton_growth(N, t, env, limiting_res = limiting_growth_keys):
    
    # separate densities into phytoplankton and zooplankton
    N_phyto = N[:len(t["mu_P"])]
    N_zoo = N[len(t["mu_P"]):]
    
    grazed = grazing(N_phyto,t)
    
    dZ_dt = N_zoo*(zoop_growth(grazed, t) - t["m_Z"])
    
    dP_dt = (N_phyto*phyto_growth(N_phyto, t, env, limiting_res)
             - uc["h_day"]/uc["ml_L"]*np.einsum("z,zp->p", N_zoo, grazed)
             - N_phyto*0.001)
             
    return np.append(dP_dt, dZ_dt)

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

r_phyto = 2
r_zoo = 2
traits = gp.generate_plankton(r_phyto, r_zoo,10)
t2 = {}
for key in traits.keys():
    t2[key] = traits[key][0]
    
N = np.array(r_phyto*[1e3] + r_zoo*[1])
#print(plankton_growth(N, t, env))

N_phyto = N[:r_phyto]

grazed = grazing(N_phyto, t2)




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    
    traits = gp.generate_plankton(3,4,100)
    
    time = np.linspace(0,5000)
    sol = odeint(lambda N,t: plankton_growth(N, t2, env), N, time)
    
    fig, ax = plt.subplots(2, sharex = True)
    ax[0].plot(time, sol[:,:r_phyto], '-')
    ax[1].plot(time, sol[:,r_phyto:], '-')
    
    