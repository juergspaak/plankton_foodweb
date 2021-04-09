import numpy as np
import data_load as dl

def generate_phytoplankton(r_spec, n_coms):
    """ Generate traits of phytoplankton communities
    
    Parameters:
        r_spec (integer): Species richness per community
        n_coms (integer): Number of communities to generate
    Returns:
        traits (dictionary): traits is a dictionary containing the phytoplankton
            traits. Each entry is an array of shape (r_spec, n_coms)
    """
    
    traits = {}
    for key in dl.gen_data.keys():
        trait = dl.gen_data[key].resample(10*r_spec*n_coms)
        traits[key] = trait[trait>0][:r_spec*n_coms].reshape(r_spec, n_coms)
    traits["m"] = np.random.uniform(0,0.03, size = n_coms)
    
    traits["N_star"] = traits["m"]*traits["k_n"]/(traits["mu_n"]-traits["m"])
    traits["P_star"] = traits["m"]*traits["k_p"]/(traits["mu_p"]-traits["m"])
    
    ###########################################################################
    # to be adapted
    traits["k"] = np.random.uniform(1e-8, 1e-7, traits["mu_l"].shape)
    return traits

env = {"I_in": 100,
       "P": 5,
       "N": 25,
       "m": 0.2,
       "zm": 1}

def light_growth(N, t, env = env):
    # light limited growth at grtowth rate I_in
    tot_abs = np.sum(N * t["a"], axis = -2, keepdims = True)
    return t["mu"]/tot_abs*np.log((t["k_l"] + env["I_in"])/
            (t["k_l"] + env["I_in"]*np.exp(-env["zm"]*tot_abs)))
    
def phosphor_growth(N, t, env = env):
    P = env["P"] - np.sum(N * t["c_p"], axis = -2, keepdims = True)
    P[P<0] = 0
    return t["mu"]*P/(P + t["k_p"])

def nitrogen_growth(N, t, env = env):
    nit = env["N"] - np.sum(N * t["c_n"], axis = -2, keepdims = True)
    nit[nit<0] = 0
    return t["mu"]*nit/(nit + t["k_n"])

limiting_growth = {"N": nitrogen_growth,
                  "P": phosphor_growth,
                  "L": light_growth}
limiting_growth_keys = np.array(["N", "P", "L"])

def phyto_growth(N, t, env = env, limiting_res = limiting_growth_keys):
    
    N = N.reshape(t["mu"].shape)
    growth = np.empty(((len(limiting_res),  ) + N.shape))
    for i, key in enumerate(limiting_res):
        growth[i] = limiting_growth[key](N, t, env)

    growth = np.nanmin(growth, axis = 0)
    
    growth.shape = growth.size   
    return growth - env["m"]

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
    

traits = generate_phytoplankton(5, 1000)
"""
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    
    time = np.linspace(0,1000,1001)
    com_shape = (5, 10)
    traits = generate_phytoplankton(*com_shape)
    
    sol = odeint(lambda N, t: N*phyto_growth(N, traits),
                 np.full(traits["mu_l"].size, 1e7),
                 time)
    sol.shape = (-1, ) + com_shape
    
    rel_abund = sol[-1]/np.sum(sol[-1], axis = 0)
    richness = (rel_abund>1e-2).sum(axis = 0)
    i = np.argmax(richness)
    fig, ax = plt.subplots(2,2, sharex = True, figsize = (7,7))
    ax[0,0].plot(time, sol[...,i])
    phosphor, nitrogen, I_out, limiting_factor, growth, growth_all = evaluate_growth(sol, traits, env)
    ax[0,1].plot(time, phosphor[...,i])
    ax[0,1].set_title("phosphor")
    ax[1,0].plot(time, nitrogen[...,i])
    ax[1,0].set_title("nitrogen")
    ax[1,1].plot(time, I_out[...,i])
    ax[1,1].set_title("I_out")

    print(richness)"""