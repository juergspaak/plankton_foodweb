import numpy as np
import warnings

import zoop_traits as zt
uc = zt.uc
import phytoplankton_traits as pt
import plankton_growth as pg

"""
traits and their meaning

# environmental parameters
P: Resource supply of phosphorus [\mumol P L^-1]
N: Resource supply of nitrogen [\mumol N L^-1]
m: dilution rate (same as mortality rate) [day^-1]
I: iradiance [\mumol quanta m^-2 s^-1]
zm: depth of epilimnion [m]


# Phytoplantkon specific traits
size_P: Biovolume [\mum^3 cell^-1]
mu_P: maximal growth rate of phytoplankton [day ^-1]
k_p: Halfsaturation constant wrt. phosphorus, [\mumol P L^-1]
k_n: Halfsaturation constant wrt. nitrogen, [\mumol N L^-1]
k_l: Halfsaturation constant wrt. Light [\mumol quanta m^-2 s^-1]
c_p: maximum uptake rate of phosphorus [\mumol P cell^-1 day^-1]
c_n: maximum uptake rate, [\mumol N cell^-1 day^-1]
a: absorption coefficient of light, [mm^2 cell^{-1}]
m: mortality rate/dilution rate [day^-1]
N_P: phytoplankton density [cell \ml ^-1]
R_P: nutrient contents [mumol R cell^-1],

Zooplankton traits
size_Z: Zooplankton size [mg ind^-1]
c_Z: clearance rate [ml h^-1 ind^-1]
N_Z: zooplankton density [ind L^-1]
mu_Z: maximum specific growth rate of Zooplankton [day^-1]
m_Z: mortality rate of zooplankton
k_Z: halfsaturation constant [mumol R ind h^-1]

Joint variables
h_zp: handling time of phytoplankton by zoop, [h cell^-1 mg^-1]
s_zp: seectivity/preference of eating phytoplankton p by zooplankton z [1]
"""

# data taken from branco et al 2020, DOI: 10.1086/706251
# divided by three because they report 
sig_size = np.sqrt(0.25)

env = {"I_in": 100,
       "P": 50,
       "N": 250,
       "d": 0.1,
       "zm": 10}

def generate_env(n_coms, I_in = [50,200], P = [5,20], N = [10,100],
                 d = [0.05,0.2], zm = [10,100]):
    env = {"I_in": np.random.uniform(*I_in, (n_coms,1)),
                   "P": np.random.uniform(*P, (n_coms, 1)),
                   "N": np.random.uniform(*N, (n_coms,1)),
                   "d": np.random.uniform(*d, (n_coms,1)),
                   "zm": np.random.uniform(*zm, (n_coms,1))}
    return env

def generate_base_traits(r_spec = 1, n_com = 100, std = None, diff_std = {},
                                  corr = None, phyto = True, size = None,
                                  diff_mean = {}, tradeoffs = {}):
    # generate random size if size is not given
    if size is None:
        if phyto:
            size = np.random.normal(pt.mean_phyto["size_P"],
                                    pt.std_phyto["size_P"], (n_com,r_spec,1))
        else:
            size = np.random.normal(zt.mean_zoop["size_Z"],
                                    zt.std_zoop["size_Z"], (n_com,r_spec,1))
    else:
        size.shape = size.shape + (1,)
    
    # order species by size within each community
    size = np.sort(size, axis = 1)
    
    # set variation or tradeoffs to wanted value
    if std is None:
        std = pt.std_phyto.copy() if phyto else zt.std_zoop.copy()
    if corr is None:
        corr = pt.corr_phyto.copy() if phyto else zt.corr_zoop.copy()
    # change variation of certain traits
    for key in diff_std.keys():
        if key in std.columns:
            std[key] *= diff_std[key]
            
    
    
    # change tradeoffs between traits
    for key in tradeoffs.keys():
        key1, key2 = key.split(":")
        if key1 in corr.columns and key2 in corr.columns:
            corr.loc[key1, key2] = tradeoffs[key]
            corr.loc[key2, key1] = tradeoffs[key]


    # mean trait values per community
    mean = pt.mean_phyto.copy() if phyto else zt.mean_zoop.copy()
    # change mean of certain traits
    for key in diff_mean.keys():
        if key in mean.columns:
            mean[key] += diff_mean[key]
    mean = mean.values
   
    # combine corrlation and standard deviation into covariance matrix
    try:
        cov = (corr*std.values*std.values[0,:,np.newaxis]).values
    except AttributeError: # corr is a matrix and not a dataframe
        cov = corr*std.values*std.values[0,:,np.newaxis]
    
    # generate conditional distribution given size
    mu_cond = mean[:,1:] + cov[1:,0]/cov[0,0]*(size - mean[0,0])
    cov_cond = cov[1:,1:] - cov[1:,[0]].dot(1/cov[0,0]*cov[[0],1:])
    
    traits = mu_cond + np.random.multivariate_normal(np.zeros(len(cov[1:])),
                                                     cov_cond, size.shape[:-1])
    traits = np.exp(traits)
    trait_dict = {"size_P" if phyto else "size_Z": np.exp(size[...,0])}
    trait_names = pt.phyto_traits if phyto else zt.zoop_traits
    for i, trait in enumerate(trait_names[1:]):
        trait_dict[trait] = traits[...,i]
    
    return trait_dict

def generate_plankton(r_phyto, n_coms, r_zoop = None, evolved_zoop = True,
                      size_P = None, diff_mean = {}, tradeoffs = {},
                      size_Z = None, diff_std = {}, corr_phyto = None, corr_zoo = None):
    """ Generate traits of plankton communities
    
    Parameters:
        r_phyto (integer): Phytoplankton initial species richness
        r_zoop (integer): Zooplankton initial species richness
        n_coms (integer): Number of communities to generate
        
    Returns:
        traits (dictionary): traits is a dictionary containing the all the
        community traits
    """
    
    traits_phyto = generate_base_traits(r_phyto, n_coms, size = size_P,
                        tradeoffs = tradeoffs, diff_mean = diff_mean,
                        diff_std = diff_std, corr = corr_phyto)

    if evolved_zoop:
        size_Z = np.log(traits_phyto["size_P"]*uc["mum3_mg"]*np.exp(zt.zoop_pref))
        # add noise
        size_Z = size_Z + np.random.normal(0, zt.sig_size_noise, size_Z.shape)

        traits_zoop = generate_base_traits(r_phyto, n_coms, phyto = False,
                                           size = size_Z, diff_std = diff_std,
                                           tradeoffs = tradeoffs, corr = corr_zoo,
                                           diff_mean = diff_mean)
        r_zoop = r_phyto
    else: 
        if r_zoop is None:
            r_zoop = r_phyto
        traits_zoop = generate_base_traits(r_zoop, n_coms, phyto = False,
                                        size = size_Z, corr = corr_zoo,
                                        tradeoffs = tradeoffs,
                                        diff_mean = diff_mean)    
    
    traits_phyto.update(traits_zoop)
    traits = traits_phyto
    traits["r_phyto"] = r_phyto
    traits["r_zoo"] = r_zoop
    traits["n_coms"] = n_coms
    
    # adjust size if necessary
    if "size_P" in diff_std.keys():
        temp = np.log(traits["size_P"]) - pt.mean_phyto["size_P"].values
        temp *= np.sqrt(diff_std["size_P"])
        traits["size_P"] = np.exp(temp + pt.mean_phyto["size_P"].values)
    if "size_Z" in diff_std.keys():
        temp = np.log(traits["size_Z"]) - zt.mean_zoop["size_Z"].values
        temp *= np.sqrt(diff_std["size_Z"])
        traits["size_Z"] = np.exp(temp + zt.mean_zoop["size_Z"].values)
    
    # compute handling time
    # coefficients from Uiterwaal 2020, Ecology 101(4):e02975. 10.1002/ecy.2975
    #traits["h_zp"] = (np.log(0.005/24) -0.25*np.log(traits["size_Z"][...,np.newaxis])
    #                        + 0.34*np.log(traits["size_P"][:,np.newaxis]))
    
    # data from branco 2020, scaling volume-> length from Uye 1989
    traits["h_zp"] = np.exp(np.log(0.001/24)
                -2.11*(0.2878*np.log(traits["size_Z"][...,np.newaxis])+3.75)
                +1*(np.log(traits["size_P"][:,np.newaxis]) -np.log(np.pi/6)))
    h_zp_mean = (np.log(0.001/24)
                 -2.11*(0.2878*zt.mean_zoop["size_Z"]+3.75)
                 +1*(pt.mean_phyto["size_P"]-np.log(np.pi/6)))
    traits["h_zp"] = (-2.11*0.2878*(np.log(traits["size_Z"][...,np.newaxis])-zt.mean_zoop["size_Z"].values)
                +1*(np.log(traits["size_P"][:,np.newaxis])-pt.mean_phyto["size_P"].values))
    if "h_zp" in diff_std.keys():
        traits["h_zp"] *= diff_std["h_zp"]
    if "h_zp" in diff_mean.keys():
        h_zp_mean += diff_mean["h_zp"]
    traits["h_zp"] = np.exp(h_zp_mean.values + traits["h_zp"])
    
    # add selectivity
    # differences in traits
    size_diff = ((np.log(uc["mum3_mg"]*traits["size_P"][:,np.newaxis]))
                 - (np.log(traits["size_Z"])[...,np.newaxis]- zt.zoop_pref))**2
    if "s_zp" in diff_std.keys():
        size_diff *= diff_std["s_zp"]
    traits["s_zp_raw"] = np.exp(-size_diff/(2*sig_size**2))
    with warnings.catch_warnings(record = True):
        traits["s_zp"] = traits["s_zp_raw"]/np.sum(traits["s_zp_raw"],
                                                   axis = -1, keepdims=True)
        
        traits["s_zp"][np.isnan(traits["s_zp"])] = 0   

    return traits

def community_equilibrium(tr, env = env):
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


def phytoplankton_equilibrium(tr, env = env):

    try:
        if env["zm"].ndim == 1:
            env = {key: np.expand_dims(env[key],-1) for key in env.keys()}
    except AttributeError:
        pass
    
    # R_star values for resources
    tr["R_star_n"] = env["d"]*tr["k_n"]/(tr["mu_P"]-env["d"])
    tr["R_star_p"] = env["d"]*tr["k_p"]/(tr["mu_P"]-env["d"])
    
    # equilibrium density based on resource competition
    # if limited by phosphorus
    tr["N_star_P_p"] = (env["d"]*env["P"]-tr["R_star_p"])/(tr["c_p"]*uc["ml_L"])
    # if limited by nitrogen
    tr["N_star_P_n"] = (env["d"]*env["N"]-tr["R_star_n"])/(tr["c_n"]*uc["ml_L"])
    
    # if limited by light
    # initial guess of outcoming light, all light is absorbed
    tr["I_out"] = 0
    tr["N_star_P_l"] = np.empty(1)

    # find N_star iteratively, converges fast for almost all communities
    for i in range(20):
        N_star = tr["N_star_P_l"].copy()
        tr["N_star_P_l"] = (tr["mu_P"]/(env["d"]*tr["a"]*env["zm"])
                          *np.log((tr["k_l"]+env["I_in"])/(tr["k_l"]+ tr["I_out"])))
        tr["I_out"] = env["I_in"]*np.exp(-env["zm"]*tr["a"]*tr["N_star_P_l"])
    
    tr["N_star_P_l"][N_star<1] = np.nan
    
    
    tr["N_star_P_res"] = np.amin([tr["N_star_P_p"],
                              tr["N_star_P_n"],
                              tr["N_star_P_l"]], axis = 0)
    
    tr["limit_res_res"] = np.argmin([tr["N_star_P_p"],
                              tr["N_star_P_n"],
                              tr["N_star_P_l"]], axis = 0)
    
    for res in ["n", "p", "res", "l"]:
        tr["N_star_P_" + res][tr["N_star_P_" + res] <0] = np.nan
   
    return tr
    

def select_keys(traits):
    sel_keys = list(traits.keys())
    try:
        sel_keys.remove("n_coms")
    except ValueError:
        pass
    try:
        sel_keys.remove("r_phyto")
    except ValueError:
        pass
    try:
        sel_keys.remove("r_zoo")
    except ValueError:
        pass
    return sel_keys

def generate_communities(r_phyto, n_coms, evolved_zoop = True, r_zoo = None,
                         monoculture_equi = True, size_P = None, diff_mean = {},
                         tradeoffs = {}, size_Z = None, diff_std = {}, corr_phyto = None,
                         corr_zoo = None):
    traits = generate_plankton(r_phyto, n_coms, r_zoo, evolved_zoop=evolved_zoop,
                               size_P = size_P, diff_mean = diff_mean, tradeoffs = tradeoffs,
                               size_Z = size_Z, diff_std = diff_std, corr_phyto = corr_phyto,
                               corr_zoo = corr_zoo)
    
    env = generate_env(n_coms)
    traits = community_equilibrium(traits, env)
    
    ind1 = np.isfinite(traits["N_star_P"]).all(axis = -1)
    ind2 = np.isfinite(traits["N_star_Z"]).all(axis = -1)
    ind = ind1 & ind2
    
    traits = {key: traits[key][ind] for key in select_keys(traits)}
    traits["r_phyto"] = r_phyto
    traits["r_zoo"] = traits["size_Z"].shape[-1]
    traits["n_coms"] = len(traits["size_P"])
    env = {key: env[key][ind] for key in env.keys()}
    n_coms = len(traits["mu_P"])
    
    if monoculture_equi:
        traits = phytoplankton_equilibrium(traits, env)
    
    return traits, env

def select_i(traits, env, i = None):
    if i is None:
        i = np.random.randint(traits["n_coms"])
    tr_i = {key: traits[key][i] for key in select_keys(traits)}
    tr_i["r_phyto"] = traits["r_phyto"]
    tr_i["r_zoo"] = traits["r_zoo"]
    env_i = {key: env[key][i] for key in env.keys()}
    return tr_i, env_i,i

if __name__ == "__main__" and False:
    import matplotlib.pyplot as plt
    
    # generate phytoplankton communities
    r_phyto, r_zoo, n_coms = [2,2, int(1e4)]
    traits = generate_plankton(r_phyto, n_coms, r_zoo, evolved_zoop=False)
    env = generate_env(n_coms)
    traits = phytoplankton_equilibrium(traits, env)
    
    env = generate_env(n_coms)
    
    # compute equilibria
    traits = community_equilibrium(traits, env)
    traits = phytoplankton_equilibrium(traits, env)
    
    # how many do coexist?
    ind1 = np.isfinite(traits["N_star_P"]).all(axis = -1)
    ind2 = np.isfinite(traits["N_star_Z"]).all(axis = -1)
    ind = ind1 & ind2
    print(np.sum(ind1), np.sum(ind2), np.sum(ind1&ind2))
    
    bins = np.linspace(0,20, 100)
    fig, ax = plt.subplots(2,1, figsize = (7,5))
    for i in ["P_n", "P_p", "P_l", "P_res", "P"]:
        ax[0].hist(np.log(traits["N_star_"+i].flatten()),
                   alpha = 0.5, bins = bins, label = i
                 ,density = True, histtype="step")
    ax[0].legend()
    ax[0].set_xlabel("Phytoplankton densities (\mum^3 ml^-1")
    
    ax[1].hist(np.log(traits["N_star_Z"]).flatten(), density = True,
               bins = bins)
    
    fig.tight_layout()
    fig.savefig("Figure_equilibrium_densities_phytoplankton.pdf")
    

    # expand dimension of size traits
    traits["size_P"] = np.repeat(traits["size_P"][:,np.newaxis], r_zoo, axis = 1)
    traits["size_Z"] = np.repeat(traits["size_Z"][...,np.newaxis], r_phyto,
                                 axis = 1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tf = {key: np.log(traits[key].flatten())
              for key in select_keys(traits)}
    
    
    fig, ax = plt.subplots(2,2, figsize = (9,9))
    
    # handling time graph
    vmin, vmax = np.percentile(tf["h_zp"], [5,95])
    cmap = ax[0,0].scatter(tf["size_P"], tf["size_Z"], c = tf["h_zp"], s = 1,
                         vmin = vmin, vmax = vmax)
    ax[0,0].set_xlabel("size_P")
    ax[0,0].set_ylabel("size_Z")
    ax[0,0].set_title("h_zp")
    fig.colorbar(cmap, ax = ax[0,0])
    ax[0,1].hist(tf["h_zp"], bins = 30)
    ax[0,1].set_xlabel("h_zp")
    
    # selectivity graph
    vmin, vmax = [0,1]
    cmap = ax[1,0].scatter(tf["size_P"], tf["size_Z"], c = np.exp(tf["s_zp"]), s = 1,
                         vmin = vmin, vmax = vmax)
    fig.colorbar(cmap, ax = ax[1,0])
    ax[1,0].set_xlabel("size_P")
    ax[1,0].set_ylabel("size_Z")
    ax[1,0].set_title("s_zp")
    ax[1,1].hist(np.exp(tf["s_zp"]), bins = 30)
    ax[1,1].set_xlabel("s_zp")
    
    fig.tight_layout()  
    fig.savefig("Figure_joint_traits.pdf")
    
    