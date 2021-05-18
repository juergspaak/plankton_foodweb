import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt

import zoop_traits as zt
uc = zt.uc
import phytoplankton_traits as pt

"""
traits and their meaning

# environmental parameters
P: Resource supply of phosphorus [\mumol P L^-1]
N: Resource supply of nitrogen [\mumol N L^-1]
m: dilution rate (same as mortality rate) [day^-1]
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
c_Z: clearance rate [ml h^-1 mg^-1]
N_Z: zooplankton density [mg L^-1]
mu_Z: maximum specific growth rate of Zooplankton [day^-1]
m_Z: mortality rate of zooplankton

Joint variables
h_zp: handling time of phytoplankton by zoop, [s/cell]
s_zp: seectivity/preference of eating phytoplankton p by zooplankton z [1]
"""

# data taken from branco et al 2020, DOI: 10.1086/706251
sig_size = np.sqrt(1.5)

# mortality rate of zooplankton
m_Z = 1/15 # assuming a life span of 30 days

env = {"I_in": 100,
       "P": 50,
       "N": 250,
       "d": 0.1,
       "zm": 10}

def generate_env(n_coms, I_in = [50,200], P= [5,20], N = [50,100],
                 d = [0.01,0.2], zm = [1,100]):
    env = {"I_in": np.random.uniform(*I_in, (n_coms,1)),
                   "P": np.random.uniform(*P, (n_coms, 1)),
                   "N": np.random.uniform(*N, (n_coms,1)),
                   "d": np.random.uniform(*d, (n_coms,1)),
                   "zm": np.random.uniform(*zm, (n_coms,1))}
    return env

def generate_plankton(r_phyto, n_coms, r_zoop = None, evolved_zoop = True):
    """ Generate traits of plankton communities
    
    Parameters:
        r_phyto (integer): Phytoplankton initial species richness
        r_zoop (integer): Zooplankton initial species richness
        n_coms (integer): Number of communities to generate
        
    Returns:
        traits (dictionary): traits is a dictionary containing the all the
        community traits
    """
    
    traits_phyto = pt.generate_phytoplankton_traits(r_phyto, n_coms)

    if evolved_zoop:
        traits_zoop = zt.generate_conditional_zooplankton_traits(traits_phyto)
        r_zoop = r_phyto
    else: 
        if r_zoop is None:
            r_zoop = r_phyto
        traits_zoop = zt.generate_zooplankton_traits(r_zoop, n_coms)    
    
    traits_phyto.update(traits_zoop)
    traits = traits_phyto
    
    # compute handling time
    # coefficients from Uiterwaal 2020, Ecology 101(4):e02975. 10.1002/ecy.2975
    traits["h_zp"] = (-0.25*np.log(traits["size_Z"][...,np.newaxis])
                            + 0.34*np.log(traits["size_P"][:,np.newaxis]))
    # coefficients from Branco 2020, DOI: 10.1086/706251
    traits["h_zp"] = (-0.7*np.log(traits["size_Z"][...,np.newaxis])
                            + 1.0*np.log(traits["size_P"][:,np.newaxis]))
    # average handling time should be 4 seconds = 4/3600 hours xxx
    traits["h_zp"] = np.exp(np.log(4/3600)
                            + traits["h_zp"] - np.mean(traits["h_zp"]))
    
    # add selectivity
    # differences in traits
    size_diff = ((np.log(uc["mum3_mg"]*traits["size_P"][:,np.newaxis]))
                 - (np.log(traits["size_Z"])[...,np.newaxis]- zt.zoop_pref))**2
    traits["s_zp_raw"] = np.exp(-size_diff/2*sig_size**2)
    traits["s_zp"] = traits["s_zp_raw"]/np.sum(traits["s_zp_raw"], axis = -1,
                                               keepdims=True)

    # mortality rate of zooplankton
    traits["m_Z"] = np.full(traits["mu_Z"].shape, m_Z)
    
    # rescaling of zooplankton growth rate to have maximum growth rates
    traits["alpha_Z"] = recompute_alpha(traits)

    return traits

def recompute_alpha(traits):
    return (np.einsum("nz,nzp,np->nz", traits["c_Z"],
                                   traits["s_zp"], traits["R_P"])/
                    np.einsum("nz,nzp,nzp->nz", traits["c_Z"], traits["h_zp"],
                              traits["s_zp"]))

def community_equilibrium(tr, env = env):
    try:
        if env["zm"].ndim == 1:
            env = {key: np.expand_dims(env[key],-1) for key in env.keys()}
    except AttributeError:
        pass
    
    c_Z = np.expand_dims(tr["c_Z"],-1)
    tr["A_zoop"] = c_Z*tr["s_zp"]*(-tr["h_zp"]*tr["m_Z"][...,np.newaxis] +
                (tr["mu_Z"]/tr["alpha_Z"])[...,np.newaxis]*tr["R_P"][:,np.newaxis])
    
    # equilibrium density of phytoplankton species
    tr["N_star_P"] = np.linalg.solve(tr["A_zoop"], tr["m_Z"])
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
    tr["limit_res"] = np.argmin([tr["growth_p"],
                                         tr["growth_n"],
                                         tr["growth_l"]], axis = 0)
    

    numerator = uc["h_day"]/uc["ml_L"]*c_Z*tr["s_zp"]
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
   
    return tr

def N_star_Z_mono(tr, env = env):    

    # zooplankton secondary traits competing for only one phytoplankton
    # halfsaturation constant of zooplankton competing for one phytoplankton
    tr["k_zp"] = 1/(tr["c_Z"][...,np.newaxis]*tr["h_zp"]*tr["s_zp"])
    
    # compute equilibrium densities of phytoplankton regulated by zooplankton
    tr["N_star_P_z_mono"] = (tr["k_zp"]*(tr["m_Z"]/
                                   (tr["mu_Z"]-tr["m_Z"]))[...,np.newaxis])
    tr["N_star_P_z_mono"][tr["N_star_P_z_mono"]<0] = np.nan
    
    # compute zooplankton equilibrium densities
    
    try:
        if env["zm"].ndim == 1:
            env = {key: np.expand_dims(env[key],-1) for key in env.keys()}
    except AttributeError:
        pass
    
    # nitrogen concentration
    tr["N_conc_n"] = (env["d"]*env["N"]
                    - uc["ml_L"]*tr["N_star_P_z_mono"]*tr["c_n"][:,np.newaxis])
    tr["N_conc_n"][tr["N_conc_n"]<0] = 0
    # compute resource concentrations for phytoplankton    
    tr["N_conc_p"] = (env["d"]*env["P"]
                    - uc["ml_L"]*tr["N_star_P_z_mono"]*tr["c_p"][:,np.newaxis])
    tr["N_conc_p"][tr["N_conc_p"]<0] = 0
    
    # outcoming light concentration
    tr["I_out"] = env["I_in"]*np.exp(
            -env["zm"]*tr["a"][:,np.newaxis]*tr["N_star_P_z_mono"]) 
    
    # compute relative growth rates of phytoplankton for each resource
    tr["growth_p"] = tr["N_conc_p"]/(tr["k_p"][:,np.newaxis] + tr["N_conc_p"])
    tr["growth_n"] = tr["N_conc_n"]/(tr["k_n"][:,np.newaxis] + tr["N_conc_n"])
    tr["growth_l"] = (1/(env["zm"]*tr["N_star_P_z_mono"]*tr["a"][:,np.newaxis])
                    *np.log((tr["k_l"][:,np.newaxis] + env["I_in"])/
                            (tr["k_l"][:,np.newaxis] + tr["I_out"])))
    
    # compute growth rate of phytoplankton
    tr["growth_P"] = tr["mu_P"][:,np.newaxis]*np.amin([tr["growth_p"],
                                                       tr["growth_n"],
                              tr["growth_l"]], axis = 0)
    
    numerator = tr["c_Z"][...,np.newaxis]*tr["s_zp"]
    denom = 1 + tr["c_Z"][...,np.newaxis]*tr["h_zp"]*tr["s_zp"]*tr["N_star_P_z_mono"]
    tr["grazing_star"] = numerator/denom
    
    # compute zooplankton equilibrium density
    tr["N_star_Z_mono"] = ((tr["growth_P"]-env["d"])
                      /(uc["h_day"]/uc["ml_L"]*tr["grazing_star"]))
    
    return tr
    
if __name__ == "__main__":
    r_phyto, r_zoo, n_coms = [5,5, 400]
    traits = generate_plankton(r_phyto, n_coms, r_zoo, evolved_zoop=False)
    traits = N_star_Z_mono(traits)
    traits = phytoplankton_equilibrium(traits)
    
    traits = community_equilibrium(traits)    

if __name__ == "__main__":

    bins = np.linspace(0,20, 100)
    fig, ax = plt.subplots(2,1, figsize = (7,5))
    for i in ["P_n", "P_p", "P_l", "P_z_mono"]:
        ax[0].hist(np.log(traits["N_star_"+i].flatten()),
                   alpha = 0.5, bins = bins, label = i
                 ,density = True, histtype="step")
    ax[0].legend()
    ax[0].set_xlabel("Phytoplankton densities (\mum^3 ml^-1")
    
    ax[1].hist(np.log(traits["N_star_Z_mono"]).flatten(), density = True, bins = 30)
    ax[1].set_xlabel("Zooplankton density (mg L^-1 = 1e6 \mum^3 ml^-1)")
    fig.tight_layout()
    fig.savefig("Figure_equilibrium_densities_phytoplankton.pdf")
    
    plt.figure()

    # expand dimension of size traits
    traits["size_P"] = np.repeat(traits["size_P"][:,np.newaxis], r_zoo, axis = 1)
    traits["size_Z"] = np.repeat(traits["size_Z"][...,np.newaxis], r_phyto,
                                 axis = 1)
    tf = {key: np.log(traits[key].flatten()) for key in traits.keys()}
    
    plt.figure()
    plt.scatter(tf["size_P"], np.exp(tf["s_zp"]), s = 1)
    
    fig, ax = plt.subplots(3,2, figsize = (9,9))
    
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
    
    # halfsaturation constant
    vmin, vmax = np.percentile(tf["k_zp"], [5,95])
    cmap = ax[2,0].scatter(tf["size_P"], tf["size_Z"], c = tf["k_zp"], s = 1,
                         vmin = vmin, vmax = vmax)
    fig.colorbar(cmap, ax = ax[2,0])
    ax[2,0].set_xlabel("size_P")
    ax[2,0].set_ylabel("size_Z")
    ax[2,0].set_title("k_zp")
    ax[2,1].hist(tf["h_zp"], bins = 30)
    ax[2,1].set_xlabel("h_zp")
    
    
    
    
    fig.tight_layout()
    
    
    
    fig.savefig("Figure_joint_traits.pdf")
    
    
    
    
    
    
    
    
    
    