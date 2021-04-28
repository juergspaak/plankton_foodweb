import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import zoop_traits as zt
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
sig_size = np.sqrt(0.5)
# conversion factor from phytoplankton to zooplankton densities
# mum^3 = 1e-18 m^3 =1e-18 (1e3 kg) = 1e-18 (1e3 *1e6 mg) = 1e-9 mg
mum3_mg = np.log(1e-9)

# zooplankton prefer phytoplankton that are about 40**-3 times smaller
# we scale such that mean size zoop prefer mean sized phyto
zoop_pref = zt.mean_zoop[0] - (pt.mean_traits[0] + mum3_mg)
# this corresponds to zoop prefering 20**-3 times smaller
np.exp(zoop_pref)**(1/3)


def generate_plankton(r_phyto, r_zoop, n_coms):
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
    traits_zoop = zt.generate_zooplankton_traits(r_zoop, n_coms)
    traits_phyto.update(traits_zoop)
    traits = traits_phyto
    
    # compute handling time
    # coefficients from Uiterwaal 2020, Ecology 101(4):e02975. 10.1002/ecy.2975
    traits["h_zp"] = (-0.25*np.log(traits["size_Z"][...,np.newaxis])
                            + 0.34*np.log(traits["size_P"][:,np.newaxis]))
    traits["h_zp"] = (-0.7*np.log(traits["size_Z"][...,np.newaxis])
                            + 1.0*np.log(traits["size_P"][:,np.newaxis]))
    # average handling time should be 4 seconds = 4/3600 hours xxx
    traits["h_zp"] = np.exp(np.log(4/3600)
                            + traits["h_zp"] - np.mean(traits["h_zp"]))
    
    # add selectivity
    # differences in traits
    size_diff = ((mum3_mg + np.log(traits["size_P"][:,np.newaxis]))
                 - (np.log(traits["size_Z"])[...,np.newaxis] - zoop_pref))**2
    traits["s_zp"] = np.exp(-size_diff/2*sig_size**2)

    # mortality rate of zooplankton
    traits["m_Z"] = np.full(traits["mu_Z"].shape, 0.001)
    
    # rescaling of zooplankton growth rate to have maximum growth rates
    traits["alpha_Z"] = (np.einsum("nz,nzp,np->nz", traits["c_Z"],
                                   traits["s_zp"], traits["R_P"])/
                    np.einsum("nz,nzp,nzp->nz", traits["c_Z"], traits["h_zp"],
                              traits["s_zp"]))
    return traits

if __name__ == "__main__":
    r_phyto, r_zoo, n_coms = [5,5,400]
    traits = generate_plankton(r_phyto, r_zoo, n_coms)
    
    # expand dimension of size traits
    traits["size_P"] = np.repeat(traits["size_P"][:,np.newaxis], r_zoo, axis = 1)
    traits["size_Z"] = np.repeat(traits["size_Z"][...,np.newaxis], r_phyto,
                                 axis = 1)
    tf = {key: np.log(traits[key].flatten()) for key in traits.keys()}
    
    fig, ax = plt.subplots(2,2, figsize = (9,9))
    
    vmin, vmax = np.percentile(tf["h_zp"], [5,95])
    cmap = ax[0,0].scatter(tf["size_P"], tf["size_Z"], c = tf["h_zp"], s = 1,
                         vmin = vmin, vmax = vmax)
    ax[0,0].set_xlabel("size_P")
    ax[0,0].set_ylabel("size_Z")
    ax[0,0].set_title("h_zp")
    fig.colorbar(cmap, ax = ax[0,0])
    ax[0,1].hist(tf["h_zp"], bins = 30)
    ax[0,1].set_xlabel("h_zp")
    
    
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

    
    
    
    
    
    
    
    