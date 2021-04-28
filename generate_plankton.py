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
c_Z: clearance rate [m^2/day]
N_Z: zooplankton density [mg/xxx]
mu_Z: maximum specific growth rate of Zooplankton [day^-1]
m_Z: mortality rate of zooplankton
xxx needs conversion from resource to zoop? needs to sum phyto P

Joint variables
h_zp: handling time of phytoplankton by zoop, [s/cell]
s_zp: seectivity/preference of eating phytoplankton p by zooplankton z [1]
"""

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
    # average handling time should be 4 seconds = 4/3600 hours xxx
    traits["h_zp"] = np.exp(np.log(4/3600)
                            + traits["h_zp"] - np.mean(traits["h_zp"]))
    
    # add selectivity
    traits["s_zp"] = np.ones((n_coms, r_zoop, r_phyto))

    # mortality rate of zooplankton
    traits["m_Z"] = np.full(traits["mu_Z"].shape, 0.001) 
    
    # rescaling of zooplankton growth rate to have maximum growth rates
    traits["alpha_Z"] = (np.einsum("nz,nzp,np->nz", traits["c_Z"],
                                   traits["s_zp"], traits["R_P"])/
                    np.einsum("nz,nzp,nzp->nz", traits["c_Z"], traits["h_zp"],
                              traits["s_zp"]))
    return traits

if __name__ == "__main__":
    traits = generate_plankton(3,4,10)
    for key in traits:
        print(key, traits[key].shape)
    
    
    
    
    
    
    
    