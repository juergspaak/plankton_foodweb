import numpy as np
import pandas as pd
from scipy.stats import linregress
import phytoplankton_traits as pt

import matplotlib.pyplot as plt

# growth rates
growth = pd.read_csv("growth_rates_brun2017.csv", encoding = 'ISO-8859-1')
growth = growth[["Body mass (mg)", "Growth (15°C)",
                "Specific growth (15°C)"]]
growth.columns = ["pred_mass", "growth", "specific_growth"]

# unit conversion (data is reported as per day, but comparison with
# Kiorboe 2014 shows that this is not the case, but as per hour)
growth.growth *= 24
growth.specific_growth *= 24

# log transform data
growth = np.log(growth)
growth = growth[np.isfinite(growth.growth)]

clear = pd.read_csv("clearance_rates_brun2017.csv", encoding = 'ISO-8859-1')
clear = clear[["Body mass (mg)", "Fmax (15°C)",
               "Specific Fmax (15°C)"]]
clear.columns = ["pred_mass", "fmax", "specific_fmax"]
clear = np.log(clear)

# empirically measured traits
raw_data = {}
raw_data["mu_Z"] = growth.specific_growth.values

raw_data["c_Z"] = clear.fmax.values
raw_data["size_Z"] = np.append(growth.pred_mass, clear.pred_mass)

zoop_traits = ["size_Z", "mu_Z", "c_Z"]
A_zoop = pd.DataFrame(np.zeros((3,3)),
        index = zoop_traits, columns = zoop_traits)
mean_zoop = np.empty(3)

for i,trait in enumerate(zoop_traits):
    mean_zoop[i] = np.nanmean(raw_data[trait])
    A_zoop.loc[trait, trait] = np.nanvar(raw_data[trait])
    
# covariance between size and clearance, linear regression suggests 1
A_zoop.loc["size_Z", "c_Z"] = A_zoop.loc["size_Z", "size_Z"]
A_zoop.loc["c_Z", "size_Z"] = A_zoop.loc["size_Z", "c_Z"]

def generate_zooplankton_traits(r_spec = 1, n_com = 100):
    traits = np.exp(np.random.multivariate_normal(mean_zoop, A_zoop,
                                                  (n_com,r_spec)))
    trait_dict = {}
    for i, trait in enumerate(zoop_traits):
        trait_dict[trait] = traits[...,i]
    
    return trait_dict

# conditional trait distributions, assuming size is known
a = ["mu_Z", "c_Z"]
s = "size_Z"
A_num = A_zoop.values
A_conditional = A_num[1:,1:] - A_num[1:,[0]].dot(1/A_num[0,0]*A_num[[0],1:])

# conversion factor from phytoplankton to zooplankton densities
# mum^3 = 1e-18 m^3 =1e-18 (1e3 kg) = 1e-18 (1e3 *1e6 mg) = 1e-9 mg
mum3_mg = np.log(1e-9)

# zooplankton prefer phytoplankton that are about 40**-3 times smaller
# we scale such that mean size zoop prefer mean sized phyto
zoop_pref = mean_zoop[0] - (pt.mean_traits[0] + mum3_mg)
# this corresponds to zoop prefering 20**-3 times smaller
np.exp(zoop_pref)**(1/3)

# variance of noise term
sig_size_noise = np.sqrt(A_zoop.loc["size_Z", "size_Z"]
                         - pt.cov_matrix.loc["size_P", "size_P"])
def generate_conditional_zooplankton_traits(phyto):
    # generate zooplankton assuming they adapted to phyto_sizes
    size_Z = np.log(phyto["size_P"]) + mum3_mg + zoop_pref
    # add noise
    size_Z = size_Z + np.random.normal(0, sig_size_noise, size_Z.shape)
    size_Z = size_Z.reshape(-1,1)
    
    other_traits = (mean_zoop[1:] +
                    A_zoop.loc[a,s].values/A_zoop.loc[s,s]*(size_Z-mean_zoop[0]))
    other_traits += np.random.multivariate_normal(np.zeros(len(a)),
                                                  A_conditional,
                                                  (size_Z.size))
    trait_dict = {"size_Z": np.exp(size_Z).reshape(phyto["size_P"].shape)}
    for i, trait in enumerate(zoop_traits[1:]):
        trait_dict[trait] = np.exp(other_traits[...,i]).reshape(
                                                        phyto["size_P"].shape)
    
    return trait_dict

if __name__ == "__main__":
    traits = np.random.multivariate_normal(mean_zoop, A_zoop.values, 1000)
    traits_2 = generate_conditional_zooplankton_traits(pt.generate_phytoplankton_traits(1,1000))
    traits_2 = {key: np.log(traits_2[key].flatten()) for key in traits_2.keys()}
    traits_2 = np.array([traits_2["size_Z"], traits_2["mu_Z"], traits_2["c_Z"]]).T
    traits = traits_2
    bins = 15
    
    n = len(zoop_traits)
    fig, ax = plt.subplots(n,n,figsize = (9,9), sharex = "col", sharey = "row")
    
    for i in range(n):
        for j in range(n):
            if i>j:
                ax[i,j].scatter(traits[:,j], traits[:,i], s = 2, alpha = 0.1)
        ax_hist = fig.add_subplot(n,n,1 + (n+1)*i)
        ax_hist.hist(traits[:,i], bins, density = True)
        ax_hist.set_xticklabels([])
        ax_hist.set_yticklabels([])
        ax_hist.hist(raw_data[zoop_traits[i]], bins, density = True,
                    alpha = 0.5)
        ax[i,0].set_ylabel(zoop_traits[i])
        ax[-1,i].set_xlabel(zoop_traits[i])
        
    ax[1,0].scatter(growth.pred_mass, growth.specific_growth, alpha = 0.5)
    ax[2,0].scatter(clear.pred_mass, clear.fmax, alpha = 0.5)
    
    ax[0,0].set_ylim(ax[0,0].get_xlim())
    ax[-1,-1].set_xlim(ax[-1,-1].get_ylim())
    
    fig.savefig("Figure_zooplankton_traits.pdf")



