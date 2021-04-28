import numpy as np
import pandas as pd
from scipy.stats import linregress

import matplotlib.pyplot as plt

# growth rates
growth = pd.read_csv("growth_rates_brun2017.csv", encoding = 'ISO-8859-1')
growth = growth[["Body mass (mg)", "Growth (15°C)",
                "Specific growth (15°C)"]]
growth.columns = ["pred_mass", "growth", "specific_growth"]
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


if __name__ == "__main__":
    traits = np.random.multivariate_normal(mean_zoop, A_zoop.values, 1000)
    
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



