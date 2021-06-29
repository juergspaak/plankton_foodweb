import numpy as np
import pandas as pd
import warnings
from scipy.stats import linregress

# traits relevant for pyhtoplankton growth rates
phyto_traits = np.array(["size_P", "mu_P", "k_n", "k_p", "k_l",
                         "c_n", "c_p", "a","e_P", "R_P"])

# to store empirically measured data
raw_data = pd.DataFrame(columns = phyto_traits) 
allo_scal = {} # allometric scaling parameters

# gaussians stores log normal distribution
A_phyto = pd.DataFrame(np.full((len(phyto_traits), len(phyto_traits)), np.nan),
                       index = phyto_traits, columns = phyto_traits)


##############################################################################
"""
phytoplankton traits are taken from
https://esapubs.org/archive/ecol/E096/202/
"""
phyto_data = pd.read_csv("empirical_data/Phytoplankton traits.csv")
# data from outside this temperature range is rejected
temp_range = [15, 25]
phyto_data = phyto_data[(phyto_data.temperature <= temp_range[1]) &
                        (phyto_data.temperature >= temp_range[0])]
phyto_data = phyto_data[["mu_p", "k_p_m", "vmax_p",
                           "mu_nit", "k_nit_m", "vmax_nit", "volume",
                           "qmin_nit"]].copy()
phyto_data.columns = ["mu_p", # maximal phosphor growth rate [day^-1]
                "k_p", # half saturation for phosphor growth rate [mumol L^-1]
                "c_p", # phosphor consumption rate [mumol P cell^-1 day^-1]
                "mu_n", # maximal nitrate growth rate [day^-1]
                "k_n", # half saturation for nitrate growth rate [mumol L^-1]
                "c_n", # nitrate uptake rate [mumol N cell^-1 day^-1]
                "size_P", # cell volume [\mum ^3]
                "R_P"] # nitrogen concentration [\mumol N cell^-1]

# change q_min to mean resource concentration
phyto_data["R_P"] = phyto_data["R_P"]*10
with warnings.catch_warnings(record = True):
    phyto_data["mu_P"] = np.nanmean(phyto_data[["mu_p", "mu_n"]], axis = 1)
del phyto_data["mu_n"], phyto_data["mu_p"]

with warnings.catch_warnings(record = True):
    phyto_data = np.log(phyto_data)
    
raw_data = raw_data.append(phyto_data, ignore_index=True)
    
allo_scal.update(dict(k_p = 1/3, k_n = 1/3, c_p = 2/3, c_n = 2/3, size_P = 1,
                      R_P = 0.8, mu_P = -0.25))

##############################################################################
# data from https://aslopubs.onlinelibrary.wiley.com/doi/epdf/10.1002/lno.10282
#light traits
light_data = pd.read_csv("empirical_data/Light_traits.csv")
# only take species where model 1 was acceptable fit
ind = light_data.AIC_1 > light_data.AIC_2 - 10
light_data = light_data[ind]
light_data["k_l"] = light_data.mu_l/light_data.alpha
light_data = light_data[["mu_l",# maximum light growth rate [day^-1]
    "k_l"# halfsaturation constant [day^-1 quanta^-1 mumol photon m^-2s^-1]
    ]].copy()
light_data = np.log(light_data)
light_data.columns = ["mu_P", "k_l"]

raw_data = raw_data.append(light_data, ignore_index=True)

##############################################################################
# data from ehrlich 2019
# defense data
defense_data = pd.read_csv("empirical_data/ehrlich2020.csv")
defense_data = defense_data[["Cell volume", "r", "Defense"]]
defense_data.columns = ["size_P", "mu_P", "e_P"]
defense_data["e_P"] = 1 - defense_data["e_P"]
defense_data = np.log(defense_data)

raw_data = raw_data.append(defense_data, ignore_index=True)

##############################################################################
# data from augusti1989
# phytoplankton absorption coefficients
augusti = pd.read_csv("empirical_data/augusti_data.csv")
augusti = augusti[["d", "a"]]
augusti["d"] = augusti["d"]**3 # convert diameter to volume
augusti["a"] = augusti["a"]*1e-6 # convert \mum^2 to mm^2
augusti = np.log(augusti)

raw_data = raw_data.append(augusti, ignore_index=True)
allo_scal["a"] = 0.77
###############################################################################
mean_phyto = pd.DataFrame(columns = phyto_traits, index = [1])

# remove outliers and fit one dimensional trait data
for i,trait in enumerate(phyto_traits):
    # remove outliers
    perc = np.nanpercentile(raw_data[trait], [25,75])
    iqr = perc[1]-perc[0]
    ind = ((raw_data[trait] > perc[0] - 1.5*iqr) &
           (raw_data[trait] < perc[1] + 1.5*iqr))
    raw_data.loc[~ind, trait] = np.nan
    
    mean_phyto[trait] = np.nanmean(raw_data[trait])
    A_phyto.loc[trait, trait] = np.nanvar(raw_data[trait])
    
def nan_linreg(x,y):
    x,y = raw_data[x].values, raw_data[y].values
    ind = np.isfinite(x*y)
    if np.sum(ind) == 0:
        return [0, 0, 0, 0, np.inf]
    return linregress(x[ind], y[ind])


size_var = np.nanvar(raw_data["size_P"]) 
s, i, r, p, std = nan_linreg("mu_P", "k_l")
allo_scal["k_l"] = allo_scal["mu_P"]*s
allo_scal["e_P"] = nan_linreg("size_P", "e_P")[0]

for i,trait in enumerate(phyto_traits):
    for j, traitj in enumerate(phyto_traits):
        if i != j: # different traits
            A_phyto.loc[trait, traitj] = (allo_scal[trait]*allo_scal[traitj]
                                         *size_var)
            
# compare to correlations in data
tr_norm = raw_data.values - np.nanmean(raw_data.values, axis = 0)
with warnings.catch_warnings(record = True):
    A_phyto_measured = np.nanmean(tr_norm.T[...,np.newaxis]*tr_norm, axis = 1)
##############################################################################
# find parameters for scaling size to traits
"""
#scaling factors for size, minimum value
# data from doi:10.1093/plankt/fbp098, Finkel et al 2010
gaussians = gaussians.append({"mu_P": -0.25, "k_p": 0.5, "k_n": 0.5,
                              "c_n": 2/3, "c_p": 2/3,
                              "a": 0.77, "size_P": 1,
                              "k_l": cor_mul_kl*(-0.25),
                              "R_P": 0.8}, ignore_index = True)


# add allometric scaling information
gaussians = gaussians.T
gaussians.columns = ["mean_trait", "std_trait", "beta_min"]

# size disribution mean and standarddeviation
mean_size = np.nanmean(phyto_traits["size_P"])
std_size = np.nanmin(np.abs(gaussians["std_trait"]/gaussians["beta_min"]))
std_size = np.nanstd(raw_data["size_P"])
"""


# add certain tradeoffs
A_tradeoff = pd.read_csv("empirical_data/Three_way_tradeoff.csv", index_col=0)
cov_tradeoff = A_phyto.copy()

for traiti in A_tradeoff.columns:
    for traitj in A_tradeoff.columns:
        cov_tradeoff.loc[traiti, traitj] = A_tradeoff.loc[traiti, traitj]
    
def generate_phytoplankton_traits(r_spec = 1, n_com = 100):
    # generate multivariate distribution
    traits = np.exp(np.random.multivariate_normal(mean_phyto.values[0],
                                                  A_phyto,
                                                  (n_com, r_spec)))
    # order species according to their size
    order = np.argsort(traits[...,0], axis = 1)
    trait_dict = {}
    for i, trait in enumerate(phyto_traits):
        trait_dict[trait] = traits[np.arange(n_com)[:,np.newaxis], order,i]
    
    return trait_dict

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    traits = generate_phytoplankton_traits(10,100)
    traits = {key: np.log(traits[key].flatten()) for key in traits.keys()}
    traits = pd.DataFrame(traits, columns = phyto_traits)
    
    fig, ax = plt.subplots(len(traits.keys()), len(traits.keys()),
                               figsize = (12,12), sharex = "col", sharey = "row")
    bins = 10
    for i,keyi in enumerate(phyto_traits):
        ax[-1,i].set_xlabel(keyi)
        ax[i,0].set_ylabel(keyi)
        for j, keyj in enumerate(phyto_traits):               
            if i<j:
                
                ax[j,i].scatter(traits[keyi], traits[keyj], s = 1,
                            alpha = 0.1, color = "blue")
                
                ax[j,i].scatter(raw_data[keyi], raw_data[keyj],
                                    s = 3, color = "orange")
        
        # plot histogram

        ax_hist = fig.add_subplot(len(traits.keys()),
                                  len(traits.keys()),
                                  1 + i + i*len(traits.keys()))
        ax_hist.hist(traits[keyi], bins, density = True, color = "blue")
        ax_hist.set_xticklabels([])
        ax_hist.set_yticklabels([])
        ax_hist.hist(raw_data[keyi], bins, density = True,
                     alpha = 0.5, color = "orange")
            
        ax[-1,-1].set_xlim(ax[-1,0].get_ylim())

    fig.savefig("Figure_phytoplankton_traits.pdf")
