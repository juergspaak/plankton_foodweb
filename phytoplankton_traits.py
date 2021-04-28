import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import warnings


# data from outside this temperature range is rejected
temp_range = [15, 25]
"""
phytoplankton traits are taken from
https://esapubs.org/archive/ecol/E096/202/
and 
https://aslopubs.onlinelibrary.wiley.com/doi/epdf/10.1002/lno.10282
"""
phyto_data = pd.read_csv("Phytoplankton traits.csv")
phyto_data = phyto_data[(phyto_data.temperature <= temp_range[1]) &
                        (phyto_data.temperature >= temp_range[0])]
phyto_traits = phyto_data[["mu_p", "k_p_m", "vmax_p",
                           "mu_nit", "k_nit_m", "vmax_nit", "volume",
                           "qmax_p"]].copy()
phyto_traits.columns = ["mu_p", # maximal phosphor growth rate [day^-1]
                "k_p", # half saturation for phosphor growth rate [mumol L^-1]
                "c_p", # phosphor consumption rate [mumol P cell^-1 day^-1]
                "mu_n", # maximal nitrate growth rate [day^-1]
                "k_n", # half saturation for nitrate growth rate [mumol L^-1]
                "c_n", # nitrate uptake rate [mumol N cell^-1 day^-1]
                "size_P", # cell colume [\mum ^3]
                "R_P"] # phosphorus concentration [\mumol P cell^-1]

# add an arbitrary mortality rate
m = 0.01
phyto_traits["m"] = m

# add affinities
phyto_traits["aff_n"] = phyto_traits["mu_n"]/phyto_traits["k_n"]
phyto_traits["aff_p"] = phyto_traits["mu_p"]/phyto_traits["k_p"]

# add R_star values
phyto_traits["R_star_p"] = (phyto_traits["m"]*phyto_traits["k_p"]
                            /(phyto_traits["mu_n"] - phyto_traits["k_p"]))
phyto_traits["R_star_n"] = (phyto_traits["m"]*phyto_traits["k_n"]
                            /(phyto_traits["mu_p"] - phyto_traits["k_n"]))

with warnings.catch_warnings(record = True):
    phyto_traits = np.log(phyto_traits)



#light traits
light_data = pd.read_csv("Light_traits.csv")
# only take species where model 1 was acceptable fit
ind = light_data.AIC_1 > light_data.AIC_2 -10
light_data = light_data[ind]
light_data["k_l"] = light_data.mu_l/light_data.alpha
light_data = light_data[["mu_l",# maximum light growth rate [day^-1]
    "k_l"# halfsaturation constant [day^-1 quanta^-1 mumol photon m^-2s^-1]
    ]].copy()
light_data = np.log(light_data)

# gaussians stores log normal distribution
gaussians = pd.DataFrame(None)
raw_data = {} # dictionary with all the raw data
# remove outliers and fit gaussians to trait data
for key in phyto_traits.keys():
    if key == "m":
        gaussians[key] = [np.log(0.1),0]
        continue
    perc = np.nanpercentile(phyto_traits[key], [25,75])
    iqr = perc[1]-perc[0]
    ind = ((phyto_traits[key] > perc[0] - 1.5*iqr) &
            (phyto_traits[key] < perc[1] + 1.5*iqr))
    phyto_traits.loc[~ind, key] = np.nan # remove outliers
    
    # fit  gaussian kernel
    gaussians[key] = (np.nanmean(phyto_traits.loc[ind, key]),
                        np.nanvar(phyto_traits.loc[ind, key]))
    raw_data[key] = phyto_traits.loc[ind, key]


# remove outliers ad fit gaussians to light traits
for key in light_data.keys():
    perc = np.nanpercentile(light_data[key], [25,75])
    iqr = perc[1]-perc[0]
    ind = ((light_data[key] > perc[0] - 1.5*iqr) &
            (light_data[key] < perc[1] + 1.5*iqr))
    light_data[key][~ind] = np.nan
    gaussians[key] = (np.nanmean(light_data.loc[ind, key]),
                        np.nanvar(light_data.loc[ind, key]))
    raw_data[key] = light_data.loc[ind, key]
    
    
# fit a gaussian separately for intrinsic growth rates
mu_data = phyto_traits[["mu_p", "mu_n"]].values.flatten()
mu_data = np.append(mu_data, light_data.mu_l)

gaussians = gaussians.drop(["mu_n", "mu_p", "mu_l"], axis = 1)
del raw_data["mu_n"], raw_data["mu_p"], raw_data["mu_l"]
# phytoplankton growth rate
raw_data["mu_P"] = mu_data[np.isfinite(mu_data)]

gaussians["mu_P"] = (np.nanmean(raw_data["mu_P"]),
                        np.nanvar(raw_data["mu_P"]))




# add fictional data for absorption coeficcient, to be adjusted xxx
gaussians["a"] = np.log(1e-7),  1  
##############################################################################
# find parameters for scaling size to traits

# find size scaling parameter of halfsaturation constant of light
ind = np.all(np.isfinite(light_data.values), axis = 1)
cor_mul_kl = np.cov(light_data.loc[ind, "mu_l"],
                    light_data.loc[ind, "k_l"])[0,1]


#scaling factors for size, minimum value
# data from doi:10.1093/plankt/fbp098, Finkel et al 2010
gaussians = gaussians.append({"mu_P": -0.25, "k_p": 0.5, "k_n": 0.5,
                              "c_n": 2/3, "c_p": 2/3,
                              "a": 0.23, "size_P": 1,
                              "k_l": cor_mul_kl*(-0.25),
                              "R_P": 0.8}, ignore_index = True)
# scaling factor_maximum value
gaussians = gaussians.append({"mu_P": -0.25, "k_p": 0.5, "k_n": 0.5,
                              "c_n": 2/3, "c_p": 2/3,
                              "a": 0.69,
                              "k_l": cor_mul_kl*(-0.25)}, ignore_index = True)



# add allometric scaling information
gaussians = gaussians.T
gaussians.columns = ["mean_trait", "std_trait", "beta_min", "beta_max"]

# size disribution mean and standarddeviation
mean_size = np.nanmean(phyto_traits["size_P"])
std_size = np.nanmin(np.abs(gaussians["std_trait"]/gaussians["beta_min"]))
std_size = np.nanstd(raw_data["size_P"])

"""
# intercept of allometric scaling
gaussians["alpha"] = gaussians.mean_trait - gaussians.beta_min*mean_size

# random noise added to allometric scaling
gaussians["std_err"] = np.sqrt(-(gaussians.beta_min*std_size)**2 +
                               gaussians.std_trait**2)
"""

##############################################################################
# fill in covariance matrix
trait_names = np.array(["size_P", "mu_P", "k_p", "k_n", "k_l", "c_p", "c_n", "a",
                        "R_P"])
cov_matrix = pd.DataFrame(np.zeros((len(trait_names), len(trait_names))),
                          index = trait_names, columns = trait_names)
mean_traits = np.empty(len(trait_names))

for i,trait in enumerate(trait_names):
    row = gaussians.loc[trait,:]
    # mean values for all traits
    mean_traits[i] = gaussians.loc[trait, "mean_trait"]
    # standard deviation for all traits
    cov_matrix.loc[trait, trait] = row.std_trait
    # allometric scaling
    for traitj in trait_names:
        if trait == traitj:
            continue
        cov_matrix.loc[trait, traitj] = (gaussians.loc["size_P", "std_trait"]*
                                         gaussians.loc[trait, "beta_min"]
                                         *gaussians.loc[traitj, "beta_min"])


# add certain tradeoffs
A_tradeoff = pd.read_csv("Three_way_tradeoff.csv", index_col=0)
cov_tradeoff = cov_matrix.copy()

for traiti in A_tradeoff.columns:
    for traitj in A_tradeoff.columns:
        cov_tradeoff.loc[traiti, traitj] = A_tradeoff.loc[traiti, traitj]
    
def generate_phytoplankton_traits(r_spec = 1, n_com = 100):
    # generate multivariate distribution
    traits = np.exp(np.random.multivariate_normal(mean_traits, cov_matrix,
                                                  (n_com, r_spec)))
    trait_dict = {}
    for i, trait in enumerate(trait_names):
        trait_dict[trait] = traits[...,i]
    
    return trait_dict
    

if __name__ == "__main__":
    traits = np.exp(np.random.multivariate_normal(mean_traits, cov_matrix, 1000))
    traits = pd.DataFrame(traits, columns = trait_names)
    
    fig, ax = plt.subplots(len(traits.keys()), len(traits.keys()),
                               figsize = (12,12), sharex = "col", sharey = "row")
    bins = 10
    for i,keyi in enumerate(trait_names):
        ax[-1,i].set_xlabel(keyi)
        ax[i,0].set_ylabel(keyi)
        for j, keyj in enumerate(trait_names):               
            if i<j:
                
                ax[j,i].scatter(np.log(traits[keyi]),
                                np.log(traits[keyj]), s = 1,
                            alpha = 0.1, color = "lightblue")
                try:
                    ax[j,i].scatter(phyto_traits[keyi], phyto_traits[keyj],
                                    s = 2)
                except:
                    pass
        
        # plot histogram
        try:
            ax_hist = fig.add_subplot(len(traits.keys()),
                                      len(traits.keys()),
                                      1 + i + i*len(traits.keys()))
            ax_hist.hist(np.log(traits[keyi]),
                         bins, density = True)
            ax_hist.set_xticklabels([])
            ax_hist.set_yticklabels([])
            ax_hist.hist(raw_data[keyi], bins, density = True,
                         alpha = 0.5)
            #ax_hist.set_xlim(ax[0,i].get_xlim())
            
        except KeyError:
            pass
        
        
        # add real data for maximal growth rate
        index = np.arange(len(trait_names))
        ind_mu = index[trait_names == "mu_P"][0]
        for mu_res in ["mu_n", "mu_p"]:
            for t in ["k_p", "k_n", "c_n", "c_p", "size_P"]:
                ind_t = index[trait_names == t][0]
                ax[ind_t, ind_mu].scatter(phyto_traits[mu_res], phyto_traits[t], 
                                          s = 2)
                
        ax[index[trait_names == "k_l"][0],ind_mu].scatter(light_data["mu_l"],
                                                          light_data["k_l"],
                                                    s = 2)
    fig.savefig("Figure_phytoplankton_traits.pdf")
