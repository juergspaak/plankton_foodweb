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
                           "mu_nit", "k_nit_m", "vmax_nit", "volume"]].copy()
phyto_traits.columns = ["mu_p", # maximal phosphor growth rate [day^-1]
                "k_p", # half saturation for phosphor growth rate [mumol L^-1]
                "c_p", # phosphor consumption rate [mumol P cell^-1 day^-1]
                "mu_n", # maximal nitrate growth rate [day^-1]
                "k_n", # half saturation for nitrate growth rate [mumol L^-1]
                "c_n", # nitrate uptake rate [mumol N cell^-1 day^-1]
                "size"] # cell volume [\mum ^3]

# add an arbitrary mortality rate
phyto_traits["m"] = 0.1

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
gaussians = pd.DataFrame(data = None)
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
    gaussians[key] = norm.fit(phyto_traits.loc[ind, key])
    raw_data[key] = phyto_traits.loc[ind, key]


# remove outliers ad fit gaussians to light traits
for key in light_data.keys():
    perc = np.nanpercentile(light_data[key], [25,75])
    iqr = perc[1]-perc[0]
    ind = ((light_data[key] > perc[0] - 1.5*iqr) &
            (light_data[key] < perc[1] + 1.5*iqr))
    light_data[key][~ind] = np.nan
    gaussians[key] = norm.fit(light_data.loc[ind, key])
    raw_data[key] = light_data.loc[ind, key]
    
    
# fit a gaussian separately for intrinsic growth rates
mu_data = phyto_traits[["mu_p", "mu_n"]].values.flatten()
mu_data = np.append(mu_data, light_data.mu_l)
gaussians["mu"] = norm.fit(mu_data[np.isfinite(mu_data)])

gaussians = gaussians.drop(["mu_n", "mu_p", "mu_l"], axis = 1)
del raw_data["mu_n"], raw_data["mu_p"], raw_data["mu_l"]
raw_data["mu"] = mu_data[np.isfinite(mu_data)]   


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
gaussians = gaussians.append({"mu": -0.25, "k_p": 0.5, "k_n": 0.5,
                              "c_n": 2/3, "c_p": 2/3,
                              "a": 0.23,
                              "k_l": cor_mul_kl*(-0.25)}, ignore_index = True)
# scaling factor_maximum value
gaussians = gaussians.append({"mu": -0.25, "k_p": 0.5, "k_n": 0.5,
                              "c_n": 2/3, "c_p": 2/3,
                              "a": 0.69,
                              "k_l": cor_mul_kl*(-0.25)}, ignore_index = True)



# add allometric scaling information
gaussians = gaussians.T
gaussians.columns = ["mean_trait", "std_trait", "beta_min", "beta_max"]

# size disribution mean and standarddeviation
mean_size = np.nanmean(phyto_traits["size"])
std_size = np.nanmin(np.abs(gaussians["std_trait"]/gaussians["beta_min"]))
std_size = np.nanstd(raw_data["size"])


# intercept of allometric scaling
gaussians["alpha"] = gaussians.mean_trait - gaussians.beta_min*mean_size

# random noise added to allometric scaling
gaussians["std_err"] = np.sqrt(-(gaussians.beta_min*std_size)**2 +
                               gaussians.std_trait**2)


###############################################################################
# add three way tradeof for nitrogen, phosphorus and size
# data from https://doi-org.proxy.library.cornell.edu/10.1890/11-0395.1

# find the plane on close to which all species lie
# from paper
ax_1 = np.array([0.17, 0.65, -0.74]) # first pca of data
ax_2 = np.array([0.46, -0.86, 0.21]) # second pca of data

# change to normal vector for equation format of plane a*x + b*y + c*z + d = 0
normal_vec = np.cross(ax_1, ax_2)

# change to Hesse normal form
normal_vec = normal_vec/np.linalg.norm(normal_vec)

# tree species have all R_star for both resources, but no size measured
both_R_star = np.all(np.isfinite(phyto_traits[["R_star_n", "R_star_p"]]),
                     axis = 1)
phyto_data.loc[both_R_star, :]
#○ the volumes of these species can be found in
# Planktothrix agardhii 35:  https://doi.org/10.1080/03680770.1992.11900601
# Selenastrum capricornutum 10**1.9: https://doi.org/10.1093/plankt/13.4.863
# eucampia zodiacus 8640: https://doi.org/10.1016/j.ecss.2015.05.026 

phyto_traits.loc[both_R_star, "size"] = np.log([35, 10**1.9, 8640])


# find intercept d
d_real = phyto_traits[["R_star_n","R_star_p","size"]].dot(
            normal_vec)[both_R_star]
min_d, max_d = min(d_real), max(d_real)

def generate_phytoplankton_traits(r_spec, n_com, m = 0.1,
                                  tradeofs = ["size", "resources"]):
    # generate to many species, as some do not follow the tradeofs
    fac = 100
    # asign a size category to each species
    if "size" in tradeofs:
        size = np.random.normal(mean_size, std_size, (fac*r_spec* n_com))
    else:
        size = np.full(fac*r_spec*n_com, mean_size)
    traits = {"size": np.exp(size)}
    for index, row in gaussians.iterrows():
        if index == "size":
            continue
        error = np.random.normal(0, row.std_err,
                               (fac*r_spec* n_com))
        # trait of each species, allometric scaling + noise
        traits[index] = np.exp(row.alpha + row.beta_min*size + error)
    
    # affinity, major trait
    traits["aff_n"] = traits["mu"]/traits["k_n"]
    traits["aff_p"] = traits["mu"]/traits["k_p"]
    
    # R_star values with assumed mortality
    traits["R_star_p"] = m*traits["k_p"]/(traits["mu"] - traits["k_p"])
    traits["R_star_n"] = m*traits["k_n"]/(traits["mu"] - traits["k_n"])
    
    # remove species that do not follow resource competition tradeof
    if "resources" in tradeofs:
        # compute distance of each species to tradeof plane
        d = normal_vec.dot(np.log([traits["R_star_n"], traits["R_star_p"],
                                  traits["size"]])) - np.nanmean(d_real)
        ind = np.argsort(np.abs(d))
    else:
        ind = np.arange(n_com*r_spec)
    
    # return only realistic species
    for key in traits.keys():
        traits[key] = traits[key][ind][:n_com*r_spec].reshape(r_spec, n_com)
    
    
    return traits

if __name__ == "__main__":
    # illustrate trait distributions
    data = {}
    for tradeofs in ["size", "no", "resources","size_resources"]:
        data[tradeofs] = generate_phytoplankton_traits(10,100,
                                                       tradeofs = tradeofs)
        fig = plt.figure()
        fig.suptitle(tradeofs)
        ax = fig.add_subplot(121,projection="3d")
        ax.scatter(np.log(data["size"]["R_star_n"]),
                   np.log(data["size"]["R_star_p"]),
                   np.log(data["size"]["size"]), s = 2)
        ax.scatter(np.log(data[tradeofs]["R_star_n"]),
                   np.log(data[tradeofs]["R_star_p"]),
                   np.log(data[tradeofs]["size"]), s = 2, color = "red")
        
        ax = fig.add_subplot(122)
        ax.scatter(np.log(data["size"]["R_star_n"]),
                   np.log(data["size"]["R_star_p"]), s = 2)
        ax.scatter(np.log(data[tradeofs]["R_star_n"]),
                   np.log(data[tradeofs]["R_star_p"]), s = 2, color = "red")
        fig.tight_layout()
        fig.savefig("Figure_3_way_tradeof_{}.pdf".format(tradeofs))
        traits = data[tradeofs]
        
        
        fig, ax = plt.subplots(len(traits.keys()), len(traits.keys()),
                               figsize = (12,12), sharex = "col", sharey = "row")
        bins = 15
        keys = np.array(sorted(traits.keys()))
        for i,keyi in enumerate(keys):
            ax[0,i].set_title(keyi)
            ax[i,0].set_ylabel(keyi)
            for j, keyj in enumerate(keys):               
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
                ax_hist.hist(np.log(traits[keyi].flatten()),
                             bins, density = True)
                ax_hist.set_xticklabels([])
                ax_hist.set_yticklabels([])
                ax_hist.hist(raw_data[keyi], bins, density = True,
                             alpha = 0.5)
                #ax_hist.set_xlim(ax[0,i].get_xlim())
                
            except:
                pass
            
            
            # add real data for maximal growth rate
            index = np.arange(len(keys))
            ind_mu = index[keys == "mu"][0]
            for mu in ["mu_n", "mu_p"]:
                for t in ["k_p", "k_n", "c_n", "c_p"]:
                    ind_t = index[keys == t][0]
                    ax[ind_mu, ind_t].scatter(phyto_traits[t], phyto_traits[mu],
                                              s = 2)
                    
            ax[ind_mu, index[keys == "k_l"][0]].scatter(light_data["k_l"],
                                                        light_data["mu_l"],
                                                        s = 2)
            
        fig.savefig("Figure_gaussian_trait_distribution_{}.pdf".format(tradeofs))
        # """
    

