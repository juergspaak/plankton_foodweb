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


# unit conversions used
# mum^3 = 1e-18 m^3 =1e-18 (1e3 kg) = 1e-18 (1e3 *1e6 mg) = 1e-9 mg
uc = {"ml_L": 1000,
      "h_day": 24, # hours in a day
      "mum3_mg": 1e-9 }

# unit conversion (data is reported as per day, but comparison with
# Kiorboe 2014 shows that this is not the case, but as per hour)
growth.growth *= uc["h_day"]
growth.specific_growth *= uc["h_day"]

# log transform data
growth = np.log(growth)
growth = growth[np.isfinite(growth.growth)]

clear = pd.read_csv("clearance_rates_brun2017.csv", encoding = 'ISO-8859-1')
clear = clear[["Body mass (mg)", "Fmax (15°C)",
               "Specific Fmax (15°C)"]]
clear.columns = ["pred_mass", "fmax", "specific_fmax"]
#change clearance rate from ml to \mul
clear = np.log(clear)

# nutrient contents
res_cont = pd.read_csv("uye_1989.csv")
res_cont = res_cont[["Length", "N", "C", "DW"]]
res_cont["N"] /= 14.01 # convert grams to mol
res_cont["C"] /= 1000 # convert \mug to mg
#relative nutrient content
res_cont["R_conc"] = res_cont["N"]/res_cont["C"]
res_cont = np.log(res_cont)
s_res, i_res, r_res, p_res, std_res = linregress(res_cont["C"],
                                                 res_cont["R_conc"])
# s_res is not distinguishable from 0:
if s_res -1.96*std_res < 0 < s_res + 1.96*std_res:
    s_res = 0

# empirically measured traits
raw_data = {}
raw_data["mu_Z"] = growth.specific_growth.values

raw_data["c_Z"] = clear.specific_fmax.values
raw_data["size_Z"] = np.append(growth.pred_mass, clear.pred_mass)
raw_data["k_Z"] = np.full(2, 0)

zoop_traits = ["size_Z", "mu_Z", "c_Z", "k_Z"]
A_zoop = pd.DataFrame(np.zeros((len(zoop_traits),len(zoop_traits))),
        index = zoop_traits, columns = zoop_traits)
mean_zoop = np.empty(4)

for i,trait in enumerate(zoop_traits):
    #perc = np.nanpercentile(raw_data[trait], [25,75])
    #iqr = perc[1]-perc[0]
    #ind = ((raw_data[trait] > perc[0] - 1.5*iqr) &
    #        (raw_data[trait] < perc[1] + 1.5*iqr))
    #raw_data[trait] = raw_data[trait][ind]
    mean_zoop[i] = np.nanmean(raw_data[trait])
    A_zoop.loc[trait, trait] = np.nanvar(raw_data[trait])
    
# covariance between size and clearance, linear regression suggests 1
#A_zoop.loc["size_Z", "c_Z"] = A_zoop.loc["size_Z", "size_Z"]
A_zoop.loc["c_Z", "size_Z"] = A_zoop.loc["size_Z", "c_Z"]

###############################################################################
# add correlations for k_Z = q_min * mu_Z
# average resource concentration per species
q_mean = np.nanmean(res_cont["R_conc"])
# average resource concentration per mg, divide by average species size
#q_mean += mean_zoop[0]

# q_min = s*size_Z + noise, select variance of nois to have correct r_res
var_q_mean = A_zoop.loc["size_Z", "size_Z"]*(1+s_res**2*(1-r_res**2)/r_res**2)
A_zoop.loc["k_Z", "k_Z"] = (np.nanvar(res_cont["R_conc"])
                            + A_zoop.loc["mu_Z", "mu_Z"])
q_min_q_mean = np.log(10) # ration between mean and min
mean_zoop[3] = q_mean + mean_zoop[1] - np.log(q_min_q_mean)
mean_zoop[3] = mean_zoop[3] - np.log(uc["h_day"]) # change units to hours

# corelation between k_z and other parameters
# k_Z = q_min * mu_Z combined correlation
A_zoop.loc["k_Z", "size_Z"] = (s_res*A_zoop.loc["size_Z", "size_Z"]
                             + A_zoop.loc["mu_Z", "size_Z"])
A_zoop.loc["size_Z", "k_Z"] = (s_res*A_zoop.loc["size_Z", "size_Z"]
                             + A_zoop.loc["mu_Z", "size_Z"])

A_zoop.loc["mu_Z", "k_Z"] = (A_zoop.loc["mu_Z", "mu_Z"]
                             + s_res*A_zoop.loc["mu_Z", "size_Z"])
A_zoop.loc["k_Z", "mu_Z"] = (A_zoop.loc["mu_Z", "mu_Z"]
                             + s_res*A_zoop.loc["mu_Z", "size_Z"])

"""A_zoop.loc["c_Z", "k_Z"] = (s_res*A_zoop.loc["size_Z", "c_Z"] +
                        A_zoop.loc["mu_Z", "c_Z"])
A_zoop.loc["k_Z", "c_Z"] = (s_res*A_zoop.loc["size_Z", "c_Z"] +
                        A_zoop.loc["mu_Z", "c_Z"])
"""


def generate_zooplankton_traits(r_spec = 1, n_com = 100):
    traits = np.exp(np.random.multivariate_normal(mean_zoop, A_zoop,
                                                  (n_com,r_spec)))
    trait_dict = {}
    for i, trait in enumerate(zoop_traits):
        trait_dict[trait] = traits[...,i]
    
    return trait_dict

# conditional trait distributions, assuming size is known
a = ["mu_Z", "c_Z", "k_Z"]
s = "size_Z"
A_num = A_zoop.values
A_conditional = A_num[1:,1:] - A_num[1:,[0]].dot(1/A_num[0,0]*A_num[[0],1:])


# zooplankton prefer phytoplankton that are about 40**-3 times smaller
# we scale such that mean size zoop prefer mean sized phyto
zoop_pref = mean_zoop[0] - (pt.mean_traits[0] + np.log(uc["mum3_mg"]))
# this corresponds to zoop prefering 20**-3 times smaller
np.exp(zoop_pref)**(1/3)

# variance of noise term
sig_size_noise = np.sqrt(A_zoop.loc["size_Z", "size_Z"]
                         - pt.cov_matrix.loc["size_P", "size_P"])
def generate_conditional_zooplankton_traits(phyto):
    # generate zooplankton assuming they adapted to phyto_sizes
    size_Z = np.log(phyto["size_P"]*uc["mum3_mg"]*np.exp(zoop_pref))
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

if __name__ == "__main__" and False:
    traits = np.random.multivariate_normal(mean_zoop, A_zoop.values, 1000)
    traits = generate_conditional_zooplankton_traits(
        pt.generate_phytoplankton_traits(1,10000))
    traits = {key: np.log(traits[key].flatten()) for key in traits.keys()}
    traits = np.array([traits["size_Z"], traits["mu_Z"],
                       traits["c_Z"], traits["k_Z"]]).T
    traits = traits
    bins = 15
    
    n = len(zoop_traits)
    fig, ax = plt.subplots(n,n,figsize = (9,9), sharex = "col", sharey = "row")
    
    for i in range(n):
        for j in range(n):
            if i>j:
                ax[i,j].scatter(traits[:,j], traits[:,i], s = 2, alpha = 0.1,
                                color = "blue")
        ax_hist = fig.add_subplot(n,n,1 + (n+1)*i)
        ax_hist.hist(traits[:,i], bins, density = True, color = "blue")
        ax_hist.set_xticklabels([])
        ax_hist.set_yticklabels([])
        try:
            ax_hist.hist(raw_data[zoop_traits[i]], bins, density = True,
                    alpha = 0.5, color = "orange")
        except ValueError:
            pass
        ax[i,0].set_ylabel(zoop_traits[i])
        ax[-1,i].set_xlabel(zoop_traits[i])
        
    ax[1,0].scatter(growth.pred_mass, growth.specific_growth, alpha = 0.5,
                    color = "orange")
    ax[2,0].scatter(clear.pred_mass, clear.specific_fmax, alpha = 0.5,
                    color = "orange")
    
    ax[0,0].set_ylim(ax[0,0].get_xlim())
    ax[-1,-1].set_xlim(ax[-1,-1].get_ylim())

    fig.savefig("Figure_zooplankton_traits.pdf")

"""
data = pd.read_csv("Uiterwaal_2018_data.csv", encoding = "ISO-8859-1")
ind = data["Major grouping 2"] == "Copepod"
data = data[ind]
trait = "Fittted h (day)"
data.loc[data[trait] < 1e-6/60/60/24, trait] = np.nan


data[trait] = np.log(data[trait])
perc = np.nanpercentile(data[trait],[25,75])

iqr = perc[1]-perc[0]
ind = ((data[trait] > perc[0] - 1.5*iqr) &
       (data[trait] < perc[1] + 1.5*iqr))
data = data.loc[ind]

plt.hist(data[trait],bins = 30)"""