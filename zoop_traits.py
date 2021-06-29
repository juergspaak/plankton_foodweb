import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.special import lambertw

import phytoplankton_traits as pt

# empirically measured traits
raw_data = {}
allo_scal = {} # allometric scaling values

########################
# growth rates
growth = pd.read_csv("empirical_data/growth_rates_brun2017.csv",
                     encoding = 'ISO-8859-1')
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

raw_data["mu_Z"] = growth.specific_growth.values
allo_scal["mu_Z"] = 0

########################
# clearance rate data
clear = pd.read_csv("empirical_data/clearance_rates_brun2017.csv",
                    encoding = 'ISO-8859-1')
clear = clear[["Body mass (mg)", "Fmax (15°C)",
               "Specific Fmax (15°C)"]]
clear.columns = ["pred_mass", "fmax", "specific_fmax"]
#change clearance rate from ml to \mul
clear = np.log(clear)
clearance = "fmax"

# allometric scaling for clearance rate
s_c, i_c, r_c, p_c, std_c = linregress(clear["pred_mass"],
                                                clear[clearance])
# s_c is not distinguishable from 1:
if s_c -1.96*std_c < 1 < s_c + 1.96*std_c:
    s_c = 1

raw_data["c_Z"] = clear[clearance].values
allo_scal["c_Z"] = s_c

########################
mortality = pd.read_csv("empirical_data/Hirst_Kiorboe_2002.csv")
mortality["size_Z"] /= 1000 # convert \mug to mg
mortality["size_Z"] *= 0.1 # convert DV to mg C
mortality = np.log(mortality)
raw_data["m_Z"] = mortality["m_Z"].values
allo_scal["m_Z"] = -0.092 # value from Hirst_Kiorboe_2002

########################
# nutrient contents
res_cont = pd.read_csv("empirical_data/uye_1989.csv")
res_cont = res_cont[["Length", "N", "C", "DW"]]
res_cont["N"] /= 14.01 # convert grams to mol
res_cont["C"] /= 1000 # convert \mug to mg
#relative nutrient content
res_cont["R_conc"] = res_cont["N"]
if clearance == "specific_fmax":
    res_cont["R_conc"] /= res_cont["C"]
res_cont = np.log(res_cont)
s_res, i_res, r_res, p_res, std_res = linregress(res_cont["C"],
                                                 res_cont["R_conc"])
# s_res is not distinguishable from 0:
if s_res -1.96*std_res < np.round(s_res) < s_res + 1.96*std_res:
    s_res = np.round(s_res)
    
raw_data["k_Z"] = np.full(2, 0)
allo_scal["k_Z"] = s_res

########################
# empirically measured sizes
raw_data["size_Z"] = np.append(np.append(growth.pred_mass, clear.pred_mass),
                               mortality["size_Z"])
#raw_data["size_Z"] = np.append(growth.pred_mass, clear.pred_mass)
allo_scal["size_Z"] = 1

zoop_traits = ["size_Z", "mu_Z", "c_Z", "m_Z", "k_Z"]
A_zoop = pd.DataFrame(np.full((len(zoop_traits),len(zoop_traits)), np.nan),
        index = zoop_traits, columns = zoop_traits)
mean_zoop = pd.DataFrame(columns = zoop_traits, index = [1])

for i,trait in enumerate(zoop_traits):
    # remove outliers
    perc = np.nanpercentile(raw_data[trait], [25,75])
    iqr = perc[1]-perc[0]
    ind = ((raw_data[trait] > perc[0] - 1.5*iqr) &
           (raw_data[trait] < perc[1] + 1.5*iqr))
    raw_data[trait] = raw_data[trait][ind]
    
    if trait != "k_Z":
        mean_zoop[trait] = np.nanmean(raw_data[trait])
        A_zoop.loc[trait, trait] = np.nanvar(raw_data[trait])
    if trait == "size_Z":
        # record value for size variation
        size_var = np.nanvar(raw_data[trait])
    for j, traitj in enumerate(zoop_traits):
        if i != j: # different traits
            A_zoop.loc[trait, traitj] = (allo_scal[trait]*allo_scal[traitj]
                                         *size_var)
# remove data for k_Z
raw_data["k_Z"] = np.full(2, np.nan)

###############################################################################
# handle special cases for k_Z = mu_Z*q_min
# average resource concentration per species
q_mean = np.nanmean(res_cont["R_conc"])

# q_min = s*size_Z + noise, select variance of nois to have correct r_res
var_q_mean = A_zoop.loc["size_Z", "size_Z"]*(1+s_res**2*(1-r_res**2)/r_res**2)
A_zoop.loc["k_Z", "k_Z"] = (var_q_mean + A_zoop.loc["mu_Z", "mu_Z"])
q_min_q_mean = np.log(10) # ration between mean and min
mean_zoop["k_Z"] = q_mean + mean_zoop["mu_Z"] - np.log(q_min_q_mean)
mean_zoop["k_Z"] = mean_zoop["k_Z"] - np.log(uc["h_day"]) # change units to hours

# corelation between k_z and other parameters
# k_Z = q_min * mu_Z combined correlation
# covariance between size and clearance
A_zoop.loc["mu_Z", "k_Z"] = (A_zoop.loc["mu_Z", "mu_Z"]
                             + s_res*A_zoop.loc["mu_Z", "size_Z"])
A_zoop.loc["k_Z", "mu_Z"] = (A_zoop.loc["mu_Z", "mu_Z"]
                             + s_res*A_zoop.loc["mu_Z", "size_Z"])

# increase growth rate of zooplankton by 15%, because of couple holling types
mean_zoop["mu_Z"] += np.log(4)
    

def generate_zooplankton_traits(r_spec = 1, n_com = 100):
    traits = np.exp(np.random.multivariate_normal(mean_zoop.values[0],
                                                  A_zoop,
                                                  (n_com,r_spec)))
    trait_dict = {}
    for i, trait in enumerate(zoop_traits):
        trait_dict[trait] = traits[...,i]
    
    return trait_dict

# conditional trait distributions, assuming size is known
a = zoop_traits[1:]
s = "size_Z"
A_num = A_zoop.values
A_conditional = A_num[1:,1:] - A_num[1:,[0]].dot(1/A_num[0,0]*A_num[[0],1:])


# zooplankton prefer phytoplankton that are about 40**-3 times smaller
# we scale such that mean size zoop prefer mean sized phyto
zoop_pref = (mean_zoop["size_Z"] - (pt.mean_phyto["size_P"]
                                   + np.log(uc["mum3_mg"]))).values
# this corresponds to zoop prefering 20**-3 times smaller
np.exp(zoop_pref)**(1/3)

# variance of noise term
sig_size_noise = np.sqrt(A_zoop.loc["size_Z", "size_Z"]
                         - pt.A_phyto.loc["size_P", "size_P"])
def generate_conditional_zooplankton_traits(phyto):
    # generate zooplankton assuming they adapted to phyto_sizes
    size_Z = np.log(phyto["size_P"]*uc["mum3_mg"]*np.exp(zoop_pref))
    # add noise
    size_Z = size_Z + np.random.normal(0, sig_size_noise, size_Z.shape)
    size_Z = size_Z.reshape(-1,1)
    
    other_traits = (mean_zoop.values[:,1:] +
                    A_zoop.loc[a,s].values/
                    A_zoop.loc[s,s]*(size_Z-mean_zoop["size_Z"].values))
    other_traits += np.random.multivariate_normal(np.zeros(len(a)),
                                                  A_conditional,
                                                  (size_Z.size))
    trait_dict = {"size_Z": np.exp(size_Z).reshape(phyto["size_P"].shape)}
    for i, trait in enumerate(zoop_traits[1:]):
        trait_dict[trait] = np.exp(other_traits[...,i]).reshape(
                                                        phyto["size_P"].shape)
    return trait_dict

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    traits = np.random.multivariate_normal(mean_zoop.values[0],
                                           A_zoop.values, 1000)
    traits = generate_conditional_zooplankton_traits(
        pt.generate_phytoplankton_traits(1,1000))
    traits = {key: np.log(traits[key].flatten()) for key in traits.keys()}
    traits = np.array([traits["size_Z"], traits["mu_Z"],
                       traits["c_Z"],  traits["m_Z"], traits["k_Z"]]).T
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
        if (zoop_traits[i] != "k_Z"):
            ax_hist.hist(raw_data[zoop_traits[i]], bins, density = True,
                    alpha = 0.5, color = "orange")
        ax[i,0].set_ylabel(zoop_traits[i])
        ax[-1,i].set_xlabel(zoop_traits[i])
        
    ax[1,0].scatter(growth.pred_mass, growth.specific_growth, alpha = 0.5,
                    color = "orange")
    ax[2,0].scatter(clear.pred_mass, clear[clearance], alpha = 0.5,
                    color = "orange")
    ax[3,0].scatter(mortality.size_Z, mortality.m_Z, alpha = 0.5,
                    color = "orange")
    
    
    ax[0,0].set_ylim(ax[0,0].get_xlim())
    ax[-1,-1].set_xlim(ax[-1,-1].get_ylim())

    fig.savefig("Figure_zooplankton_traits.pdf")