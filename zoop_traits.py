import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.special import lambertw
import warnings

import phytoplankton_traits as pt

# traits relevant for pyhtoplankton growth rates
zoop_traits = ["size_Z", "mu_Z", "c_Z", "m_Z", "k_Z"]

# to store empirically measured data
raw_data = pd.DataFrame(columns = zoop_traits) 
allo_scal = {} # allometric scaling parameters

########################
# growth rates
growth = pd.read_csv("empirical_data/growth_rates_brun2017.csv",
                     encoding = 'ISO-8859-1')
growth = growth[["Body mass (mg)", "Specific growth (15°C)"]]
growth.columns = ["size_Z", "mu_Z"]


# unit conversions used
# mum^3 = 1e-18 m^3 =1e-18 (1e3 kg) = 1e-18 (1e3 *1e6 mg) = 1e-9 mg
uc = {"ml_L": 1000,
      "h_day": 24, # hours in a day
      "mum3_mg": 1e-9 }

# unit conversion (data is reported as per day, but comparison with
# Kiorboe 2014 shows that this is not the case, but as per hour)
growth.mu_Z *= uc["h_day"]

# log transform data
growth = np.log(growth)
growth = growth[np.isfinite(growth.mu_Z)]

raw_data = raw_data.append(growth, ignore_index=True)
allo_scal["mu_Z"] = 0

########################
# clearance rate data
clear = pd.read_csv("empirical_data/clearance_rates_brun2017.csv",
                    encoding = 'ISO-8859-1')
clear = clear[["Body mass (mg)", "Fmax (15°C)"]]
clear.columns = ["size_Z", "c_Z"]
#change clearance rate from ml to \mul
clear = np.log(clear)

raw_data = raw_data.append(clear, ignore_index=True)
allo_scal["c_Z"] = 1

########################
mortality = pd.read_csv("empirical_data/Hirst_Kiorboe_2002.csv")
mortality["size_Z"] /= 1000 # convert \mug to mg
mortality["size_Z"] *= 0.1 # convert DV to mg C
mortality = np.log(mortality)
raw_data = raw_data.append(mortality, ignore_index = True)
allo_scal["m_Z"] = -0.092 # value from Hirst_Kiorboe_2002

########################
# nutrient contents
res_cont = pd.read_csv("empirical_data/uye_1989.csv")
res_cont = res_cont[["Length", "N", "C", "DW"]]
res_cont["N"] /= 14.01 # convert grams to mol
res_cont["C"] /= 1000 # convert \mug to mg
#relative nutrient content
res_cont["R_conc"] = res_cont["N"]
"""
if clearance == "specific_fmax":
    res_cont["R_conc"] /= res_cont["C"]"""
res_cont = np.log(res_cont)
s_res, i_res, r_res, p_res, std_res = linregress(res_cont["C"],
                                                 res_cont["R_conc"])
# s_res is not distinguishable from 0:
if s_res -1.96*std_res < np.round(s_res) < s_res + 1.96*std_res:
    s_res = np.round(s_res)
    
#raw_data["k_Z"] = np.full(2, 0)
allo_scal["k_Z"] = s_res

###############################################################################
# empirically measured sizes
allo_scal["size_Z"] = 1

corr_theory = pd.DataFrame(np.full((len(zoop_traits),len(zoop_traits)), np.nan),
        index = zoop_traits, columns = zoop_traits)
mean_zoop = pd.DataFrame(columns = zoop_traits, index = [1])
std_zoop = pd.DataFrame(columns = zoop_traits, index = [1])
n_measurments = pd.DataFrame(np.full((len(zoop_traits), len(zoop_traits)), np.nan),
                       index = zoop_traits, columns = zoop_traits)

raw_data = raw_data.astype(float)
for i,trait in enumerate(zoop_traits):
    if trait == "k_Z":
        continue
    # remove outliers
    perc = np.nanpercentile(raw_data[trait], [25,75])
    iqr = perc[1]-perc[0]
    ind = ((raw_data[trait] > perc[0] - 1.5*iqr) &
           (raw_data[trait] < perc[1] + 1.5*iqr))
    raw_data.loc[~ind, trait] = np.nan
    
    
    mean_zoop[trait] = np.nanmean(raw_data[trait])
    std_zoop[trait] = np.nanstd(raw_data[trait])

size_var = np.nanvar(raw_data["size_Z"])
size_std = np.nanstd(raw_data["size_Z"])
  
for i,trait in enumerate(zoop_traits):
    corr_theory.loc[trait, trait] = 1
    for j, traitj in enumerate(zoop_traits):
        if i != j: # different traits
            corr_theory.loc[trait, traitj] = (allo_scal[trait]*allo_scal[traitj]
                        *size_var
                        /(std_zoop[trait]*std_zoop[traitj])).values
            n_measurments.loc[trait, traitj] = np.sum(np.isfinite(
                                        raw_data[trait]*raw_data[traitj]))

def nan_linreg(x,y):
    x,y = raw_data[x].values, raw_data[y].values
    ind = np.isfinite(x*y)
    if np.sum(ind) == 0:
        return [0, 0, 0, 0, np.inf]
    return linregress(x[ind], y[ind])

with warnings.catch_warnings(record = True):           
    # compute confidence interval of correlation based on empirical measurments
    corr_empirical = raw_data.corr()
    # convert to fisher z'
    fisher_z = np.arctanh(corr_empirical.values)
    # confidence interval
    alpha = 1.96 # 95% confidence 
    alpha = np.array([-alpha, alpha]).reshape(-1,1,1)
    fisher_confidence = fisher_z + alpha/np.sqrt(n_measurments.values - 3)
    corr_confidence = np.tanh(fisher_confidence)
    
    # where not in confidence interval
    corr_zoop = corr_theory.copy()
    ind = (corr_zoop<corr_confidence[0]) | (corr_zoop>corr_confidence[1])
    corr_zoop[ind] = corr_empirical
    


###############################################################################
# handle special cases for k_Z = mu_Z*q_min
# average resource concentration per species
q_mean = np.nanmean(res_cont["R_conc"])

# q_min = s*size_Z + noise, select variance of nois to have correct r_res
var_q_mean = size_var*(1+s_res**2*(1-r_res**2)/r_res**2)
std_zoop["k_Z"] = np.sqrt(var_q_mean + std_zoop["mu_Z"])
q_min_q_mean = np.log(10) # ration between mean and min
mean_zoop["k_Z"] = q_mean + mean_zoop["mu_Z"] - np.log(q_min_q_mean)
mean_zoop["k_Z"] = mean_zoop["k_Z"] - np.log(uc["h_day"]) # change units to hours

# corelation between k_z and other parameters
# k_Z = q_min * mu_Z combined correlation, mu_Z and q_min are uncorrelated
traitj = "k_Z"
for i, trait in enumerate(zoop_traits):
    if trait == "k_Z":
        continue
    corr_zoop.loc[trait, traitj] = (allo_scal[trait]*allo_scal[traitj]
                        *size_var
                        /(std_zoop[trait]*std_zoop[traitj])).values
    corr_zoop.loc[traitj, trait] = corr_zoop.loc[trait, traitj]

# correlation between k_Z and maximum growth
corr_zoop.loc["k_Z", "mu_Z"] = (std_zoop["mu_Z"]/std_zoop["k_Z"]).values
corr_zoop.loc["mu_Z", "k_Z"] = (std_zoop["mu_Z"]/std_zoop["k_Z"]).values

# increase growth rate of zooplankton, because of couple holling types
mean_zoop["mu_Z"] += np.log(4)
# the base covariance matrix of phytoplankton
cov_base = corr_zoop*std_zoop.values*std_zoop.values[0,:,np.newaxis]


# zooplankton prefer phytoplankton that are about 40**-3 times smaller
# we scale such that mean size zoop prefer mean sized phyto
zoop_pref = (mean_zoop["size_Z"] - (pt.mean_phyto["size_P"]
                                   + np.log(uc["mum3_mg"]))).values

# this corresponds to zoop prefering 20**-3 times smaller
np.exp(zoop_pref)**(1/3)

# variance of noise term
sig_size_noise = np.sqrt(size_var - pt.size_var)
"""
# conditional trait distributions, assuming size is known
a = zoop_traits[1:]
s = "size_Z"
A_num = A_zoop.values
A_conditional = A_num[1:,1:] - A_num[1:,[0]].dot(1/A_num[0,0]*A_num[[0],1:])




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
"""


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import generate_plankton as gp
    import plankton_growth as pg
    import warnings

    # generate communities
    n_coms = 5000
    r_phyto = 2
    traits = gp.generate_plankton(r_phyto, n_coms)

    env = gp.generate_env(n_coms)
    
    # compute maximum attained growth rate
    traits = gp.phytoplankton_equilibrium(traits, env)
    N = np.append(traits["N_star_P_res"], np.zeros((n_coms, 1)), axis = 1)
    with warnings.catch_warnings(record = True):
        traits["mu_Z_effective"] = pg.per_cap_plankton_growth(N,traits, env)[:,r_phyto:]

    traits = np.array([traits["size_Z"], traits["mu_Z_effective"],
                       traits["c_Z"],  traits["m_Z"], traits["k_Z"]])
    trait_names = ["Size", "Growth", "Clearance", "Mortality",
                   "Half\nsaturation"]
    with warnings.catch_warnings(record = True):
        traits = np.log(traits.reshape(len(trait_names),-1)).T
    bins = 15
    
    n = len(zoop_traits)
    fig, ax = plt.subplots(n,n,figsize = (9,9), sharex = "col", sharey = "row")
    
    
    fs = 16
    s = 5
    for i in range(n):
        for j in range(n):
            if i>j:
                ax[i,j].scatter(traits[:,j], traits[:,i], s = 2, alpha = 0.1,
                                color = "blue")
                ax[i,j].scatter(raw_data[zoop_traits[j]],
                                raw_data[zoop_traits[i]],
                                    s = 3, color = "orange")
            if j>i:
                ax[i,j].set_frame_on(False)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
        ax_hist = fig.add_subplot(n,n,1 + (n+1)*i)
        ax_hist.hist(traits[:,i], bins, density = True, color = "blue")
        ax_hist.set_xticklabels([])
        ax_hist.set_yticklabels([])
        ax_hist.set_title(trait_names[i], fontsize = fs)
        if (zoop_traits[i] != "k_Z"):
            ax_hist.hist(raw_data[zoop_traits[i]], bins, density = True,
                    alpha = 0.5, color = "orange")
        ax[i,0].set_ylabel(trait_names[i], fontsize = fs, rotation = 0,
                           ha = "right")
        ax[-1,i].set_xlabel(trait_names[i], fontsize = fs)
    
    
    ax[0,0].set_ylim(ax[0,0].get_xlim())
    ax[-1,-1].set_xlim(ax[-1,-1].get_ylim())

    fig.savefig("Figure_zooplankton_traits.png")