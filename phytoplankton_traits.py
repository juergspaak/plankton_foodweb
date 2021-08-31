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
augusti["a"] = augusti["a"]*1e-6 # convert \mum^2 to mm^2
del augusti["d"]
augusti = np.log(augusti)

raw_data = raw_data.append(augusti, ignore_index=True)
allo_scal["a"] = 0.77
###############################################################################
mean_phyto = pd.DataFrame(columns = phyto_traits, index = [1])
std_phyto = pd.DataFrame(columns = phyto_traits, index = [1])
corr_theory = pd.DataFrame(np.full((len(phyto_traits), len(phyto_traits)), np.nan),
                       index = phyto_traits, columns = phyto_traits)
n_measurments = pd.DataFrame(np.full((len(phyto_traits), len(phyto_traits)), np.nan),
                       index = phyto_traits, columns = phyto_traits)
# remove outliers and fit one dimensional trait data
for i,trait in enumerate(phyto_traits):
    # remove outliers
    perc = np.nanpercentile(raw_data[trait], [25,75])
    iqr = perc[1]-perc[0]
    ind = ((raw_data[trait] > perc[0] - 1.5*iqr) &
           (raw_data[trait] < perc[1] + 1.5*iqr))
    raw_data.loc[~ind, trait] = np.nan
    
    mean_phyto[trait] = np.nanmean(raw_data[trait])
    std_phyto[trait] = np.nanstd(raw_data[trait])
    
    
def nan_linreg(x,y):
    x,y = raw_data[x].values, raw_data[y].values
    ind = np.isfinite(x*y)
    if np.sum(ind) <= 1:
        return [0, 0, 0, 0, np.inf]
    return linregress(x[ind], y[ind])


size_var = np.nanvar(raw_data["size_P"])
size_std = np.nanstd(raw_data["size_P"])
s, i, r, p, std = nan_linreg("mu_P", "k_l")
allo_scal["k_l"] = allo_scal["mu_P"]*s
allo_scal["e_P"] = nan_linreg("size_P", "e_P")[0]

# check allometric scaling
fac = 1.96
allo_scal_emp = {}
for key in phyto_traits:
    s, i, r, p, std = nan_linreg("size_P", key)
    if not (s-fac*std <= allo_scal[key] <= s+fac*std):
        allo_scal[key] = s
    allo_scal_emp[key] = [s-fac*std, s+fac*std]

for i,trait in enumerate(phyto_traits):
    corr_theory.loc[trait, trait] = 1
    for j, traitj in enumerate(phyto_traits):
        if i != j: # different traits
            corr_theory.loc[trait, traitj] = (allo_scal[trait]*allo_scal[traitj]
                        *size_var
                        /(std_phyto[trait]*std_phyto[traitj])).values
            n_measurments.loc[trait, traitj] = np.sum(np.isfinite(
                                        raw_data[trait]*raw_data[traitj]))

corr_phyto = corr_theory.copy()
"""
def proj_cov(A, tol = 1e-10, max_iter = 100):
    '''
    Reference:  N. J. Higham, Computing the nearest correlation
    matrix---A problem from finance. IMA J. Numer. Anal.,
    22(3):329-343, 2002.
    '''

    dx = np.zeros(np.shape(A))
    # x projected onto covariance matrix space
    X_spd = np.copy(A)
    # relative difference between last and this iteration
    rel_diff = np.inf
    iteration = 1
    while rel_diff > tol:
        iteration +=1
        # avoid infinite loop
        if iteration > max_iter:
            raise RuntimeError
        # copy old value
        X_spd_old = np.copy(X_spd)
        
        # move away from covariance matrices
        X_off = X_spd - dx
        
        # project onto smi positive definite amtrices
        eigval, eigvec = np.linalg.eigh(X_off)
        X_spd = (eigvec*np.maximum(eigval,1e-7)).dot(eigvec.T)
        
        # find next step length
        dx = X_spd - X_off
        
        # project onto covariance matrices
        np.fill_diagonal(X_spd, 1)
        rel_diff = np.linalg.norm(X_spd - X_spd_old)/np.linalg.norm(X_spd)
        
    return X_spd

with warnings.catch_warnings(record = True):           
    # compute confidence interval of correlation based on empirical measurments
    corr_empirical = raw_data.corr()
    # convert to fisher z'
    fisher_z = np.arctanh(corr_empirical.values)
    # confidence interval
    # 95% confidence 
    alpha = np.array([-fac, fac]).reshape(-1,1,1)
    fisher_confidence = fisher_z + alpha/np.sqrt(n_measurments.values - 3)
    corr_confidence = np.tanh(fisher_confidence)
    
    # where not in confidence interval
    corr_phyto = corr_theory.copy()
    ind_di = (corr_phyto<corr_confidence[0]) | (corr_phyto>corr_confidence[1])
    # correlation between size_P has already been checked
    ind_di["size_P"] = False
    ind_di.loc["size_P",:] = False
    #ind_di.loc["mu_P", ]
    corr_phyto[ind_di] = corr_empirical
    
    
# this matrix might not be positive semidefinite (i.e. not a covariance matrix)
# find the closest covariance matrix (measured in Frobenius norm)
corr_phyto = pd.DataFrame(proj_cov(corr_phyto), index = phyto_traits,
                          columns = phyto_traits)"""

# the base covariance matrix of phytoplankton
cov_phyto = corr_phyto*std_phyto.values*std_phyto.values[0,:,np.newaxis]
cov_base = corr_phyto*std_phyto.values*std_phyto.values[0,:,np.newaxis]
    
def generate_phytoplankton_traits(r_spec = 1, n_com = 100,
                                  std = None,
                                  corr = None,
                                  const_traits = {}, tradeoffs = {}):
    
    if std is None:
        std = std_phyto.copy()
    if corr is None:
        corr = corr_phyto.copy()
    
    # change variation of certain traits
    for key in const_traits.keys():
        std[key] = const_traits[key]
    
    # change tradeoffs between traits
    for key in tradeoffs.keys():
        key1, key2 = key.split(":")
        corr.loc[key1, key2] = tradeoffs[key]
        corr.loc[key2, key1] = tradeoffs[key]    
    
    # combine corrlation and standard deviation into covariance matrix
    A_phyto = corr*std.values*std.values[0,:,np.newaxis]
    
    # covariance matrix for meta community
    A_meta_com = cov_phyto - A_phyto
    
    # generate multivariate distribution
    # mean trait values per community
    com_mean = np.random.multivariate_normal(mean_phyto.values[0],
                                      A_meta_com, (n_com, 1))
    # variation around community mean
    species_differences = np.random.multivariate_normal(
                                        np.zeros(len(A_meta_com)),
                                      A_phyto, (n_com, r_spec))
    
    # traits are community man + species differences
    traits = np.exp(com_mean + species_differences)
    
    # order species according to their size
    order = np.argsort(traits[...,0], axis = 1)
    trait_dict = {}
    for i, trait in enumerate(phyto_traits):
        trait_dict[trait] = traits[np.arange(n_com)[:,np.newaxis], order,i]
    
    return trait_dict

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from generate_plankton import generate_base_traits
    
    traits = generate_phytoplankton_traits(10,100)
    traits = generate_base_traits(10,1000)
    traits = {key: np.log(traits[key].flatten()) for key in traits.keys()}
    traits = pd.DataFrame(traits, columns = phyto_traits)
    
    fig, ax = plt.subplots(len(traits.keys()), len(traits.keys()),
                               figsize = (12,12), sharex = "col", sharey = "row")
    
    name = ["Size", "Growth rate\n$\mu_P$", "Half-\nsaturation N\n$k_n$",
            "Half-\nsaturation P\n$k_p$", "Half-\nsaturation\nlight\n$k_l$",
            "N uptake\n$c_n$", "P uptake\n$c_p$", "Absorption\n$a$",
            "Edibility\n$e_P$", "Resource\ncontent\n$R_P$"]
    bins = 10
    for i,keyi in enumerate(phyto_traits):
        for j, keyj in enumerate(phyto_traits):               
            if i<j:
                
                ax[j,i].scatter(traits[keyi][:300], traits[keyj][:300], s = 5,
                            alpha = 0.1, color = "blue")
                
                ax[j,i].scatter(raw_data[keyi], raw_data[keyj],
                                    s = 3, color = "orange")
        
            else:
                ax[j,i].set_frame_on(False)
                ax[j,i].tick_params(axis = "y", colors = "None")
                ax[j,i].tick_params(axis = "x", colors = "None")
        
        # plot histogram

        ax_hist = fig.add_subplot(len(traits.keys()),
                                  len(traits.keys()),
                                  1 + i + i*len(traits.keys()))
        ax_hist.hist(traits[keyi], bins, density = True, color = "blue")
        ax_hist.set_xticklabels([])
        ax_hist.set_yticklabels([])
        ax_hist.set_title(name[i])
        ax_hist.hist(raw_data[keyi], bins, density = True,
                     alpha = 0.5, color = "orange")
            
        ax[-1,-1].set_xlim(ax[-1,0].get_ylim())
        
        ax[i,0].set_ylabel(name[i], rotation = 0,
                           ha = "right", va = "center")
        ax[-1,i].set_xticks(np.round(np.nanpercentile(traits[keyi],[5,95]),1))
        ax[i,0].set_yticks(np.round(np.nanpercentile(traits[keyi],[5,95]),1))

    fig.savefig("Figure_phytoplankton_traits.pdf")
