import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde, linregress

# data from outside this temperature range is rejected
temp_range = [15, 25]
"""
phytoplankton traits are taken from
http://www.esapubs.org/archive/ecol/E096/202/
and 
https://aslopubs.onlinelibrary.wiley.com/doi/epdf/10.1002/lno.10282
"""
phyto_data = pd.read_csv("Phytoplankton traits.csv")
phyto_data = phyto_data[(phyto_data.temperature <= temp_range[1]) &
                        (phyto_data.temperature >= temp_range[0])]
phyto_traits = phyto_data[["mu_p", "k_p_m", "vmax_p",
                           "mu_nit", "k_nit_m", "vmax_nit"]].copy()
phyto_traits.columns = ["mu_p", # maximal phosphor growth rate [day^-1]
                "k_p", # half saturation for phosphor growth rate [mumol L^-1]
                "c_p", # phosphor consumption rate [mumol P cell^-1 day^-1]
                "mu_n", # maximal nitrate growth rate [day^-1]
                "k_n", # half saturation for nitrate growth rate [mumol L^-1]
                "c_n"] # nitrate uptake rate [mumol N cell^-1 day^-1]

# remove outliers
for key in phyto_traits.keys():
    perc = np.nanpercentile(phyto_traits[key], [25,75])
    iqr = perc[1]-perc[0]
    ind = ((phyto_traits[key] > perc[0] - 1.5*iqr) &
            (phyto_traits[key] < perc[1] + 1.5*iqr))
    phyto_traits.loc[~ind, key] = np.nan

#light traits
light_data = pd.read_csv("Light_traits.csv")
light_data = light_data[["mu_l",# maximum light growth rate [day^-1]
    "alpha",# initial slope of light grtowth [quanta^-1 mumol photon^-1 m^2s^1]
   "I_o"]].copy()# optimal light intensity [mumol photons m^-2s^-1]

# remove outliers
for key in light_data.keys():
    perc = np.nanpercentile(light_data[key], [25,75])
    iqr = perc[1]-perc[0]
    ind = ((light_data[key] > perc[0] - 1.5*iqr) &
            (light_data[key] < perc[1] + 1.5*iqr))
    light_data[key][~ind] = np.nan
                  
labels = [r"$w_{ip}$  [$day^{-1}$]",
          r"$k_{ip}$  [$\mu$mol P $L^{-1}$]",
          r"$c_{ip}$  [$cell^{-1}day^{-1}$]",
          r"$w_{in}$  [$day^{-1}$]",
          r"$k_{in}$  [$\mu$mol N $L^{-1}$]",
          r"$c_{in}$  [$cell^{-1}day^{-1}$]"]

gen_data = {}
for key in phyto_traits.keys():
    gen_data[key] = gaussian_kde(
                        phyto_traits[key][np.isfinite(phyto_traits[key])])
for key in light_data.keys():
    gen_data[key] = gaussian_kde(
                        light_data[key][np.isfinite(light_data[key])])

if __name__ == "__main__":
    alpha = 0.5
    s = 5
    n_coms = 1000
    
    ###########################################################################
    
    # resource uptake traits
    fig, ax = plt.subplots(3,2, figsize = (7,9))
    ax_f = ax.T.flatten()
    for i,key in enumerate(phyto_traits.keys()):
        traits = gaussian_kde(
                        phyto_traits[key][np.isfinite(phyto_traits[key])])
        traits = traits.resample(10*n_coms)
        # remove negative values
        traits = traits[traits>0][:2*n_coms].reshape(2,n_coms)
        ax_f[i].hist(traits.flatten(), bins = 50, density = True,
            stacked = True)
        ax_f[i].set_xlabel(labels[i], fontsize = 16)
        ax_f[i].set_ylabel("counts", fontsize = 16)
        ax_f[i].set_title("ABCDEF"[i], loc = "left")
        ax_f[i].hist(phyto_traits[key], bins = 50, alpha = 0.5, density = True)
        
    
        
    ax[0,0].set_title("Phosphorus", loc = "center", fontsize = 16)
    ax[0,1].set_title("Nitrogen", loc = "center", fontsize = 16)
    fig.tight_layout()
    fig.savefig("Figure_ap_trait_distribution.pdf")
    
    ###########################################################################
    # correlation of resource uptake traits
    
    fig, ax = plt.subplots(len(phyto_traits.keys()), len(phyto_traits.keys()),
                           figsize = (9,9), sharex = "col", sharey = "row")
    for i, keyi in enumerate(phyto_traits.keys()):
        dati = phyto_traits[keyi]
        lim = np.nanpercentile(dati,[0,100])
        lim = lim + np.array([-.05, .05]) * (lim[1]-lim[0])
        ax[i,i].set_xlim(lim)
        ax[i,i].set_ylim(lim)
        ax[i,i].text(np.mean(lim),np.mean(lim),
                      keyi, ha = "center", va = "center", fontsize = 18)
        for j, keyj in enumerate(phyto_traits.keys()):
            if j<i:
                datj = phyto_traits[keyj]
                ax[i,j].scatter(datj, dati)
                ind = np.isfinite(dati*datj)
                if np.sum(ind) > 5:
                    res = linregress(datj[ind], dati[ind])
                    ax[i,j].plot(ax[i,j].get_xlim(),
                      res[1] + res[0] * np.array(ax[i,j].get_xlim()), "r-")
                    x_cord = np.mean(ax[j,i].get_xlim())
                    ycord = np.linspace(*ax[j,i].get_ylim(),6)
                    ax[j,i].text(x_cord, ycord[1],
                      "n_points={}".format(np.sum(ind)),
                      ha = "center", va = "center")
                    ax[j,i].text(x_cord, ycord[2],
                      "slope={}".format(np.round(res[0],2)),
                      ha = "center", va = "center")
                    ax[j,i].text(x_cord, ycord[3],
                      "R2={}".format(np.round(res[2]**2,2)),
                      ha = "center", va = "center")
                    ax[j,i].text(x_cord, ycord[4],
                      "p={}".format(np.round(res[3],3)),
                      ha = "center", va = "center")
    # mu and c are positively correlated
    fig.savefig("Figure_ap_trait_correlation.pdf")
    
    ###########################################################################
    # light tratis
    fig, ax = plt.subplots(1,3, figsize = (7,9))
    ax_f = ax.T.flatten()
    for i,key in enumerate(light_data.keys()):
        traits = gaussian_kde(
                        light_data[key][np.isfinite(light_data[key])])
        traits = traits.resample(10*n_coms)
        # remove negative values
        traits = traits[traits>0][:2*n_coms].reshape(2,n_coms)
        ax_f[i].hist(traits.flatten(), bins = 50, density = True,
            stacked = True)
        ax_f[i].set_xlabel(key, fontsize = 16)
        ax_f[i].set_ylabel("counts", fontsize = 16)
        ax_f[i].set_title("ABCDEF"[i], loc = "left")
        ax_f[i].hist(light_data[key], density = True, bins = 50, alpha = 0.5)
    
    fig.tight_layout()
    fig.savefig("Figure_ap_trait_distribution_light.pdf")
    
    ###########################################################################
    # light correlation between traits
    fig, ax = plt.subplots(len(light_data.keys()), len(light_data.keys()),
                           figsize = (9,9), sharex = "col", sharey = "row")
    for i, keyi in enumerate(light_data.keys()):
        dati = light_data[keyi]
        lim = np.nanpercentile(dati,[0,100])
        lim = lim + np.array([-.05, .05]) * (lim[1]-lim[0])
        ax[i,i].set_xlim(lim)
        ax[i,i].set_ylim(lim)
        ax[i,i].text(np.mean(lim),np.mean(lim),
                      keyi, ha = "center", va = "center", fontsize = 18)
        for j, keyj in enumerate(light_data.keys()):
            if j<i:
                datj = light_data[keyj]
                ax[i,j].scatter(datj, dati)
                ind = np.isfinite(dati*datj)
                if np.sum(ind) > 5:
                    res = linregress(datj[ind], dati[ind])
                    ax[i,j].plot(ax[i,j].get_xlim(),
                      res[1] + res[0] * np.array(ax[i,j].get_xlim()), "r-")
                    x_cord = np.mean(ax[j,i].get_xlim())
                    ycord = np.linspace(*ax[j,i].get_ylim(),6)
                    ax[j,i].text(x_cord, ycord[1],
                      "n_points={}".format(np.sum(ind)),
                      ha = "center", va = "center")
                    ax[j,i].text(x_cord, ycord[2],
                      "slope={}".format(np.round(res[0],2)),
                      ha = "center", va = "center")
                    ax[j,i].text(x_cord, ycord[3],
                      "R2={}".format(np.round(res[2]**2,2)),
                      ha = "center", va = "center")
                    ax[j,i].text(x_cord, ycord[4],
                      "p={}".format(np.round(res[3],5)),
                      ha = "center", va = "center")
                    print(res[3])
    # mu and c are positively correlated
    
    fig.savefig("Figure_ap_trait_correlation_light.pdf")