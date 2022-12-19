import numpy as np
import matplotlib.pyplot as plt


path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/fluctuations/"

###############################################################################
# plot results of each environmental factor fluctuating on its own

fig, ax = plt.subplots(2,2,sharex = True, sharey = True, figsize = (9,9))
title = ["With predators", "Without predators"]
for i, pred in enumerate(["", "_no_pred"]):
##############################################################################
    # N and P combined flucutations with different phase differences
    ax[0,i].set_title(title[i])
    for key in ["no_fluct", "in_phase", "quarter_phase", "off_phase"]:
        try:
            file = np.load(path  + "data_fluctuations_temp_richness{}_N_P_{}.npz".format(pred, key))
            dens = file["dens"]
        except (FileNotFoundError, KeyError):
                continue
        rich = np.mean(np.sum(dens>0, axis = 2), axis = 0)
        
        years = np.arange(file["size_P"].shape[1])
        
        ax[0,i].plot(years, rich[0], label = key)
        ax[1,i].plot(years, rich[1], label = key)
    ax[1,i].set_xlabel("Year")
    ax[0,i].set_ylabel("Phytoplankton richness")
    ax[1,i].set_ylabel("Zooplankton richness")
    ax[0,i].legend()

fig.savefig("Figure_ap_temporal_richness1.pdf")
# combined richness of the past n years
fig, ax = plt.subplots(2,2,sharex = True, sharey = True, figsize = (9,9))

count_years = 5

for i, pred in enumerate(["", "_no_pred"]):
##############################################################################
    # N and P combined flucutations with different phase differences
    ax[0,i].set_title(title[i])
    for key in ["no_fluct", "in_phase", "quarter_phase", "off_phase"]:
        try:
            file = np.load(path  + "data_fluctuations_temp_richness{}_N_P_{}.npz".format(pred, key))
            dens = file["dens"]
        except (FileNotFoundError, KeyError):
                continue
        dens[np.isnan(dens)] = 0
        # count all species which where present within the last year
        cum_dens = np.cumsum(dens, axis = -1)
        cum_dens = cum_dens[...,count_years:] - cum_dens[...,:-count_years]
        rich = np.mean(np.sum(cum_dens>0, axis = 2), axis = 0)
        
        years = np.arange(count_years, file["size_P"].shape[1])
        
        ax[0,i].plot(years, rich[0], label = key)
        ax[1,i].plot(years, rich[1], label = key)
    ax[1,i].set_xlabel("Year")
    ax[0,i].set_ylabel("Phytoplankton richness")
    ax[1,i].set_ylabel("Zooplankton richness")
    ax[0,i].legend()


fig.savefig("Figure_ap_temporal_richness2.pdf")
