import numpy as np
import matplotlib.pyplot as plt


path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/fluctuations/"

# load reference data without any temporal fluctuations
data = np.load("data/assembly_long_60_1000_0.npz")
richness_ref = [np.mean(np.sum(data["dens"]>0, axis = 2), axis = 0)]
years_ref = [np.arange(richness_ref[0].shape[1])]
data = np.load(path + "data_fluctuations_temperature_ref_fast_invasion.npz")
richness_ref.append(np.mean(np.sum(data["dens"]>0, axis = 2), axis = 0))
years_ref.append(np.arange(richness_ref[1].shape[1]))

###############################################################################
# plot results of each environmental factor fluctuating on its own

fig, ax = plt.subplots(2,2, sharex = True, sharey = "row", figsize = (13,9))

for i, pred in enumerate(["", "_no_pred"]):    
    years = np.arange(20)
    ax[0,i].plot(years_ref[i], richness_ref[i][0], '--k', label = "Ref")
    ax[1,i].plot(years_ref[i], richness_ref[i][1], '--k', label = "Ref")
    for period in [5,10,25,50,100]:
        try:
            save = path + "data_fluctuations_temperature_no_pred_{}.npz".format(period)
            file = np.load(path + "data_fluctuations_temperature{}_{}.npz".format(pred, period))
            dens = file["dens"]
        except (FileNotFoundError, KeyError):
            continue
        dens = file["dens"]
        rich = np.mean(np.sum(dens>0, axis = 2), axis = 0)
        
        ax[0,i].plot(years, rich[0], label = period)
        ax[1,i].plot(years, rich[1], label = period)
        ax[1,i].set_xlabel("Year")
        
    
        ax[0,i].legend()
        ax[1,i].legend()
ax[0,0].set_ylabel("Phytoplankton richness")
ax[1,0].set_ylabel("Zooplankton richness")

ax[0,0].set_title("Predators present")
ax[0,1].set_title("Predators absent")
ax[0,0].set_xlim(years[[0,-1]])

fig.tight_layout()
fig.savefig("Figure_ap_temperature_fluctuations.pdf")

###############################################################################
# faster invasion times

fig, ax = plt.subplots(2,2, sharex = True, sharey = "row", figsize = (13,9))

for i, pred in enumerate(["", "_no_pred"]):    
    years = np.arange(30)
    ax[0,i].plot(years_ref[i], richness_ref[i][0], '--k', label = "Ref")
    ax[1,i].plot(years_ref[i], richness_ref[i][1], '--k', label = "Ref")
    for period in [5,10,25,50,100]:
        try:
            file = np.load(path + "data_fluctuations_temperature{}_{}_fast_invasion.npz".format(pred, period))
            dens = file["dens"]
        except (FileNotFoundError, KeyError):
            continue
        dens = file["dens"]
        rich = np.mean(np.sum(dens>0, axis = 2), axis = 0)
        
        ax[0,i].plot(years, rich[0], label = period)
        ax[1,i].plot(years, rich[1], label = period)
        ax[1,i].set_xlabel("Year")
        
    
        ax[0,i].legend()
        ax[1,i].legend()
ax[0,0].set_ylabel("Phytoplankton richness")
ax[1,0].set_ylabel("Zooplankton richness")

ax[0,0].set_title("Predators present")
ax[0,1].set_title("Predators absent")
ax[0,0].set_xlim(years[[0,-1]])

fig.tight_layout()
fig.savefig("Figure_ap_temperature_fluctuations_fast_invasion.pdf")