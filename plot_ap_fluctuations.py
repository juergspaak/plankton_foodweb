import numpy as np
import matplotlib.pyplot as plt


path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/fluctuations/"

###############################################################################
# plot results of each environmental factor fluctuating on its own

for pred in ["", "_no_pred"]:
    fig, ax = plt.subplots(2,5, sharex = True, sharey = "row", figsize = (13,9))
    
    file = np.load("no_fluct.npz")
    years = np.arange(20)
    for i, key in enumerate(["N", "P", "I_in", "d", "zm"]):
        for period in [1,10,25,50,100]:
            try:
                file = np.load(path + "data_fluctuations{}_{}_{}.npz".format(pred, key, period))
                dens = file["dens"]
            except (FileNotFoundError, KeyError):
                continue
            dens = file["dens"]
            rich = np.mean(np.sum(dens>0, axis = 2), axis = 0)
            
            ax[0,i].plot(years, rich[0], label = period)
            ax[1,i].plot(years, rich[1], label = period)
            ax[0,i].set_title(key)
            ax[1,i].set_xlabel("Year")
            
        
            ax[0,i].legend()
            ax[1,i].legend()
    ax[0,0].set_ylabel("Phytoplankton richness")
    ax[1,0].set_ylabel("Zooplankton richness")
    
    fig.tight_layout()
    fig.savefig("Figure_in_fluctuations{}.pdf".format(pred))

##############################################################################
    # N and P combined flucutations with different phase differences
    fig, ax = plt.subplots(2,1, sharex = True, sharey = False)
    
    years = np.arange(20)
    for key in ["no_fluct", "in_phase", "quarter_phase", "off_phase"]:
        try:
            file = np.load(path  + "data_fluctuations_N_P{}_{}.npz".format(pred, key))
            dens = file["dens"]
        except (FileNotFoundError, KeyError):
                continue
        rich = np.mean(np.sum(dens>0, axis = 2), axis = 0)
        
        ax[0].plot(years, rich[0], label = key)
        ax[1].plot(years, rich[1], label = key)
    ax[1].set_xlabel("Year")
    ax[0].set_ylabel("Phytoplankton richness")
    ax[1].set_ylabel("Zooplankton richness")
    ax[0].legend()
    
    fig.savefig("Figure_in_fluctuations{}_2.pdf".format(pred))


file = np.load(path + "data_fluctuations_no_pred_{}_{}.npz".format("N", 100))
file2 = np.load(path + "data_fluctuations_no_pred_{}_{}.npz".format("N", 10))
file = np.load(path + "data_fluctuations_N_P_no_pred_{}.npz".format("in_phase"))
file2 = np.load(path + "data_fluctuations_N_P_no_pred_{}.npz".format("quarter_phase"))

surv = file["dens"][:,0]>0
surv2 = file2["dens"][:,0]>0
    
plt.figure()
plt.plot(years, np.mean(np.any(surv != surv2, axis = 2), axis = 0))
plt.axhline(0, color = "k")
