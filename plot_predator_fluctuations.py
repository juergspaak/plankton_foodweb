import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/fluctuations/"
save = path + "data_predator_fluctuations.npz"
file = np.load(save)
cases = file["cases"]
r_is = file["r_is"]
fluctuations = file["fluctuations"]
diff = np.empty((2, len(cases), len(cases)))

for i in range(len(cases)):
    for j in range(len(cases)):
        diff[:, i,j] = np.sum(np.sign(r_is[:,i]) != np.sign(r_is[:,j]), axis = 0)/file["n_coms"]
        
fig, ax = plt.subplots(3,2, figsize = (9,9), sharey = "row")

cmap = ax[0,0].imshow(diff[0], vmin = 0, vmax = 0.03, origin = "lower")
fig.colorbar(cmap, ax = ax[0,0])

cmap = ax[0,1].imshow(diff[1], vmin = 0, vmax = 0.03, origin = "lower")
fig.colorbar(cmap, ax = ax[0,1])

c = cases == 1
ticks = ["".join(np.array(["R", "P", "Z"])[i]) for i in c]

ax[0,0].set_xticks(np.arange(len(cases)))
ax[0,0].set_xticklabels(ticks)
ax[0,0].set_yticks(np.arange(len(cases)))
ax[0,0].set_yticklabels(ticks)

ax[0,1].set_xticks(np.arange(len(cases)))
ax[0,1].set_xticklabels(ticks)
ax[0,1].set_yticks(np.arange(len(cases)))
ax[0,1].set_yticklabels(ticks)

ax[0,0].set_title("Phytoplankton")
ax[0,0].set_ylabel("Difference in species richness")
ax[0,0].set_xlabel("Constant factors")
ax[0,1].set_xlabel("Constant factors")
ax[0,1].set_title("Zooplankton")

##############################################################################
# quantitative differences in invasion growth rates

bins = np.linspace(-.1, .1, 51)
bins = np.concatenate([[-10,], bins, [10,]])
ax[1,0].hist((r_is[:,-1,0] - r_is[:,0,0])/r_is[:,0,0], bins = bins
             , label = "Phytoplankton")
ax[1,0].hist((r_is[:,-1,1] - r_is[:,0,1])/r_is[:,0,1], bins = bins, alpha = 0.5,
             label = "Zooplankton")
ax[1,0].set_xlim([-0.11, 0.11])

bins = np.linspace(-1, 1, 51)
bins = np.concatenate([[-10,], bins, [10,]])
ax[1,1].hist((r_is[:,-1,0] - r_is[:,0,0])/r_is[:,0,0], bins = bins
             , label = "Phytoplankton")
ax[1,1].hist((r_is[:,-1,1] - r_is[:,0,1])/r_is[:,0,1], bins = bins, alpha = 0.5,
             label = "Zooplankton")
ax[1,1].set_xlim([-1.1, 1.1])

ax[1,0].legend()

ax[1,0].set_xlabel("Differences in invasion growth rate")
ax[1,0].set_ylabel("Frequency")
ax[1,1].set_xlabel("Differences in invasion growht rate")

##############################################################################
ax[2,0].scatter(fluctuations[:,0], np.abs((r_is[:,4,0] - r_is[:,0,0])/r_is[:,0,0]), s = 2)
ax[2,0].set_ylim([0,0.5])
ax[2,0].set_xlim([0,2])

ax[2,1].scatter(fluctuations[:,1], np.abs((r_is[:,2,1] - r_is[:,0,1])/r_is[:,0,1]), s = 2)
ax[2,1].set_ylim([0,0.5])
ax[2,1].set_xlim([0,10])

ax[2,0].set_ylabel("Differences in invasion growth rate")
ax[2,0].set_xlabel("Strength of resource fluctuations")
ax[2,1].set_xlabel("Strength of phytoplankton fluctuations")

fig.tight_layout()

fig.savefig("Figure_ap_internal_fluctuatins.pdf")