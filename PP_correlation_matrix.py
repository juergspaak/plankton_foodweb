import matplotlib.pyplot as plt
plt.style.use('dark_background')
import pandas as pd
import numpy as np

import generate_plankton as gp
from matplotlib import colors

def plot(save_id = [0]):
    fig.savefig("PP_slides/PP_correlation_matrix_{}.png".format(save_id[0]))
    save_id[0] += 1

# prepare data
data = {}
responses = {"phyto": [], "zoo": []}
for trait in ["phyto", "zoo"]:
    for response  in ["phyto", "zoo"]:
        data[trait + "_" + response] = pd.read_csv("data/simplified_corr_{}_traits_{}.csv".format(trait, response), 
                                                   index_col = 0)
        responses[response].extend(data[trait + "_" + response].values.flatten())


ref = pd.read_csv("data/assembly_reference.csv").values
sig = 2.5
mean_ref = np.nanmean(ref, axis = 0)
std_ref = np.nanstd(ref, axis = 0)
bounds = mean_ref + sig*std_ref*[[-1],[1]]

fig = plt.figure(figsize = (9,9))
ax = np.empty((2,2), dtype ="object")

ylabels = [["Nutritional\nvalues", "edibility", "absorption", "Uptake P",
           "Uptake N", "Halfsat. L", "Halfsat. P", "Halfsat. N",
           "Intrinsic\ngrowth"][::-1],
            ["Halfsaturation", "Mortality", "Clearance", "Intrinsic\ngrowth"][::-1]]
 
for i, trait in enumerate(["phyto", "zoo"]):
    for j, response in enumerate(["phyto", "zoo"]):
        ax[i,j] = fig.add_subplot(2,2,1+i*2+j)
        if i + j == 0:
            ax[i,j].set_title("Phytoplankton richness")
            ax[i,j].set_ylabel("Phytoplankton Correlations")
        if (i == 0) & (j == 1):
            ax[i,j].set_title("Zooplankton richness")
        if (i == 1) & (j == 0):
            ax[i,j].set_ylabel("Zooplankton traits")
        divnorm=colors.TwoSlopeNorm(vmin=np.nanmin(responses[response]),
                            vcenter=mean_ref[j],
                            vmax=np.nanmax(responses[response]))
        cmap = ax[i,j].imshow(data[trait + "_" + response], aspect = "auto",
                                cmap = "RdBu_r", norm = divnorm,
                                origin = "lower")
        
        # highlight the ones with outside of probability dots
        loc = np.where(data[trait + "_" + response] < bounds[0,j])
        ax[i, j].scatter(loc[1], loc[0], s = 25, color = "purple")
        loc = np.where(data[trait + "_" + response] > bounds[1,j])
        ax[i, j].scatter(loc[1], loc[0], s = 25, color = "purple")

        ax[i,j].set_xticks(np.arange(len(data[trait + "_" + response])))
        ax[i,j].set_yticks(np.arange(len(data[trait + "_" + response])))
        
        ax[i,j].set_xticklabels(data[trait + "_" + response].columns, rotation = 90)
        
        if j == 0:
            ax[i,j].set_yticklabels(ylabels[i])
        else:
            ax[i,j].set_yticklabels("")
        fig.colorbar(cmap, ax = ax[i, j])
        plot()
#"""