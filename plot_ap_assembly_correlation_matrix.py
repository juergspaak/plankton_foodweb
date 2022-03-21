import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import colors

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

fig, ax = plt.subplots(3,2, figsize = (9,12),
                       sharex = "row", sharey = "row")


for i, trait in enumerate(["phyto", "zoo"]):
    bins = np.linspace(np.nanmin(responses[trait]),
                       np.nanmax(responses[trait]), 35)
    ax[0,i].hist(responses[trait], bins = bins, density = True)
    #ax[0,i].hist(ref[trait], bins = bins, alpha = .5, density = True)
    ax[0,i].axvline(bounds[0,i], color = "r", linestyle = "--") 
    ax[0,i].axvline(bounds[1,i], color = "r", linestyle = "--")

ax[0,0].set_xlabel("Frequency")
ax[0,0].set_xlabel("Phytoplankton\nrichness")
ax[0,1].set_xlabel("Zooplankton\nrichness")
    
for i, trait in enumerate(["phyto", "zoo"]):
    for j, response in enumerate(["phyto", "zoo"]):
        divnorm=colors.TwoSlopeNorm(vmin=np.nanmin(responses[response]),
                            vcenter=mean_ref[j],
                            vmax=np.nanmax(responses[response]))
        cmap = ax[i+1,j].imshow(data[trait + "_" + response], aspect = "auto",
                                cmap = "RdBu_r", norm = divnorm,
                                origin = "lower")
        
        # highlight the ones with outside of probability dots
        loc = np.where(data[trait + "_" + response] < bounds[0,j])
        ax[i+1, j].scatter(loc[1], loc[0], s = 25, color = "purple")
        loc = np.where(data[trait + "_" + response] > bounds[1,j])
        ax[i+1, j].scatter(loc[1], loc[0], s = 25, color = "purple")

        ax[i+1,j].set_xticks(np.arange(len(data[trait + "_" + response])))
        ax[i+1,j].set_yticks(np.arange(len(data[trait + "_" + response])))
        
        ax[i+1,j].set_xticklabels(data[trait + "_" + response].columns, rotation = 90)
        ax[i+1,j].set_yticklabels(data[trait + "_" + response].columns)
        fig.colorbar(cmap, ax = ax[i+1, j])
        
for i, a in enumerate(ax.flatten()):
    a.set_title("ABCDEF"[i], loc = "left")
fig.tight_layout()
fig.savefig("Figure_ap_assembly_correlation_matrix.pdf")
#"""

###############################################################################
# are there tradeoffs with non-monotone effect?

# c_n:mu_P is a weirdly behaving tradeoff, not sure why, rest behaves normally
for trait in ["phyto", "zoo"]:
    for j, response in enumerate(["phyto", "zoo"]):
        df = data[trait + "_" + response].values
        interest = (df < bounds[0,j]) & (df.T < bounds[0,j])
        print(trait, response, np.sum(interest))

        interest = (df > bounds[1,j]) & (df.T > bounds[1,j])
        print(trait, response, np.sum(interest))
        
# which tradeoffs are the strongest?
response = pd.DataFrame(np.nan, index = np.arange(data["zoo_zoo"].size + data["phyto_zoo"].size),
                        columns = ["tradeoff", "richness_phyto", "richness_zoo"])
counter = 0
for trait in ["phyto", "zoo"]:
    for traiti in data[trait + "_zoo"].columns:
        for traitj in data[trait + "_zoo"].columns:
            traits_sort = sorted([traiti, traitj])
            response.loc[counter] = [traits_sort[0] + ":" + traits_sort[1],
                                     data[trait + "_phyto"].loc[traiti, traitj],
                                     data[trait + "_zoo"].loc[traiti, traitj]]
            counter += 1
            
# remove self-tradeoffs
response = response[np.isfinite(response.richness_phyto)]
response = response.reset_index(drop = True)

# find the most important x tradeoffs
n_interest = 7
ind = np.append(np.arange(n_interest), np.arange(-n_interest, 0,1))
tradeoffs = response.loc[np.argsort(response.richness_phyto).values[ind], "tradeoff"]
tradeoffs2 = response.loc[np.argsort(response.richness_zoo).values[ind], "tradeoff"]

important_tradeoffs = set(list(tradeoffs) + list(tradeoffs2))