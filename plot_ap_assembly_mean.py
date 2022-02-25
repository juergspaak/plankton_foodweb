import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import generate_plankton as gp
from layouts import labels, color, ax_id

data = pd.read_csv("data/assembly_mean.csv")
change_trait = np.sort(list(set(data["change"])))
traits = set(data["trait"])

fig, ax = plt.subplots(2,3, sharex = True, sharey = True, figsize = (13,13))

ax[0,0].set_ylim([1, 2.6])
# add layout
ax[-1,0].set_xlim(change_trait[[0,-1]]+[-0.1,0.1])
ax[1,0].set_xticks(change_trait[[0,len(change_trait)//2,-1]])
ax[1,1].set_xticks(change_trait[[0,len(change_trait)//2,-1]])
ax[1,0].set_xticklabels(change_trait[[0,len(change_trait)//2,-1]])
ax[1,1].set_xticklabels(change_trait[[0,len(change_trait)//2,-1]])

ax[0,0].set_title("Growth traits")
ax[0,1].set_title("Resource traits")
ax[0,2].set_title("Species interaction\ntraits")

ax[-1,0].set_xlabel("Change in Mean")
ax[-1,1].set_xlabel("Change in Mean")
ax[-1,2].set_xlabel("Change in Mean")

for i,a in enumerate(ax.flatten()):
    a.set_title("ABCDEF"[i], loc = "left")

for i, trait in enumerate(sorted(set(data["trait"]))):        
    # sort traits by resource traits, growth traits, species interactoins
    ind = data["trait"] == trait
    ax[0,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_phyto"],
                 '.-', color = color[trait], label = labels[trait],
                 lw = 3)
    ax[1,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_zoo"],
                 '.-', color = color[trait], label = labels[trait], lw = 3)
    



ref = pd.read_csv("data/assembly_reference.csv").values
sig = 2.5
mean_ref = np.nanmean(ref, axis = 0)
std_ref = np.nanstd(ref, axis = 0)
bounds = mean_ref + sig*std_ref*[[-1],[1]]
# add reference data 
for i in range(2):
    for j in range(3):
        ax[i,j].axhline(bounds[0,i], color = "red", linestyle = "--")
        ax[i,j].axhline(bounds[1,i], color = "red", linestyle = "--")
        ax[i,j].axhline(mean_ref[i], color = "red")
        ax[i,j].axvline(change_trait[len(change_trait)//2], color = "k")
        ax[0,j].legend(fontsize = 14)
        

ax[0,0].set_ylabel("Phytoplankton\nRichness")
ax[1,0].set_ylabel("Zooplankton\nRichness")  
    
fig.savefig("Figure_assembly_mean.pdf")