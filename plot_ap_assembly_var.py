import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from string import ascii_uppercase as ABC
from layouts import labels, color, ax_id

data = pd.read_csv("data/assembly_var.csv")
change_trait = np.sort(list(set(data["change"])))
traits = set(data["trait"])

fig, ax = plt.subplots(2,3, figsize = (13,13), sharey = "row", sharex = True)

ax[0,0].semilogx()
ax[0,0].set_xticks([0.5, 1, 2])
ax[0,0].set_xticklabels([0.5, 1, 2])

ax[0,0].set_xticks([], minor = True)

for i, trait in enumerate(sorted(set(data["trait"]))):        
    # sort traits by resource traits, growth traits, species interactoins
    ind = data["trait"] == trait
    ax[0,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_phyto"],
                 '.-', color = color[trait], label = labels[trait],
                 lw = 1)
    ax[1,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_zoo"],
                 '.-', color = color[trait], label = labels[trait], lw = 1)
    
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
        ax[0,j].legend(fontsize = 14, loc = "lower left")
        
        ax[-1, j].set_xlabel("Change in variance")
        

# add layout
ax[0,0].set_ylabel("Phytoplankton\nspecies richness")
ax[1,0].set_ylabel("Zooplankton\nspecies richness")

ax[0,0].set_title("Growth traits")
ax[0,1].set_title("Resource traits")
ax[0,2].set_title("Species interaction\ntraits")

for i, a in enumerate(ax.flatten()):
    a.set_title("ABCDEF"[i], loc = "left")
    a.axvline(1, color = "k")
    
fig.savefig("Figure_ap_assembly_var.pdf")
fig.savefig("PP_slides/Change_in_variance.png")