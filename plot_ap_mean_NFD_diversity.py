import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from string import ascii_uppercase as ABC
from layouts import labels, color, ax_id

data = pd.read_csv("data/assembly_mean.csv")
change_trait = np.sort(list(set(data["change"])))
traits = set(data["trait"])

fig, ax = plt.subplots(3,3, figsize = (13,13))

for i, trait in enumerate(sorted(set(data["trait"]))):        
    # sort traits by resource traits, growth traits, species interactoins
    ind = data["trait"] == trait
    ax[0,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_phyto"],
                 '.-', color = color[trait], label = labels[trait],
                 lw = 3)
    ax[1,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_zoo"],
                 '.-', color = color[trait], label = labels[trait], lw = 2)
    
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
        
##############################################################################
# NFD data
    
mean_traits = ["R_P", "k_Z", "h_zp", "e_P", "c_Z"]
NFD_mean = pd.read_csv("data/NFD_mean.csv")

for tr in mean_traits:
    ind = NFD_mean.trait == tr
    ax[2,1].plot(NFD_mean.loc[ind, "change"],
                  NFD_mean.loc[ind, "ND_mean_1"], '.-', color=color[tr],
                  label=labels[tr], lw = 2)
    ax[2,2].plot(NFD_mean.loc[ind, "change"],
                  NFD_mean.loc[ind, "FD_median_2"], '.-', color=color[tr]
                  , lw = 2)
    
    ind2 = data["trait"] == tr

ref = pd.read_csv("data/NFD_ref.csv")
sig = 2
mean_ref = np.nanmean(ref, axis=0)
std_ref = np.nanstd(ref, axis=0)
bounds = mean_ref + sig * std_ref * [[-1], [0], [1]]
# add reference values
ax[2, 1].axhline(bounds[0, 0], color="red", linestyle="--")
ax[2, 1].axhline(bounds[1, 0], color="red", linestyle="-")
ax[2, 1].axhline(bounds[2, 0], color="red", linestyle="--")


ax[2, 2].axhline(bounds[0, 7], color="red", linestyle="--")
ax[2, 2].axhline(bounds[1, 7], color="red", linestyle="-")
ax[2, 2].axhline(bounds[2, 7], color="red", linestyle="--")


ax[0,0].set_ylim([1, 2.6])



##############################################################################
# add layout
ax[-1,0].set_xlim(change_trait[[0,-1]]+[-0.1,0.1])
ax[1,0].set_xticks(change_trait[[0,len(change_trait)//2,-1]])
ax[1,1].set_xticks(change_trait[[0,len(change_trait)//2,-1]])
ax[1,0].set_xticklabels(change_trait[[0,len(change_trait)//2,-1]])
ax[1,1].set_xticklabels(change_trait[[0,len(change_trait)//2,-1]])

for i,a in enumerate(ax.flatten()):
    if i == 6:
        continue
    a.set_xticks([-1,0,1])
    a.set_xticklabels(3*[""])
    a.set_title(ABC[i], loc = "left")
    a.axvline(change_trait[len(change_trait)//2], color = "k")
    
for a in ax[[1,2,2], np.arange(3)]:
    a.set_xticklabels([-1,0,1])
    
for a in ax[:2].flatten():
    a.set_ylim([1,2.6])
    a.set_yticks([1,1.5,2, 2.5])
    a.set_yticklabels(len(a.get_yticks())*[""])
    
ax[0,0].set_yticklabels([1,1.5,2,2.5])
ax[1,0].set_yticklabels([1,1.5,2,2.5])

ax[2,0].set_xticks([])
ax[2,0].set_yticks([])
ax[2,0].set_frame_on(False)

# titles
fs = 16
ax[0,0].set_title("Growth traits", fontsize = fs)
ax[0,1].set_title("Resource traits", fontsize = fs)
ax[0,2].set_title("Species interaction\ntraits", fontsize = fs)
ax[2,1].set_title("Species interaction\ntraits", fontsize = fs)
ax[2,2].set_title("Species interaction\ntraits", fontsize = fs)


ax[1,0].set_xlabel("Change in Mean", fontsize = fs)
ax[-1,1].set_xlabel("Change in Mean", fontsize = fs)
ax[-1,2].set_xlabel("Change in Mean", fontsize = fs)        

ax[0,0].set_ylabel("Phytoplankton\nRichness", fontsize = fs)
ax[1,0].set_ylabel("Zooplankton\nRichness", fontsize = fs)
ax[2,1].set_ylabel("Niche differences", fontsize = fs)
ax[2,2].set_ylim([0,.2])
ax[2,2].set_yticks([0,.1,.2])
ax[2,1].set_ylim([-1.25,0])
ax[2,1].set_yticks([-1,-.5,0])
ax[2,2].set_ylabel("Fitness differecnes", fontsize = fs)
    
fig.tight_layout()
fig.savefig("Figure_ap_mean_NFD_diversity.pdf")