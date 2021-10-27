import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import generate_plankton as gp

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

for i, trait in enumerate(set(data["trait"])):        
    # sort traits by resource traits, growth traits, species interactoins
    if trait in ["c_n", "c_p", "a", "k_n", "k_p", "k_l"]:
        k = 1
    elif trait in ["mu_Z", "mu_P", "m_Z"]:
        k = 0
    elif trait in ["h_zp", "s_zp", "c_Z", "k_Z", "R_P", "e_P"]:
        k = 2
    else:
        continue
    ind = data["trait"] == trait
    ax[0,k].plot(data.loc[ind, "change"], data.loc[ind, "richness_phyto"],
                 '.-', label = trait)
    ax[1,k].plot(data.loc[ind, "change"], data.loc[ind, "richness_zoo"],
                 '.-', label = trait)
    

ind_ref = data["change"] == 0
perc = np.nanpercentile(data.loc[ind_ref, ["richness_phyto", "richness_zoo"]].values,
                        [25,50,75], axis = 0).T
ref = data.loc[ind_ref, ["richness_phyto", "richness_zoo"]].values
sig = 2.5
mean_ref = np.nanmean(ref, axis = 0)
std_ref = np.nanstd(ref, axis = 0)
bounds = mean_ref + sig*std_ref*[[-1],[1]]
# add reference data 
for i in range(2):
    for j in range(3):
        ax[i,j].axhline(bounds[0,i], color = "red", linestyle = "--")
        ax[i,j].axhline(bounds[1,i], color = "red", linestyle = "--")
        ax[i,j].axhline(perc[i, 1], color = "red")
        ax[i,j].axvline(change_trait[len(change_trait)//2], color = "k")
        ax[0,j].legend(ncol = 2)
        

ax[0,0].set_ylabel("Phytoplankton\nRichness")
ax[1,0].set_ylabel("Zooplankton\nRichness")  
    
fig.savefig("Figure_assembly_mean.pdf")