import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from string import ascii_uppercase as ABC

from layouts import labels, color, ax_id, corrs

data = pd.read_csv("data/assembly_corr.csv")     

fig, ax = plt.subplots(4,4, figsize = (12,12))

### diversity plots

for corr in corrs:
    i = np.arange(len(corrs))[[(corr in co) for co in corrs]][0]
    
    ind = data["tradeoff"] == corr
    traiti, traitj = corr.split(":")
    label = labels[traiti] + " : " + labels[traitj]
    ax[0,ax_id[corr]].plot(data.loc[ind, "corr"],
                           data.loc[ind, "richness_phyto"], '.-',
                     label = label, color = color[corr])
    ax[1,ax_id[corr]].plot(data.loc[ind, "corr"],
                           data.loc[ind, "richness_zoo"], '.-',
                     label = label, color = color[corr])
    
# add reference values

ref = pd.read_csv("data/assembly_reference.csv").values
sig = 2.5
mean_ref = np.nanmean(ref, axis = 0)
std_ref = np.nanstd(ref, axis = 0)
bounds = mean_ref + sig*std_ref*[[-1],[1]]
# add reference data 
for i in range(2):
    for j in range(ax.shape[-1]):
        ax[i,j].axhline(bounds[0,i], color = "red", linestyle = "--")
        ax[i,j].axhline(bounds[1,i], color = "red", linestyle = "--")
        ax[i,j].axhline(mean_ref[i], color = "red")    
        
##############################################################################
# NFD plots
NFD_corr = pd.read_csv("data/NFD_corr.csv")

for corr in list(set(NFD_corr["corr"])):
    ind = NFD_corr["corr"] == corr
    traiti, traitj = corr.split(":")
    label = labels[traiti] + " : " + labels[traitj]
    ax[3, ax_id[corr]].plot(NFD_corr.loc[ind, "change"],
                                NFD_corr.loc[ind, "ND_mean_1"], '.-', label=label,
                                color=color[corr])
    ax[2, ax_id[corr]].plot(NFD_corr.loc[ind, "change"],
                                NFD_corr.loc[ind, "FD_median_2"], '.-', color=color[corr])

ref = pd.read_csv("data/NFD_ref.csv")
sig = 2
mean_ref = np.nanmean(ref, axis=0)
std_ref = np.nanstd(ref, axis=0)
bounds = mean_ref + sig * std_ref * [[-1], [0], [1]]
# add reference values
for i in range(3):
    ax[3, i].axhline(bounds[0, 0], color="red", linestyle="--")
    ax[3, i].axhline(bounds[1, 0], color="red", linestyle="-")
    ax[3, i].axhline(bounds[2, 0], color="red", linestyle="--")
    #ax[3, i].set_yticks([-1, .5, 0])

    ax[2, i].axhline(bounds[0, 7], color="red", linestyle="--")
    ax[2, i].axhline(bounds[1, 7], color="red", linestyle="-")
    ax[2, i].axhline(bounds[2, 7], color="red", linestyle="--")
    #ax[2, i].set_yticks([0, .1, .2])
    

    
##############################################################################
# layout
fs = 16
ax[0,0].set_ylabel("Phytoplankton\nrichness", fontsize = fs)
ax[1,0].set_ylabel("Zooplankton\nrichness", fontsize = fs)
ax[2,0].set_ylabel("Fitness differences", fontsize = fs)
ax[3,0].set_ylabel("Niche differences", fontsize = fs)

ax[0,0].set_title("Gleaner\nopportunist", fontsize = fs)
ax[0,0].legend(ncol = 2)
ax[0,1].set_title("Growth\ndefense", fontsize = fs)
ax[0,1].legend()
ax[0,2].set_title("Quality\nquantitiy", fontsize = fs)
ax[0,2].legend()
ax[0,3].set_title("resource\ntradeoffs", fontsize = fs)
ax[0,3].legend(ncol = 2)


# xlabels
for i, a in enumerate(ax.flatten()):
    a.set_title(ABC[i], loc = "left")
    a.set_xlim([-1,1])
    a.set_xticks([-1,0,1])
    a.set_xticklabels(3*[""])
    
for a in ax[[-1,-1,-1,1], np.arange(4)]:
    a.set_xticklabels([-1,0,1])
    a.set_xlabel("Correlation", fontsize = fs)


ylims = [[1,2.6], [1, 2.6], [0,.2], [-1.25,0]]
ylims = [[1.7, 2.6], [1.7, 2.6], [0.05, 0.15], [-1,0]]
#yticks = [[1, 1.5, 2, 2.5], [1, 1.5, 2, 2.5], [0,.1,.2], [-1,-.5,0]]
for i in range(4):
    for j in range(4):
        ax[i,j].set_ylim(ylims[i])
        #ax[i,j].set_yticks(yticks[i])
        if j != 0:
            ax[i,j].set_yticklabels(len(ax[i,j].get_yticks())*[""])

for a in ax[2:,-1]:
    a.set_frame_on(False)
    a.set_xticks([])
    a.set_yticks([])
    a.set_title("", loc = "left")
        
fig.tight_layout()
fig.savefig("Figure_correlation_NFD_diversity.pdf")