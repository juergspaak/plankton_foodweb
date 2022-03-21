import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from layouts import labels, color, ax_id, corrs

data = pd.read_csv("data/assembly_corr.csv")     

fig, ax = plt.subplots(2,4, sharex = True, sharey = True, figsize = (9,9))

ax[0,0].set_ylim([1, 2.6])

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
    
for i in range(ax.shape[-1]):
    ax[1,i].legend(loc = "lower left")
    ax[1,i].set_xlim([-1,1])
    
    ax[1,i].set_xlabel("Correlation")

ax[0,0].set_ylabel("Phytoplankton\nrichness")
ax[1,0].set_ylabel("Zooplankton\nrichness")

ax[0,0].set_title("Gleaner\nopportunist")
ax[0,1].set_title("Growth\ndefense")
ax[0,2].set_title("Quality\nquantitiy")
ax[0,3].set_title("resource\ntradeoffs")

for i, a in enumerate(ax.flatten()):
    a.set_title("ABCDEFGHIJKLMN"[i], loc = "left")
    
##############################################################################
# add reference values

ref = pd.read_csv("data/assembly_reference.csv").values
sig = 2.5
mean_ref = np.nanmean(ref, axis = 0)
std_ref = np.nanstd(ref, axis = 0)
bounds = mean_ref + sig*std_ref*[[-1],[1]]
# add reference data 
for i in range(len(ax)):
    for j in range(ax.shape[-1]):
        ax[i,j].axhline(bounds[0,i], color = "red", linestyle = "--")
        ax[i,j].axhline(bounds[1,i], color = "red", linestyle = "--")
        ax[i,j].axhline(mean_ref[i], color = "red")
        

fig.savefig("Figure_assembly_correlation.pdf")