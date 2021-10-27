import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import generate_plankton as gp

data = pd.read_csv("data/assembly_corr.csv")     

fig, ax = plt.subplots(2,5, sharex = True, sharey = True, figsize = (9,9))

ax[0,0].set_ylim([1, 2.6])

zoop = ["c_Z:m_Z", "m_Z:mu_Z"]
gleaner = ["c_n:k_n", "c_p:k_p", "c_Z:k_Z", "k_Z:mu_Z", "a:k_l"]
resources = ["c_n:c_p", "k_n:k_p", "k_l:k_n", "k_l:k_p"]
growth_defense = ["R_p:e_P", "R_P:mu_P", "e_P:mu_P"]
weird = ["c_n:mu_P","c_Z:mu_Z", "R_P:k_n"]

corrs = [zoop, gleaner, resources, growth_defense, weird]

for corr in set(data["tradeoff"]):
    i = np.arange(len(corrs))[[(corr in co) for co in corrs]][0]
    
    ind = data["tradeoff"] == corr
    
    ax[0,i].plot(data.loc[ind, "corr"], data.loc[ind, "richness_phyto"], '.-',
                     label = corr)
    ax[1,i].plot(data.loc[ind, "corr"], data.loc[ind, "richness_zoo"], '.-',
                     label = corr)
    
for i in range(3):
    ax[1,i].legend(loc = "lower left")
    ax[1,i].set_xlim([-1,1])
    
    ax[1,i].set_xlabel("Correlation")

ax[0,0].set_ylabel("Phytoplankton\nrichness")
ax[1,0].set_ylabel("Zooplankton\nrichness")
    
ax[0,0].set_title("Growth defense")
ax[0,1].set_title("Gleaner-Opportunist")
ax[0,2].set_title("Resource tradeoffs")

ax[0,0].set_title("random")
ax[0,1].set_title("Gleaner\nopportunist")
ax[0,2].set_title("resource\ntradeoffs")
ax[0,3].set_title("growth\ndefense")
ax[0,4].set_title("New\ntradeoffs")

for i, a in enumerate(ax.flatten()):
    a.set_title("ABCDEFGHIJKLMN"[i], loc = "left")
    
##############################################################################
# add reference values

data = pd.read_csv("data/assembly_mean.csv")
ind_ref = data["change"] == 0
perc = np.nanpercentile(data.loc[ind_ref, ["richness_phyto", "richness_zoo"]].values,
                        [25,50,75], axis = 0).T
ref = data.loc[ind_ref, ["richness_phyto", "richness_zoo"]].values
sig = 2.5
mean_ref = np.nanmean(ref, axis = 0)
std_ref = np.nanstd(ref, axis = 0)
bounds = mean_ref + sig*std_ref*[[-1],[1]]
# add reference data 
for i in range(len(ax)):
    for j in range(ax.shape[-1]):
        ax[i,j].axhline(bounds[0,i], color = "red", linestyle = "--")
        ax[i,j].axhline(bounds[1,i], color = "red", linestyle = "--")
        ax[i,j].axhline(perc[i, 1], color = "red")
        ax[0,j].legend()
        

fig.savefig("Figure_assembly_correlation.pdf")