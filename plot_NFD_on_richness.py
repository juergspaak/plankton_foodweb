import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from string import ascii_uppercase as ABC
from layouts import labels, color, ax_id
from scipy.stats import linregress

data_mean = pd.read_csv("data/assembly_mean.csv")
NFD_mean = pd.read_csv("data/NFD_mean.csv")

data_corr = pd.read_csv("data/assembly_corr.csv")
NFD_corr = pd.read_csv("data/NFD_corr.csv")

data_corr["trait"] = data_corr["tradeoff"]
NFD_corr["trait"] = NFD_corr["corr"]

data_rich = [data_mean, data_corr]
data_NFD = [NFD_mean, NFD_corr]

fig, ax = plt.subplots(2,2, sharex = "col", sharey = "row",
                       figsize = (9,9))

style = [dict(marker = "o", color = "b", ls = ""),
         dict(marker = "^", color = "g", ls = "")]

NFD_all = [[[[] for i in range(3)] for j in range(2)] for k in range(2)]
rich_all = [[[[] for i in range(3)] for j in range(2)] for k in range(2)]

for i, change in enumerate(["Mean", "correlation"]):
    for j, NFD in enumerate(["ND_mean_1", "FD_median_2"]):
        for trait in set(data_NFD[i].trait):
            ax[0,j].plot(data_NFD[i].loc[data_NFD[i].trait == trait, NFD],
                         data_rich[i].loc[data_rich[i].trait == trait, "richness_phyto"],
                         **style[i])
            NFD_all[i][j][0].extend(data_NFD[i].loc[data_NFD[i].trait == trait, NFD].values)
            rich_all[i][j][0].extend(data_rich[i].loc[data_rich[i].trait == trait, "richness_phyto"].values)
            
            ax[1,j].plot(data_NFD[i].loc[data_NFD[i].trait == trait, NFD],
                         data_rich[i].loc[data_rich[i].trait == trait, "richness_zoo"],
                         **style[i])
            NFD_all[i][j][1].extend(data_NFD[i].loc[data_NFD[i].trait == trait, NFD].values)
            rich_all[i][j][1].extend(data_rich[i].loc[data_rich[i].trait == trait, "richness_zoo"].values)
            
            
NFD_all = np.array(NFD_all, dtype = "object")
rich_all = np.array(rich_all, dtype = "object")

xlim = np.array([ax[0,0].get_xlim(), ax[0,1].get_xlim()])
color = ["blue", "green"]
for k in range(2):
    for j in range(2):
        for i in range(2):
        
            s, intercept, r, p, std = linregress(NFD_all[i,j,k],
                                                 rich_all[i,j,k])
            ls = "-" if p<.05/12 else "--"
            ax[k,j].plot(xlim[j], xlim[j]*s +intercept, ls, color = color[i],
                         label = "$R^2={}$".format(np.round(r**2,2)))
        
        s, intercept, r, p, std = linregress(NFD_all[0,j,k] + NFD_all[1,j,k],
                                                 rich_all[0,j,k]+rich_all[1,j,k])
        ls = "-" if p<.01 else "--"
        ax[k,j].plot(xlim[j], xlim[j]*s +intercept, ls, color = "cyan",
                         label = "$R^2={}$".format(np.round(r**2,2)))

for a in ax.flatten():
    a.legend()

# add reference values for fitness differences
ref = pd.read_csv("data/NFD_ref.csv")
sig = 2
mean_ref = np.nanmean(ref, axis=0)
std_ref = np.nanstd(ref, axis=0)
bounds = mean_ref + sig * std_ref * [[-1], [0], [1]]

for i, var in enumerate([0,7]):
    for j in range(2):
        ax[j,i].axvline(bounds[0,var], color = "red", ls = "--")
        ax[j,i].axvline(bounds[1,var], color = "red", ls = "-")
        ax[j,i].axvline(bounds[2,var], color = "red", ls = "--")
        
for i, a in enumerate(ax.flatten()):
    a.set_title(ABC[i], loc = "left")

ax[-1,0].set_xlabel("Niche differences")
ax[-1,1].set_xlabel("Fitness differences")

ax[0,0].set_ylabel("Phytoplankton\nrichness")
ax[1,0].set_ylabel("Zooplankton\nrichness")

fig.savefig("Figure_NFD_on_richness.pdf")

###############################################################################

fig = plt.figure()
plt.plot(data_mean.richness_zoo, data_mean.richness_phyto, 'bo')
plt.plot(data_corr.richness_zoo, data_corr.richness_phyto, 'g^')
xlim = np.array(plt.gca().get_xlim())
s, intercept, r, p, std = linregress(data_mean.richness_zoo,
                                     data_mean.richness_phyto)
plt.plot(xlim, s*xlim + intercept, 'b', label = "$R^2 = {}$".format(np.round(r**2,2)))

s, intercept, r, p, std = linregress(data_corr.richness_zoo,
                                     data_corr.richness_phyto)
plt.plot(xlim, s*xlim + intercept, 'g', label = "$R^2 = {}$".format(np.round(r**2,2)))

s, intercept, r, p, std = linregress(np.append(data_corr.richness_zoo,data_mean.richness_zoo),
                                     np.append(data_corr.richness_phyto, data_mean.richness_phyto))
plt.plot(xlim, s*xlim + intercept, color = "cyan", label = "$R^2 = {}$".format(np.round(r**2,2)))

plt.legend()

plt.xlabel("Zooplankton richness")
plt.ylabel("Phytoplankton richness")

fig.savefig("Figure_ap_phyto_vs_zoo_richness.pdf")

