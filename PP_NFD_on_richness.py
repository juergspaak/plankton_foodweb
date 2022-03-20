import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import pandas as pd
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
ax[-1,0].set_xlabel("Niche differences")
ax[-1,1].set_xlabel("Fitness differences")

ax[0,0].set_ylabel("Phytoplankton\nrichness")
ax[1,0].set_ylabel("Zooplankton\nrichness")

def plot(save_id = [0]):
    fig.savefig("PP_slides/PP_NFD_on_richness_{}.png".format(save_id[0]))
    save_id[0] += 1

style = [dict(marker = "o", color = "b", ls = ""),
         dict(marker = "^", color = "g", ls = "")]

NFD_all = [[[[] for i in range(3)] for j in range(2)] for k in range(2)]
rich_all = [[[[] for i in range(3)] for j in range(2)] for k in range(2)]

ax[0,0].plot(np.nan, np.nan, **style[0], label = "Change in Mean")
ax[0,0].plot(np.nan, np.nan, **style[1], label = "Change in Tradeoffs")
ax[0,0].legend()

for j, NFD in enumerate(["ND_mean_1", "FD_median_2"]):
    for k, rich in enumerate(["richness_phyto", "richness_zoo"]):
        for i, change in enumerate(["Mean", "correlation"]):
            for trait in set(data_NFD[i].trait):
                ax[k,j].plot(data_NFD[i].loc[data_NFD[i].trait == trait, NFD],
                             data_rich[i].loc[data_rich[i].trait == trait, rich],
                             **style[i])
                NFD_all[i][j][k].extend(data_NFD[i].loc[data_NFD[i].trait == trait, NFD].values)
                rich_all[i][j][k].extend(data_rich[i].loc[data_rich[i].trait == trait, rich].values)
              
            plot()
            
            
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

plot()


