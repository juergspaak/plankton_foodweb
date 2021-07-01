import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data_diversity_timeseries_save.csv")
data["richness"] = data.r_phyto + data.r_zoo
data.loc[data.trait_comb != data.trait_comb, "trait_comb"] = "reference"
data = data[data.r_zoo >0]
print(data.shape)
#data = data[data.stable <0]
#data = data[data.feas > 0]
print(data.shape)

trait_combs = np.array(list(set(data.trait_comb)))

r_max = int(np.amax(data["richness"]))
spec_rich = np.empty((len(trait_combs), 3, r_max))


for i, trait in enumerate(trait_combs):
    data_c = data[data.trait_comb == trait]
    for j in range(r_max):
        for k, index in enumerate(["r_phyto", "r_zoo", "richness"]):
            
            spec_rich[i,k,j] = np.nansum(data_c[index] == j)/len(data_c)


mean_sp_richness = np.sum(spec_rich*np.arange(r_max), axis = -1)

arg_sort = np.argsort(mean_sp_richness[:,-1])

mean_sp_richness = mean_sp_richness[arg_sort]
spec_rich = spec_rich[arg_sort]
trait_combs = trait_combs[arg_sort]

fig, ax = plt.subplots(3,1,figsize = (5,9), sharey = True, sharex = True)

loc = np.arange(len(trait_combs))
for j in range(3):
    for i in range(r_max):
        ax[j].bar(loc, spec_rich[:,j,i], bottom = np.sum(spec_rich[:,j,:i], axis = 1))
    ax_mean = ax[j].twinx()
    ax_mean.plot(loc, mean_sp_richness[:,j], 'ko')
        
    
ax[-1].set_xticks(np.arange(len(trait_combs)))
ax[-1].set_xticklabels(trait_combs, rotation = 90)

fig.savefig("Figure_diversity_timeseries.pdf")