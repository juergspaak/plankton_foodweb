import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n_prec = 9
path = "data/"
save = "assembly_R_star_20_1000_{}.npz"
save = "assembly_affinity_20_1000_{}.npz"

richness = pd.DataFrame(np.nan, columns = ["phyto", "zoo", "corr"],
                        index = range(n_prec))

for i in range(n_prec):
    try:
        data = np.load(path + save.format(i))
        richness.loc[i, ["phyto", "zoo"]] = np.nanmean(np.nansum(
                                data["present"][...,-1],axis = -1),axis = 0)
        richness.loc[i, "corr"] = data["corr"]
    except (FileNotFoundError, KeyError):
        continue
    
print(richness)

fig, ax = plt.subplots(2,1, sharex = True, sharey = True, figsize = (9,9))

ax[0].set_ylim([1,2.6])
ax[0].set_xlim([-1,1])

ax[1].set_xlabel("Correlation")
ax[0].set_ylabel("Phytoplankton\nrichness")
ax[1].set_ylabel("Zooplankton\nrichness")

ax[0].plot(richness["corr"], richness.phyto, ".-")
ax[1].plot(richness["corr"], richness.zoo, ".-")

# add reference guidelines
ref = pd.read_csv("data/assembly_reference.csv").values
sig = 2.5
mean_ref = np.nanmean(ref, axis = 0)
std_ref = np.nanstd(ref, axis = 0)
bounds = mean_ref + sig*std_ref*[[-1],[1]]
# add reference data 
for i in range(2):
    ax[i].axhline(bounds[0,i], color = "red", linestyle = "--")
    ax[i].axhline(bounds[1,i], color = "red", linestyle = "--")
    ax[i].axhline(mean_ref[i], color = "red")
    
fig.savefig("Figure_int_affinities_correlation.pdf")