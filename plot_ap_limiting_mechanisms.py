import numpy as np
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt

# cases: Each of the three resources can be limiting
# predators can be present or not
cases = ["N", # nitrogen not limiting
         "P", # phosphorus not limiting
         "L", # light not limiting
         "Z"] # zooplankton not present

combs = [""] + list(combinations(cases, 1)) + list(combinations(cases, 2)) + list(combinations(cases,3))
combs.remove(("N", "P", "L")) # remove case where no resource is limiting

richness = pd.DataFrame(np.nan, columns = ["phyto", "zoo"]
                        , index = range(len(combs)))
richness["label"] = "NPLZ"
richness["P"] = True
richness["N"] = True
richness["L"] = True
richness["Z"] = True


label = np.array(["P", "N", "L", "Z"])
path = "data/"
counter = 0
for comb in combs:
    
    
    try:
        save = "assembly_non_lim{}_20_1000.npz".format((len(comb)*"_{}").format(*comb))
        values = np.load(path + save)
        richness.loc[counter, ["phyto", "zoo"]] = np.nanmean(np.nansum(
                                values["present"][...,-1],axis = -1),axis = 0)
    except:
        continue
    
    for c in comb:
        richness.loc[counter, c] = False
        richness.loc[counter, "label"] = richness.loc[counter, "label"].replace(c, "")
    counter += 1

richness.index = richness.label 

##############################################################################
# plot results
fig, ax = plt.subplots(2,1,sharex = True, sharey = True, figsize = (9,9))
ax[0].set_ylabel("Phytoplankton\nrichness")
ax[1].set_ylabel("Zooplankton\nrichness")
ax[1].set_xticks(range(6))
ax[1].set_xticklabels(["One\nResource", "Two\nResources",
                           "Three\nResources",
                           "Zooplankton\nand one\nResource",
                           "Zooplankton\nand two\nResources",
                           "Zooplankton\nand three\nResources"])


ax[0].set_ylim([1, 2.6])
ax[0].set_xlim([-0.2, 5.2])

combs = [[c[-1] for c in comb] for comb in combs]

dx = np.array([-0.1,0,0.1])
ax[0].plot(dx, richness.loc[["L", "P", "N"], "phyto"], 'bo')

ax[0].plot(1+dx, richness.loc[["NP", "NL", "PL"], "phyto"], 'bo')

ax[0].plot(2, richness.loc["NPL", "phyto"], 'bo')


ax[0].plot(3 + dx, richness.loc[["PZ", "NZ", "LZ"], "phyto"], 'bo')
ax[1].plot(3 + dx, richness.loc[["PZ", "NZ", "LZ"], "zoo"], 'bo')

ax[0].plot(4 + dx, richness.loc[["NPZ", "NLZ", "PLZ"], "phyto"], 'bo')
ax[1].plot(4 + dx, richness.loc[["NPZ", "NLZ", "PLZ"], "zoo"], 'bo')

ax[0].plot(5, richness.loc["NPLZ", "phyto"], 'bo')
ax[1].plot(5, richness.loc["NPLZ", "zoo"], 'bo')


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

fig.savefig("Figure_ap_limiting_machanisms.pdf")