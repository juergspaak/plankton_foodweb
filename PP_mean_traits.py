import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from layouts import labels, color, ax_id
import generate_plankton as gp

def plot(save_id = [0]):
    fig.savefig("PP_slides/PP_richness_over_time_{}.png".format(save_id[0]))
    save_id[0] += 1
    
    
data = pd.read_csv("data/assembly_mean.csv")
change_trait = np.sort(list(set(data["change"])))
traits = set(data["trait"])

fig = plt.figure(figsize = (7,7))
ax = np.empty((2,3), dtype = "object")

ax[0,0] = fig.add_subplot(2,3,1)
ax[1,0] = fig.add_subplot(2,3,4)


ax[0,0].set_xlim([-1,1])
ax[0,0].set_xticks([])
ax[1,0].set_xticks([-1,0,1])
ax[1,0].set_ylim([1, 2.6])
ax[0,0].set_ylim([1, 2.6])
ax[0,0].set_yticks([1, 1.5, 2, 2.5])
ax[1,0].set_yticks([1, 1.5, 2, 2.5])
ax[1,0].set_xlabel("Change in Mean")
ax[1,0].set_ylabel("Zooplankton\nrichness")
ax[0,0].set_ylabel("Phytoplankton\nrichness")
plot()

locs = [3, 8, 4]
for i in range(3):
    inset = inset_axes(ax[1,0], width = "20%", height = "20%", loc = locs[i])
    inset.set_xticks([])
    inset.set_yticks([])
    bins = np.linspace(-3.5, 3.5, 50)
    inset.hist(np.random.normal(0, 1, 10000), bins = bins)
    inset.hist(np.random.normal(2*(-1+i), 1, 10000), bins = bins, alpha = .5)

plot()

ax[0,0].set_title("Growth traits")
for i, trait in enumerate(sorted(set(data["trait"]))):        
    # sort traits by resource traits, growth traits, species interactoins
    
    if ax_id[trait] != 0:
        continue
    ind = data["trait"] == trait
    ax[0,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_phyto"],
                 '.-', color = color[trait], label = labels[trait],
                 lw = 3)
    ax[1,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_zoo"],
                 '.-', color = color[trait], label = labels[trait], lw = 3)
ax[0,0].legend(["Mortality Z", "Growth rates P", "Growth rate Z"])
plot()

# add reference guidelines
ref = pd.read_csv("data/assembly_reference.csv").values
sig = 2.5
mean_ref = np.nanmean(ref, axis = 0)
std_ref = np.nanstd(ref, axis = 0)
bounds = mean_ref + sig*std_ref*[[-1],[1]]
# add reference data 
for i in range(2):
    ax[i,0].axhline(bounds[0,i], color = "red", linestyle = "--")
    ax[i,0].axhline(bounds[1,i], color = "red", linestyle = "--")
    ax[i,0].axhline(mean_ref[i], color = "red")
    ax[i,0].axvline(change_trait[len(change_trait)//2], color = "w")

plot()

##############################################################################
# Resource traits
ax[1,1] = fig.add_subplot(2,3,5)
ax[0,1] = fig.add_subplot(2,3,2)

ax[0,1].set_xlim([-1,1])
ax[0,1].set_xticks([])
ax[1,1].set_xticks([-1,0,1])
ax[1,1].set_ylim([1, 2.6])
ax[0,1].set_ylim([1, 2.6])
ax[0,1].set_yticks([])
ax[1,1].set_yticks([])
ax[1,1].set_xlabel("Change in Mean")
ax[0,1].set_title("Resoruce traits")

for i, trait in enumerate(sorted(set(data["trait"]))):        
    # sort traits by resource traits, growth traits, species interactoins
    
    if ax_id[trait] != 1:
        continue
    ind = data["trait"] == trait
    ax[0,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_phyto"],
                 '.-', color = color[trait], label = labels[trait],
                 lw = 3)
    ax[1,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_zoo"],
                 '.-', color = color[trait], label = labels[trait], lw = 3)
ax[0,0].legend(["Mortality Z", "Growth rates P", "Growth rate Z"])
for i in range(2):
    ax[i,1].axhline(bounds[0,i], color = "red", linestyle = "--")
    ax[i,1].axhline(bounds[1,i], color = "red", linestyle = "--")
    ax[i,1].axhline(mean_ref[i], color = "red")
    ax[i,1].axvline(change_trait[len(change_trait)//2], color = "w")

ax[0,1].legend(["Absorption", "N uptake", "P uptake", "Halfsat.\nlight", "Halfsat. N", "Halfsat. P"])
plot()

##############################################################################
# Trophic traits
ax[1,2] = fig.add_subplot(2,3,6)
ax[0,2] = fig.add_subplot(2,3,3)

ax[0,2].set_xlim([-1,1])
ax[0,2].set_xticks([])
ax[1,2].set_xticks([-1,0,1])
ax[1,2].set_ylim([1, 2.6])
ax[0,2].set_ylim([1, 2.6])
ax[0,2].set_yticks([])
ax[1,2].set_yticks([])
ax[1,2].set_xlabel("Change in Mean")
ax[0,2].set_title("Species interaction\ntraits")


label = ["Edibility", "Handling time"]
for i, trait in enumerate(["e_P", "h_zp"]):        
    # sort traits by resource traits, growth traits, species interactoins
    
    if ax_id[trait] != 2:
        continue
    ind = data["trait"] == trait
    ax[0,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_phyto"],
                 '.-', color = color[trait], label = label[i],
                 lw = 3)
    ax[1,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_zoo"],
                 '.-', color = color[trait], label = label[i], lw = 3)

ax[0,2].legend()
for i in range(2):
    ax[i,2].axhline(bounds[0,i], color = "red", linestyle = "--", zorder = 3)
    ax[i,2].axhline(bounds[1,i], color = "red", linestyle = "--", zorder = 3)
    ax[i,2].axhline(mean_ref[i], color = "red", zorder = 3)
    ax[i,2].axvline(change_trait[len(change_trait)//2], color = "w")
plot()


label = ["Nutritional\nvalue", "Clearance", "Halfsat.\nZooplankton"]
for i, trait in enumerate(["R_P", "c_Z", "k_Z"]):        
    # sort traits by resource traits, growth traits, species interactoins
    
    if ax_id[trait] != 2:
        continue
    ind = data["trait"] == trait
    ax[0,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_phyto"],
                 '.-', color = color[trait], label = label[i],
                 lw = 3)
    ax[1,ax_id[trait]].plot(data.loc[ind, "change"],
                            data.loc[ind, "richness_zoo"],
                 '.-', color = color[trait], label = label[i], lw = 3)

ax[0,2].legend(fontsize = 9)
plot()