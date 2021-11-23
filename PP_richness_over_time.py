import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

#data = np.load("data/assembly_long_60_1000_0.npz")
def plot(save_id = [0]):
    fig.savefig("PP_slides/PP_richness_over_time_{}.png".format(save_id[0]))
    save_id[0] += 1

###############################################################################
# plot results    
fig, ax = plt.subplots(3,3, sharex = True, sharey = "row",
                       figsize = (7,7))

##############################################################################
# add layout
ax[0,0].set_ylim([0,7])
ax[-1,0].set_xlabel("Years")
ax[-1,1].set_xlabel("Years")
ax[-1,2].set_xlabel("Years")

ax[0,0].set_ylabel("Species richness")
ax[1,0].set_ylabel("Phytoplankton\nsize (log)")
ax[2,0].set_ylabel("Zooplankton\nsize (log)")

for i, a in enumerate(ax.flatten()):
    a.set_title("ABCDEFGHI"[i], loc = "left")
    
fig.tight_layout()

richness = np.sum(data["present"], axis = -2)

richness[..., 1:-1] = richness[..., 1:-1] -1


years = np.arange(data["r_phyto"] + 1)
focal_trait = ["size_P", "size_Z"]

for i in range(2):
    for j in range(2):
        size = np.log(data[focal_trait[j]][i])
        for k in range(data["r_phyto"]):
            ax[j+1, i].plot(years[data["present"][i, j, k]], np.repeat(size[k], sum(data["present"][i, j,k])),
                           color = "w")
            ax[j+1, i].plot(years[np.nanargmax(data["present"][i, j, k])], size[k], "w.")
        
    ax[0,i].plot(years, richness[i].T, '.-', label = "hi")
    ax[0,0].legend(["Phytoplankton", "Zooplankton"])
    plot()


richness = richness[...,1:]
# average species richness
ax[0,2].plot(years[1:], np.mean(richness[:,0], axis = 0), ".-")
ax[0,2].plot(years[1:], np.mean(richness[:,1], axis = 0), ".-")


# trait distributions
nbins = 15
hists = np.empty((2, data["r_phyto"], nbins))
p = .1

for i in range(2):
    size = np.log(data[focal_trait[i]])
    mean = np.empty(data["r_phyto"])
    bins = np.linspace(*np.nanpercentile(size, [p,100-p]), nbins+1)
    for j in range(0,data["r_phyto"]):
        hists[i,j] = np.histogram(size[data["present"][:,i,:,j+1]], bins = bins,
                                  density = True)[0]
        mean[j] = np.mean(size[data["present"][:,i,:,j]])
        
    ax[i+1,2].imshow(hists[0].T, origin = "lower", aspect = "auto",
               extent = [0,data["r_phyto"], bins[0], bins[-1]])
    ax[i+1,2].plot(years[1:]+.5, mean, "r-")

plot()