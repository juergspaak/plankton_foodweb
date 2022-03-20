import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

data = np.load("data/assembly_long_60_1000_3.npz")
#data = np.load("data/assembly_phyto_only_20_1000_5.npz")

invasion_time = np.argmax(data["present"], axis = -1)


persistence_length = np.sum(data["present"], axis = -1)-1

fig, ax = plt.subplots(2,2, sharex = "row", sharey = "row")

for i in range(2):
    a, binx, biny = np.histogram2d(invasion_time[:,i].flatten(),
                    persistence_length[:,i].flatten(),
                   bins = [np.arange(data["r_phyto"]+1), np.arange(data["r_phyto"]+2)])
    
    a = a/np.sum(a, axis = 1, keepdims = True)
    # remove cases with no species at all
    a[a==0] = np.nan
    
    #a = a/np.nansum(a, axis = 1, keepdims = True)
    
    cmap = ax[0,i].imshow(a[:,1:].T, origin = "lower")
    fig.colorbar(cmap, ax = ax[0,i], aspect = "auto")
    
    
    ax[0,i].plot(1+np.arange(len(a)), np.nansum(a*np.arange(a.shape[-1]), axis = 1),
                 color = "red")
    ax[1,i].plot(1+np.arange(len(a)), 1-a[:,0])
    # comparison to random expectation
    ax[1,i].plot(1 + np.arange(len(a)),
                 np.mean(np.sum(data["present"], axis = -2),
                         axis = 0)[i,-1]/(1+np.arange(len(a))))
    
    # best fitting curve
    x = 1 + np.arange(len(a))
    y = 1-a[:,0]
    s, inter, r, p, std = linregress(x, x*y)
    print(s, inter, std)
    ax[1,i].plot(x, inter/x + s)
    ax[1,i].axhline(s, color = "green", linestyle = "--")
    t_cut = 50
    s, inter, r, p, std = linregress(x[:t_cut], (x*y)[:t_cut])
    print(s, inter, std)


ax[1,0].set_ylim([0,1])
ax[1,0].set_ylabel("Invasion probability")

ax[0,0].set_ylabel("Survival time")
ax[1,0].set_xlabel("Arrival time")
ax[1,1].set_xlabel("Arrival time")

ax[0,0].set_title("Phytoplankton")
ax[0,1].set_title("Zooplankton")

fig.savefig("Figure_ap_uninvadable_community.pdf")