import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
"""inset_axes = inset_axes(parent_axes,
                    width="30%", # width = 30% of parent_bbox
                    height=1., # height : 1 inch
                    loc=3)"""

data = np.load("Data_fast_NFD3.npz", allow_pickle=True)
const_traits = data["const_traits"]
ind = (const_traits != "size_P") & (const_traits != "size_Z")
ND_all = data["ND"][ind]
FD_all = data["FD"][ind]
add_var = data["change_trait"]
issues_all = data["issues"][ind]
const_traits = const_traits[ind]

ND = ND_all[:, add_var == 0]
FD = FD_all[:, add_var == 0]
ND = ND.reshape(-1,2)
FD = FD.reshape(-1,2)



fig = plt.figure(figsize = (10,6))
percent = [5, 25]
dx = .05
y_top = .9
h_hist = .08
for i in range(2):
    ax_scat = fig.add_axes([i/2+dx, dx, .35, .8])
    print([i/2+dx, dx, i/2 + (.5-dx), 0.9])
    ax_scat.scatter(ND, FD, s = 1)
    ax_scat.set_xlim(np.nanpercentile(ND, [percent[i], 100-percent[i]]))
    ax_scat.set_ylim(np.nanpercentile(FD, [percent[i], 100-percent[i]]))
    ax_scat.set_xlabel("Niche differences")
    ax_scat.set_ylabel("Fitness differences")    
    
    loc = ax_scat.get_position().bounds
    ax_ND = fig.add_axes([loc[0], loc[1] + loc[-1], loc[2], h_hist])
    ax_ND.hist(ND.flatten(), bins = np.linspace(*ax_scat.get_xlim(), 30))
    ax_ND.set_xticks([])
    ax_ND.set_yticks([])
    ax_ND.set_frame_on(False)
    
    ax_FD = fig.add_axes([loc[0] + loc[2], loc[1], h_hist, loc[3]])
    ax_FD.hist(FD.flatten(), bins = np.linspace(*ax_scat.get_ylim(), 30),
               orientation = "horizontal")
    ax_FD.set_xticks([])
    ax_FD.set_yticks([])
    ax_FD.set_frame_on(False)