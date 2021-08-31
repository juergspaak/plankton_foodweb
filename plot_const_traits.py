import matplotlib.pyplot as plt
import generate_plankton as gp
import numpy as np
import phytoplankton_traits as pt
import zoop_traits as zt

r_phyto = 30
n_coms = 100
tradeoffs = {"m_Z:mu_Z": -0.6, "c_n:c_p": 0.5}
diff_std = {"a":2, "c_Z": 2}
diff_mean = {"mu_P": 2, "k_Z": 2}
traits_const = gp.generate_plankton(r_phyto, n_coms, diff_mean = diff_mean,
                                    tradeoffs = tradeoffs, diff_std = diff_std)

traits = gp.generate_plankton(r_phyto, n_coms)

bins = 30

keys = np.append(pt.phyto_traits, zt.zoop_traits)
traits = {key: np.log(traits[key]).flatten() for key in keys}
traits_const = {key: np.log(traits_const[key]).flatten() for key in keys}


fig, ax = plt.subplots(len(pt.phyto_traits), len(pt.phyto_traits),
                               figsize = (12,12), sharex = "col", sharey = "row")
bins = 20
for i,keyi in enumerate(pt.phyto_traits):
    ax[-1,i].set_xlabel(keyi)
    ax[i,0].set_ylabel(keyi)
    for j, keyj in enumerate(pt.phyto_traits):               
        if i<j:
            
            ax[j,i].scatter(traits[keyi], traits[keyj], s = 1,
                        alpha = 0.1, color = "blue")
            
            ax[j,i].scatter(traits_const[keyi], traits_const[keyj], s = 1,
                        alpha = 0.1, color = "orange")
        if j>i:
            ax[i,j].set_frame_on(False)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
    # plot histogram

    ax_hist = fig.add_subplot(len(pt.phyto_traits),
                              len(pt.phyto_traits),
                              1 + i + i*len(pt.phyto_traits))
    ax_hist.hist(traits[keyi], bins, density = True, color = "blue")
    ax_hist.set_xticklabels([])
    ax_hist.set_yticklabels([])
    ax_hist.hist(traits_const[keyi], bins, density = True,
                 alpha = 0.5, color = "orange")
        
    ax[-1,-1].set_xlim(ax[-1,0].get_ylim())
#"""


n = len(zt.zoop_traits)
fig, ax = plt.subplots(n,n,figsize = (9,9), sharex = "col", sharey = "row")

fs = 16
s = 5
for i,keyi in enumerate(zt.zoop_traits):
    ax[-1,i].set_xlabel(keyi)
    ax[i,0].set_ylabel(keyi)
    for j,keyj in enumerate(zt.zoop_traits):
        if i<j:
            ax[j,i].scatter(traits[keyi], traits[keyj], s = 1,
                        alpha = 0.1, color = "blue")
            
            ax[j,i].scatter(traits_const[keyi], traits_const[keyj], s = 1,
                        alpha = 0.1, color = "orange")
        if j>i:
            ax[i,j].set_frame_on(False)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
    ax_hist = fig.add_subplot(n,n,1 + (n+1)*i)
    ax_hist.hist(traits[keyi], bins, density = True, color = "blue")
    ax_hist.hist(traits_const[keyi], bins, density = True, color = "orange",
                 alpha = 0.5)
    ax_hist.set_xticklabels([])
    ax_hist.set_yticklabels([])

ax[0,0].set_ylim(ax[0,0].get_xlim())
ax[-1,-1].set_xlim(ax[-1,-1].get_ylim())
#"""