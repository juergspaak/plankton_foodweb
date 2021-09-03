import numpy as np
import matplotlib.pyplot as plt
import phytoplankton_traits as pt
import zoop_traits as zt

"""
c = "var"
savez = np.load("Data_assembly_{}_org.npz".format(c))
rich = np.log(savez["dens"].copy())
rich[np.isinf(rich)] = np.nan
np.savez("Data_assembly_{}.npz".format(c),
         rich = savez["rich"], res = savez["res"], dens = rich,
         change_trait = fac, traits = const_traits)
"""
cases = [[key, variable] for key in ["rich", "dens"] for variable in ["mean", "var"]]

for case in cases:
    key = case[0]
    savez = np.load("Data_assembly_{}.npz".format(case[1]))
    change_trait = savez["change_trait"]
    
    fig, ax = plt.subplots(3,2, sharex = True, sharey = True, figsize = (13,13))
    
    
    
    for i, trait in enumerate(savez["traits"]):
        if trait in pt.phyto_traits:
            k = 0
        elif trait in zt.zoop_traits:
            k = 1
        elif trait in ["h_zp", "s_zp"]:
            k = 2
        ax[k,0].plot(change_trait, np.nanmean(savez[key][i], axis = 1)[:,0], '.-')
        ax[k,1].plot(change_trait, np.nanmean(savez[key][i], axis = 1)[:,1], '.-',
                     label = trait)
        
        
    # add reference data
    rich_ref = savez[key][:, len(change_trait)//2]
    perc = np.nanpercentile(np.nanmean(rich_ref, axis = 1),
                            [5, 50, 95], axis = 0)
    for i in range(3):
        for j in range(2):
            ax[i,j].axhline(perc[0,j], color = "red", linestyle = '--')
            ax[i,j].axhline(perc[2,j], color = "red", linestyle = '--')
            ax[i,j].axhline(perc[1,j], color = "red")
            ax[i,j].axvline(0, color = "k")
            
        ax[i,0].set_ylabel(key)
        ax[i,1].legend(ncol = 3)
    
    # add layout
    ax[-1,0].set_xlim(change_trait[[0,-1]]+[-0.1,0.1])
    ax[2,0].set_xticks(change_trait[[0,len(change_trait)//2,-1]])
    ax[2,1].set_xticks(change_trait[[0,len(change_trait)//2,-1]])
    ax[2,0].set_xticklabels(change_trait[[0,len(change_trait)//2,-1]])
    ax[2,1].set_xticklabels(change_trait[[0,len(change_trait)//2,-1]])
    
    ax[0,0].set_title("Phytoplankton")
    ax[0,1].set_title("Zooplankton")
    
    ax[-1,0].set_xlabel("Change in {}".format(case[1]))
    ax[-1,1].set_xlabel("Change in {}".format(case[1]))
    
    for i,a in enumerate(ax.flatten()):
        a.set_title("ABCDEF"[i], loc = "left")
    
    if case[1] == "var":
        ax[0,0].semilogx()
        
    fig.savefig("Figure_assembly_{}_{}.pdf".format(*case))