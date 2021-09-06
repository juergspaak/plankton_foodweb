import numpy as np
import matplotlib.pyplot as plt
import phytoplankton_traits as pt
import zoop_traits as zt


cases = [[key, variable] for key in ["rich", "dens"]
         for variable in ["mean", "var"]]

for case in cases:
    key = case[0]
    savez = np.load("Data_assembly_{}2.npz".format(case[1]))
    savez = {key: savez[key] for key in savez.files}
    savez["dens"][np.isinf(savez["dens"])] = np.nan
    change_trait = savez["change_traits"]
    
    fig, ax = plt.subplots(3,2, sharex = True, sharey = True, figsize = (13,13))
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
    
    
    for i, trait in enumerate(savez["traits"]):
        if trait in pt.phyto_traits:
            k = 0
        elif trait in zt.zoop_traits:
            k = 1
        elif trait in ["h_zp", "s_zp"]:
            k = 2

        ax[k,0].plot(change_trait, np.nanmean(savez[key][i], axis = 1)[...,0],
                     '.-')
        ax[k,1].plot(change_trait, np.nanmean(savez[key][i], axis = 1)[...,1],
                     '.-', label = trait)
        
        
    # add reference data
    rich_ref = savez[key][:, len(change_trait)//2]
    perc = np.nanpercentile(np.nanmean(rich_ref, axis = 1),
                            [5, 50, 95], axis = 0)
    for i in range(3):
        for j in range(2):
            ax[i,j].axhline(perc[0,j], color = "red", linestyle = '--')
            ax[i,j].axhline(perc[2,j], color = "red", linestyle = '--')
            ax[i,j].axhline(perc[1,j], color = "red")
            ax[i,j].axvline(change_trait[len(change_trait)//2], color = "k")
            
        ax[i,0].set_ylabel(key)
        ax[i,1].legend(ncol = 3)
    
   
        
    fig.savefig("Figure_assembly_{}_{}.pdf".format(*case))


###############################################################################
# resource availabilities
for case in ["mean", "var"]:
    savez = np.load("Data_assembly_{}2.npz".format(case))
    savez = {key: savez[key] for key in savez.files}
    for i in ["P", "N", "L"]:
        savez[i][savez[i]>1e3] = np.nan
    change_trait = savez["change_traits"]
    
    fig, ax = plt.subplots(3,3, sharex = True, sharey = "row",
                           figsize = (13,13))
    # add layout
    ax[-1,0].set_xlim(change_trait[[0,-1]]+[-0.1,0.1])
    ax[2,0].set_xticks(change_trait[[0,len(change_trait)//2,-1]])
    ax[2,1].set_xticks(change_trait[[0,len(change_trait)//2,-1]])
    ax[2,0].set_xticklabels(change_trait[[0,len(change_trait)//2,-1]])
    ax[2,1].set_xticklabels(change_trait[[0,len(change_trait)//2,-1]])
    
    ax[-1,0].set_xlabel("Change in {}".format(case))
    ax[-1,1].set_xlabel("Change in {}".format(case))
    ax[-1,2].set_xlabel("Change in {}".format(case))
    
    ax[0,0].set_title("Phytoplankton traits")
    ax[0,1].set_title("Zooplankton traits")
    ax[0,2].set_title("Combined traits")
    
    for i,a in enumerate(ax.flatten()):
        a.set_title("ABCDEFGHI"[i], loc = "left")
    
    if case == "var":
        ax[0,0].semilogx()
    
    for j, res in enumerate(["N", "P", "L"]):
        ax[j,0].set_ylabel(res)
        for i, trait in enumerate(savez["traits"]):
            if trait in pt.phyto_traits:
                k = 0
            elif trait in zt.zoop_traits:
                k = 1
            elif trait in ["h_zp", "s_zp"]:
                k = 2
                
            ax[j,k].plot(change_trait, np.nanmean(savez[res][i], axis = 1),
                 '.-', label = trait)
               
        # add reference values to it
        res_ref = savez[res][:, len(change_trait)//2]
        perc = np.nanpercentile(np.nanmean(res_ref, axis = 1),
                            [5, 50, 95], axis = 0)
        for k in range(3):
            ax[j,k].axhline(perc[0], color = "red", linestyle = "--")
            ax[j,k].axhline(perc[1], color = "red", linestyle = "-")
            ax[j,k].axhline(perc[2], color = "red", linestyle = "--")
            ax[j,k].axvline(change_trait[len(change_trait)//2], color = "k")
    
    ax[-1,0].legend(ncol = 3)
    ax[-1,1].legend(ncol = 3)
    ax[-1,2].legend(ncol = 3)
    
        
    fig.savefig("Figure_assembly_res_{}_{}.pdf".format(*case))