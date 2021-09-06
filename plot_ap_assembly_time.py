import numpy as np
import matplotlib.pyplot as plt
import generate_plankton as gp

env = gp.generate_env(10000)

savez = np.load("Data_assembly_years.npz")
savez = {file: savez[file] for file in savez.files}
savez["dens"][np.isinf(savez["dens"])] = np.nan

for key in ["rich", "dens", "res"]:
    savez[key].shape = len(savez["change_var"]), 100, -1, savez[key].shape[-1]

savez["P"] = savez["res"][...,0]
savez["N"] = savez["res"][...,1]
savez["L"] = savez["res"][...,2]

savez["P"][savez["P"] > 20] = np.nan

del savez["res"]




fig, ax = plt.subplots(2,2, sharex = True, figsize = (10,10))

# richness
phyto = ax[0,0].plot(savez["change_var"]-0.2,
                     np.mean(savez["rich"][...,0], axis = 1),
             'ro', label = "Phyto")
zoo = ax[0,0].plot(savez["change_var"]+0.2,
                   np.mean(savez["rich"][...,1], axis = 1),
             'bo', label = "Zoo")
ax[0,0].legend(handles = [phyto[0], zoo[0]], labels = ["Phyto", "Zoo"])

# total denstiy
ax[0,1].plot(savez["change_var"]-0.2, np.nanmean(savez["dens"][...,0], axis = 1),
             'ro', label = "Phyto")
ax[0,1].plot(savez["change_var"]+0.2, np.nanmean(savez["dens"][...,1], axis = 1),
             'bo', label = "Zoo")

# resource levels
P = ax[1,0].plot(savez["change_var"], np.nanmean(savez["P"], axis = 1),
             'o', label = "P", color = "r")
ax[1,0].axhline(np.mean(env["P"]), color = 'r', ls = '--', alpha = 0.5)

N = ax[1,0].plot(savez["change_var"], np.nanmean(savez["N"], axis = 1),
             'o', label = "N", color = "b")
ax[1,0].axhline(np.mean(env["N"]), color = 'b', ls = '--', alpha = 0.5)

L = ax[1,0].plot(savez["change_var"], np.nanmean(savez["L"], axis = 1),
             'o', label = "L", color = "g")
ax[1,0].axhline(np.mean(env["I_in"]), color = 'g', ls = '--', alpha = 0.5)
ax[1,0].legend()
ax[1,0].legend(handles = [P[0], N[0], L[0]], labels = ["P", "N", "L"])
ax[1,0].semilogy()

# computation time needed
ax[1,1].plot(savez["change_var"], savez["time"], '-o')

##############################################################################
# added layout
ax[0,0].set_ylabel("Richness")
ax[0,1].set_ylabel("Total density")
ax[1,0].set_ylabel("Resource levels")
ax[1,1].set_ylabel("computation time [s]")

ax[1,1].set_xlabel("Metacommunity richness")
ax[1,0].set_xlabel("Metacommunity richness")

for i,a in enumerate(ax.flatten()):
    a.set_title("ABCD"[i])