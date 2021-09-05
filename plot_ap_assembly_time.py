import numpy as np
import matplotlib.pyplot as plt
import generate_plankton as gp

env = gp.generate_env(10000)

savez = np.load("Data_assembly_years2.npz")
dens = savez["dens"]
dens[np.isinf(dens)] = np.nan

fig, ax = plt.subplots(2,2, sharex = True, figsize = (10,10))

# richness
ax[0,0].plot(savez["change_var"], np.mean(savez["rich"][...,0], axis = 1),
             '-o', label = "Phyto")
ax[0,0].plot(savez["change_var"], np.mean(savez["rich"][...,1], axis = 1),
             '-o', label = "Zoo")
ax[0,0].legend()

# total denstiy
ax[0,1].plot(savez["change_var"], np.nanmean(dens[...,0], axis = 1),
             '-o', label = "Phyto")
ax[0,1].plot(savez["change_var"], np.nanmean(dens[...,1], axis = 1),
             '-o', label = "Zoo")

# resource levels
ax[1,0].plot(savez["change_var"], np.nanmean(savez["res"], axis = 1)[:,0],
             '-o', label = "P", color = "r")
ax[1,0].axhline(np.mean(env["P"]), color = 'r', ls = '--', alpha = 0.5)

ax[1,0].plot(savez["change_var"], np.nanmean(savez["res"], axis = 1)[:,1],
             '-o', label = "N", color = "b")
ax[1,0].axhline(np.mean(env["N"]), color = 'b', ls = '--', alpha = 0.5)

ax[1,0].plot(savez["change_var"], np.nanmean(savez["res"], axis = 1)[:,2],
             '-o', label = "L", color = "g")
ax[1,0].axhline(np.mean(env["I_in"]), color = 'g', ls = '--', alpha = 0.5)
ax[1,0].legend()

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