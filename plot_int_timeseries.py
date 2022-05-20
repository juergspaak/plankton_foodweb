import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

from assembly_time_fun import evolve_time, select_survivors
import generate_plankton as gp
from timeit import default_timer as timer

np.random.seed(hash("Fancy project")%(2**32-1))

start = timer()
n_coms = 2
traits = gp.generate_plankton(20, n_coms)
env = gp.generate_env(n_coms, fluct_env=["N"])
traits = gp.phytoplankton_equilibrium(traits, env)
env["freq_N"][:] = 1

ti, envi, i = gp.select_i(traits, env, 0)

species_order = np.argsort(np.random.rand(ti["r_phyto"]))
ind_phyto = species_order[[0]]
ind_zoo = species_order[[0]]
N_start = np.array([envi["N"], envi["P"],
                    ti["N_star_P_res"][species_order[0]], 1])
N_start[np.isnan(N_start)] = 1
time_org = np.array([0, 365])
time = time_org.copy()


fig, ax = plt.subplots(3,2, sharex = "col", figsize = (12,12))
colors = viridis(np.linspace(0,1,traits["r_phyto"]))

def rescale(x):
    return (x - np.amin(x))/(np.amax(x)-np.amin(x))
colors = np.array([viridis(rescale(np.log(ti["size_P"]))),
          viridis(rescale(np.log(ti["size_Z"])))])
colors = np.array([viridis(np.linspace(0,1,ti["r_phyto"])),
                   viridis(np.linspace(0,1,ti["r_phyto"]))])

present = np.full((2, traits["r_phyto"], traits["r_phyto"]+1), False, dtype = bool)
#for i in range(ti["r_phyto"]):
#    present[:, species_order[i], i] = 1

for j in range(ti["r_phyto"]):
    present[0, ind_phyto, j] = True
    present[1, ind_zoo, j] = True
    tc = select_survivors(ti, ind_phyto, ind_zoo)
    s_phyto, s_zoo, N_start, sol = evolve_time(N_start, tc,
                                               envi, time)
    
    
    # plot resources
    ax[0,0].semilogy(sol.t, np.exp(sol.y[0].T), 'r', label = "N")
    ax[0,0].semilogy(sol.t, np.exp(sol.y[1].T), 'b', label = "P")
    
    for k, ind in enumerate(ind_phyto):
        ax[1,0].semilogy(sol.t, np.exp(sol.y[2+k]),
                       '-' if s_phyto[k] else '--',
                       alpha = 0.3+s_phyto[k]/2,
                       color = colors[0, ind])
    for k, ind in enumerate(ind_zoo):
        ax[2,0].semilogy(sol.t, np.exp(sol.y[2 + len(ind_phyto) + k]),
                       '-' if s_zoo[k] else '--',
                       alpha = 0.3+s_zoo[k]/2,
                       color = colors[1, ind])
        
    ind_phyto = ind_phyto[s_phyto]
    ind_zoo = ind_zoo[s_zoo]
    
    if j < ti["r_phyto"]-1:
        N_start = np.insert(np.exp(N_start),
                            [2+sum(s_phyto), 2+sum(s_phyto)+sum(s_zoo)],
                            [10,0.1])
        
        ind_phyto = np.append(ind_phyto, species_order[j+1])
        ind_zoo = np.append(ind_zoo, species_order[j+1])
        time = time[-1] + time_org
        
present[0, ind_phyto, -1] = True
present[1, ind_zoo, -1] = True  
  

###############################################################################
# add presence patterns      
        


years = np.arange(ti["r_phyto"] + 1)
for i in range(2):
    size = np.log(ti["size_P"]) if i == 0 else np.log(ti["size_Z"])
    for j in range(ti["r_phyto"]):
        ax[i+1,1].plot(years[present[i, j]], np.repeat(size[j], sum(present[i,j])),
                        color = colors[i, j])
        ax[i+1,1].plot(years[np.nanargmax(present[i,j])], size[j],
                        color = colors[i, j], marker= "o")
        
richness = np.sum(present, axis = 1)
richness[:, 1:-1] = richness[:, 1:-1] -1
ax[0,1].plot(richness.T, '.-', label = "hi")

###############################################################################
# layout
ax[0,0].legend(["N", "P"])


ax[1,0].set_ylim([1e-0, 1e6])
ax[2,0].set_ylim([1e-2, 1e5])
ax[0,1].set_ylim([0, None])

ax[-1,0].set_xlabel("Time [days]")
ax[-1,1].set_xlabel("Time [years]")

ax[0,0].set_ylabel("Resource\nconcentration")
ax[1,0].set_ylabel("Phytoplankton\ndensity")
ax[2,0].set_ylabel("Zooplankton\ndensity")

ax[0,1].set_ylabel("Species\nrichness")
ax[1,1].set_ylabel("Phytoplankton\nsize")
ax[2,1].set_ylabel("Zooplankton\nsize")
ax[0,1].legend(["Phytoplankton", "Zooplankton"])

for i, a in enumerate(ax.flatten()):
    a.set_title("ABCDEF"[i])
    
fig.savefig("Figure_int_timeseries.pdf")