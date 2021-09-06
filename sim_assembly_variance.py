import numpy as np

from assembly_time_fun import assembly_richness
import generate_plankton as gp
from timeit import default_timer as timer

n_prec = 11

n_coms = 100


const_traits = np.append(gp.pt.phyto_traits, gp.zt.zoop_traits)
const_traits = np.append(const_traits, ["s_zp", "h_zp"])

fac = 4**np.linspace(-1,1,n_prec)
rich_all = np.full((len(const_traits), n_prec, n_coms, 2), np.nan)
res_all = np.full((len(const_traits), n_prec, n_coms, 3), np.nan)
dens_all = np.full((len(const_traits), n_prec, n_coms, 2), np.nan)

all_start = timer()
for j, trait in enumerate(const_traits):
    
    for i in range(n_prec):
        start = timer()
        traits = gp.generate_plankton(5, n_coms, diff_std = {trait: fac[i]})
        env = gp.generate_env(n_coms)
        traits = gp.phytoplankton_equilibrium(traits, env)
        
                # simulate densities
        richness, id_survive, res_equi, dens_equi = assembly_richness(
                        traits, env, plot_until = 0, ret_all = True)
        rich_all[j,i] = richness
        res_all[j,i] = res_equi
        dens_all[j,i] = np.nansum(dens_equi, axis = -1)
        dens_all[dens_all == 0] = np.nan
        print(j,i,fac[i], trait, np.mean(rich_all[j,i], axis = 0),
              timer()-start, timer()-all_start)
    np.savez("Data_assembly_var2", rich = rich_all, N = res_all[...,1],
             P = res_all[...,0], L = res_all[...,2],
             dens = np.log(dens_all),
         traits = const_traits, change_traits = fac)
