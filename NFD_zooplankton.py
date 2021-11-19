import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import solve_ivp, simps
from timeit import default_timer as timer
import warnings

import generate_plankton as gp
import plankton_growth as pg

def focal_species(N_focal, ind_focal, ti, envi, species_order = None,
                  N_mid = None, n_grid = int(1e3), min_value = 1e-5,
                  tol = 1e-5):
    # number of nonfocal species
    r_nonfoc = sum(~ind_focal)
    # index of nonfocal species
    index_nonfoc = np.arange(len(ind_focal))[~ind_focal]
    
    # test whether species can grow at lowest density
    N_test = np.empty(len(ind_focal))
    N_test[ind_focal] = N_focal
    N_test[~ind_focal] = min_value
    if (pg.per_cap_plankton_growth(N_test, ti, envi)[~ind_focal]<tol).all():
        # predation to strong, return 0 density
        return np.full(r_nonfoc, min_value), "equi"
    
    # ifthe problem is one-dimensional, use brentq
    if sum(~ind_focal) == 1:
        # one dimensional problem, use brentq algorithm
        ind_nonfoc = np.arange(len(ind_focal))[~ind_focal]
        try:
            return brentq(lambda N: pg.per_cap_plankton_growth(
                np.insert(N_focal, ind_nonfoc, N), ti, envi)[ind_nonfoc],
                min_value, 1e10), "equi"
        except ValueError:
            return min_value # predation to strong for survival of species
    
    ##########################################################################
    # start grid search
    
    # determine grid for initial search
    if N_mid is None: # center of grid
        N_mid = np.sqrt(ti["N_star_P_res"])
    num = int(n_grid**(1/r_nonfoc)) # number of grid points
    N_opt = N_mid.copy()
    
    # boundaries of grid are N_mid +- dN
    dN = np.log(N_opt) # creates a grid from 1-N_opt**2
    N_range = np.inf # to start while loop
    
    while N_range > 1:
        
        # create grid for search, grid in log space
        N, dN = np.linspace(np.log(N_opt) - dN,np.log(N_opt) + dN,
                            num = num, retstep=True)
        
        # change from log to actual densities
        N = np.exp(N)
        N_range = np.amax(N[-1]-N[0]) # extent of grid
        N = np.meshgrid(*(N.T))

        
        N_all = np.full((num**r_nonfoc, len(ind_focal)), np.nan)
        N_all[:,ind_focal] = N_focal # densities of species
        
        # add non-focal species
        for i in range(r_nonfoc):
            N_all[:,index_nonfoc[i]] = N[i].reshape(-1)
        
        # compute growth at every grid point
        growth = pg.per_cap_plankton_growth(N_all, ti, envi)
        # location of minimal growth
        ind_opt = np.argmin(np.amax(np.abs(growth[:,~ind_focal]),
                                        axis = 1))
        N_opt = N_all[ind_opt,~ind_focal]        
        
    # is the optimal density actually an equilibrium?
    if np.amax(np.abs(growth[ind_opt,~ind_focal])) < tol:
        return N_opt, "equi"
    else:
        # remove species one at a time
        if species_order is None:
            species_order = np.arange(r_nonfoc)
        equi_dens = []
        equi_potential = []
        growth_potential = []
        for i in species_order:
            ind_focal_cop = ind_focal.copy()
            
            # remove species from species order, relabel rest of the species
            species_order_update = species_order[species_order != i]
            species_order_update[species_order_update > i] -=1
            
            # set density of ith nonfocal species constant to zero
            ind_focal_cop[index_nonfoc[i]] = True
            N_focal_cop = np.empty(len(ind_focal))
            N_focal_cop[ind_focal] = N_focal # densities of real focal species
            N_focal_cop[index_nonfoc[i]] = 1e-5 # small but non-zero number
            N_focal_cop = N_focal_cop[ind_focal_cop]
            
            # recursive search for equilibrium given the new set of focal sp.
            N_opt, case = focal_species(N_focal_cop, ind_focal_cop, ti, envi,
                                  N_mid = N_mid[np.arange(r_nonfoc) != i],
                                  species_order = species_order_update)
            
            
            N = np.empty(len(ind_focal))
            N[ind_focal_cop] = N_focal_cop
            N[~ind_focal_cop] = N_opt
            # check whether other species has negative growth
            growth = pg.per_cap_plankton_growth(N, ti, envi)
            equi_potential.append(np.insert(N_opt, i, min_value))
            growth_potential.append(growth[:2])
            if growth[index_nonfoc[i]] > tol:
                continue # absent species could invade, not a stable equilibrium
            # is found equilibrium actually stable?
            if np.amax(np.abs(growth[~ind_focal_cop]))<tol:
                # record equilibrium
                equi_dens.append( np.insert(N_opt, i, min_value))
        
        if len(equi_dens) == 1:
            return equi_dens[0], "equi" # only one stable equilibrium
        if len(equi_dens) >1:
            # potential issue, multiple equilibria, return first
            warnings.warn("Multiple equilibria found", RuntimeWarning)
            return equi_dens, "multiple_equi"
        if len(equi_dens) == 0:
            # potnetial isse, no equilibrium found, need cycle
            warnings.warn("No equilibria found", RuntimeWarning)
            return np.full(r_nonfoc, np.nan), "no-equi"
        
def evolve_phyto(t, N_phyto, N_all, ind_phyto, ti, envi):
    N_all[ind_phyto] = N_phyto
    return pg.plankton_growth(N_all, ti, envi)[ind_phyto]

def phyto_conditional_distribution(N_focal, ind_focal, ti, envi,
                                   N_start = None, min_value = 1e-5):
    N = np.empty(len(ind_focal))
    N[ind_focal] =N_focal
    if N_start is None:
        N_start = np.full(sum(~ind_focal), min_value)    
    
    time = [0,5*365]
    # burnin time
    burnin = solve_ivp(evolve_phyto, time, N_start,
                       args = (N, ~ind_focal,ti, envi),
                        method = "LSODA")
    
    # 
    time = [0,10*365]
    distribution = solve_ivp(evolve_phyto, time, burnin.y[:, -1],
                       args = (N, ~ind_focal,ti, envi),
                        method = "LSODA")
    return distribution

def growth_zooplankton(N_zoo, ti, envi):
    # get phytoplankton densities
    ind_zoop = np.array(ti["r_phyto"]*[False] + ti["r_zoo"]*[True])
    N_phyto, case = focal_species(N_zoo, ind_zoop, ti, envi)
    if case == "equi":
        return pg.per_cap_plankton_growth(np.append(N_phyto, N_zoo), ti, envi)[ind_zoop]
    if case == "multiple_equi":
        return pg.per_cap_plankton_growth(np.append(N_phyto[0], N_zoo),
                                          ti, envi)[ind_zoop]
    if case == "no-equi":
        # compute equilibrium distribution of phytoplankton
        distribution = phyto_conditional_distribution(N_zoo, ind_zoop, ti, envi)
        N_all = np.empty((distribution.y.shape[-1], ti["r_phyto"] + ti["r_zoo"]))
        N_all[:,ind_zoop] = N_zoo
        N_all[:,~ind_zoop] = distribution.y.T
        # compute growth at equilibrium distribution
        growth_distribution = pg.per_cap_plankton_growth(N_all,
                                                         ti, envi)[:,ind_zoop]
        # computed weighted average growth rate
        return simps(growth_distribution, x = distribution.t, axis = 0)/distribution.t[-1]
        
if __name__ == "__main__":
    
    n_coms = int(1e4)
    r_phyto = 2
    traits, env = gp.generate_communities(r_phyto, n_coms)
    
    # select species with vastly different sizes to increase niche differences
    ti, envi, i = gp.select_i(traits, env)
    
    itera = 21
    N_zoo = np.exp(np.linspace(np.log(ti["N_star_Z"]/10),
                               np.log(ti["N_star_Z"]*10), itera))
    N_zoo = np.meshgrid(*(N_zoo.T))
    
    ind_zoo = np.array(ti["r_phyto"]*[False] + ti["r_zoo"]*[True])
    
    N_zoo_all = np.array([N_zoo[0].reshape(-1), N_zoo[1].reshape(-1)]).T
    N_phyto_all = np.full(N_zoo_all.shape, np.nan)
    
    start = timer()
    cases = np.empty(len(N_phyto_all), dtype = "object")
    growth_new = np.empty(N_zoo_all.shape)
    for i, N in enumerate(N_zoo_all):
        res, cases[i] = focal_species(N, ind_zoo, ti, envi)
        if cases[i] == "equi":
            N_phyto_all[i] = res.copy()
        growth_new[i] = growth_zooplankton(N, ti, envi)
    print(timer()-start)
    N_all = np.append(N_phyto_all, N_zoo_all, axis = -1)
    growth = pg.per_cap_plankton_growth(N_all, ti, envi)
    zero_growth = growth[:,:ti["r_phyto"]] < 1e-5
    spe_absent = N_all[:,:ti["r_phyto"]] <1e-4
    found_equi = zero_growth | spe_absent
    
    N_phyto_all.shape = itera, itera, -1
    #N_phyto_all[N_phyto_all<=1e-5] = np.nan
        
    ##############################################################################
    # plot results
    fig, ax = plt.subplots(3,2, figsize = (9,9), sharex = True, sharey = True)
    
    extent = np.log([N_zoo[0][0,0], N_zoo[0][-1,-1],
              N_zoo[1][0,0], N_zoo[1][-1,-1]])
    cmap = ax[0,0].imshow(np.log(N_phyto_all[...,0]), origin = "lower", extent = extent,
                   aspect = "auto")
    ax[0,0].set_title("Densities, sp1")
    fig.colorbar(cmap, ax = ax[0,0])
    
    cmap = ax[0,1].imshow(np.log(N_phyto_all[...,1]), origin = "lower", extent = extent,
                   aspect = "auto")
    ax[0,1].set_title("Densities, sp2")
    fig.colorbar(cmap, ax = ax[0,1])
    
    cmap = ax[1,0].imshow(growth[...,-2].reshape(itera,itera), origin = "lower"
                   , extent = extent,
                   aspect = "auto")
    ax[1,0].set_title("Growth, sp1")
    fig.colorbar(cmap, ax = ax[1,0])
    
    cmap = ax[1,1].imshow(growth[...,-1].reshape(itera,itera), origin = "lower"
                   , extent = extent,
                   aspect = "auto")
    ax[1,1].set_title("Growth, sp2")
    fig.colorbar(cmap, ax = ax[1,1])
    
    cmap = ax[2,0].imshow(growth_new[...,-2].reshape(itera,itera), origin = "lower"
                   , extent = extent,
                   aspect = "auto")
    ax[2,0].set_title("Growth, sp1")
    fig.colorbar(cmap, ax = ax[2,0])
    
    cmap = ax[2,1].imshow(growth_new[...,-1].reshape(itera,itera), origin = "lower"
                   , extent = extent,
                   aspect = "auto")
    ax[2,1].set_title("Growth, sp2")
    fig.colorbar(cmap, ax = ax[2,1])
    
    for a in ax.flatten():
        a.plot(*np.log(ti["N_star_Z"]), 'ro', zorder = 3)
#"""