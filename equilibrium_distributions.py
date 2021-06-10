import numpy as np
import matplotlib.pyplot as plt
import warnings
from timeit import default_timer as timer

import generate_plankton as gp

itera = 1000

try:
    t_coex
except:
    start = timer()
    n_coms = int(1e5)
    r_phyto = 2

    traits = gp.generate_plankton(r_phyto, n_coms, evolved_zoop=True)
    env = gp.generate_env(n_coms)
    traits = gp.community_equilibrium(traits, env)
    
    ind1 = np.isfinite(traits["N_star_P"]).all(axis = -1)
    ind2 = np.isfinite(traits["N_star_Z"]).all(axis = -1)
    ind = ind1 & ind2
    n_com_new = np.sum(ind)
    print(n_com_new/n_coms, n_com_new)
    
    t_coex = {key:traits[key][ind] for key in gp.select_keys(traits)}
    env_coex = {key: env[key][ind] for key in env.keys()}
    t_coex = gp.phytoplankton_equilibrium(t_coex, env_coex)
    
    t_all = gp.generate_plankton(r_phyto, n_com_new, evolved_zoop=True)
    env_all = gp.generate_env(n_com_new)
    t_all = gp.community_equilibrium(t_all, env_all)
    t_all = gp.phytoplankton_equilibrium(t_all, env_all)
    print("time used: ", timer()-start)
    with warnings.catch_warnings(record =True):
        t_coex_log = {key: np.log(t_coex[key].flatten())[:itera]
                      for key in t_coex.keys()}
        t_all_log = {key: np.log(t_all[key].flatten())[:itera]
                     for key in gp.select_keys(t_all)}
    t_coex_log["s_zp"] = np.exp(t_coex_log["s_zp"])
    t_all_log["s_zp"] = np.exp(t_all_log["s_zp"])
   
bins = 30

###############################################################################
# differences in one-dimensional traits

fig, ax = plt.subplots(5,5, figsize =(12,12))
ax = ax.flatten()

single_traits = (list(gp.pt.trait_names) + gp.zt.zoop_traits 
                    + ["h_zp", "s_zp", "N_star_P", "N_star_Z",
                       "R_star_n", "R_star_p", "N_star_P_n",
                       "N_star_P_p", "N_star_P_l", "N_star_P_res", "k_Z"])
for i,trait in enumerate(single_traits):
    ax[i].hist(t_all_log[trait].flatten(), bins = bins, density = True)
    ax[i].hist(t_coex_log[trait].flatten(), bins = bins, density = True,
               alpha = 0.5, label = "coexisting")
    ax[i].set_xlabel(trait)

ax[0].legend()    
fig.tight_layout()

fig.savefig("Figure_histogram_equi_distribution.pdf")
###############################################################################
# differences in two dimensional trait distributions
# numerical analysis
n_digit = 10
phyto_coex = np.array([t_coex[trait].flatten() for trait in gp.pt.trait_names])
cov_coex = np.round(np.cov(np.log(phyto_coex)), n_digit)
cor_coex = np.round(cov_coex/np.sqrt(np.diag(cov_coex)*np.diag(cov_coex)[:,np.newaxis]), n_digit)
mean_coex = np.round(np.mean(np.log(phyto_coex), axis = 1), n_digit)

phyto_test = np.array([t_all[trait].flatten() for trait in gp.pt.trait_names])
cov_test = np.round(np.cov(np.log(phyto_test)), n_digit)
cor_test = np.round(cov_test/np.sqrt(np.diag(cov_test)*np.diag(cov_test)[:,np.newaxis]), n_digit)
mean_test = np.round(np.mean(np.log(phyto_test), axis = 1), n_digit)

print(np.round((mean_coex-gp.pt.mean_traits)/gp.pt.mean_traits,3))

rel_diff_cov = (cov_coex-gp.pt.cov_matrix)/gp.pt.cov_matrix
cor_real = gp.pt.cov_matrix
cor_real = cor_real/np.sqrt(np.diag(cor_real)*np.diag(cor_real)[:,np.newaxis])

rel_diff_cor = (cor_coex-cor_real)/cor_real
abs_diff_cor = (cor_coex-cor_real)

fig, ax = plt.subplots(2,2, figsize = (11,7))

cmap = ax[0,0].imshow(rel_diff_cor.values)
fig.colorbar(cmap, ax = ax[0,0])

ax[0,0].set_xticks(np.arange(len(gp.pt.trait_names)))
ax[0,0].set_xticklabels(gp.pt.trait_names, rotation= 90)

ax[0,0].set_yticks(np.arange(len(gp.pt.trait_names)))
ax[0,0].set_yticklabels(gp.pt.trait_names)

# compare to accuracy of test traits
cmap1 = ax[0,1].imshow(abs_diff_cor)
fig.colorbar(cmap1, ax = ax[0,1])

ax[0,1].set_xticks(np.arange(len(gp.pt.trait_names)))
ax[0,1].set_xticklabels(gp.pt.trait_names, rotation= 90)

ax[0,1].set_yticks(np.arange(len(gp.pt.trait_names)))
ax[0,1].set_yticklabels(gp.pt.trait_names)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# one dimensional errors
ax[1,0].plot(np.arange(len(mean_coex)),
             (mean_coex - gp.pt.mean_traits)/gp.pt.mean_traits, 'o')
ax[1,0].plot(np.arange(len(mean_coex)),
             np.diag(rel_diff_cov), '^')
ax[1,0].plot(np.arange(len(mean_coex)),
             rel_diff_cor["size_P"], '*')

ax[1,1].plot(np.arange(len(mean_coex)),
             (mean_coex - gp.pt.mean_traits), 'o', label = "mean")
ax[1,1].plot(np.arange(len(mean_coex)),
             np.diag(cov_coex - gp.pt.cov_matrix), '^', label = "variance")
ax[1,1].plot(np.arange(len(mean_coex)),
             abs_diff_cor["size_P"], '*', label = "allometric scaling")

ax[1,1].set_xticks(np.arange(len(gp.pt.trait_names)))
ax[1,1].set_xticklabels(gp.pt.trait_names, rotation= 90)
ax[1,1].set_ylabel("Absolute Error")

ax[1,0].set_xticks(np.arange(len(gp.pt.trait_names)))
ax[1,0].set_xticklabels(gp.pt.trait_names, rotation= 90)
ax[1,0].set_ylabel("Relative Error")

ax[1,1].legend()

fig.tight_layout()

fig.savefig("Figure_quantitative_equilibrium_distributions.pdf")

###############################################################################
# zooplankton trait distribution
zoop_traits = gp.zt.zoop_traits

n = len(zoop_traits)
fig, ax = plt.subplots(n,n,figsize = (9,9), sharex = "col", sharey = "row")
 
s = 5
for i, keyi in enumerate(zoop_traits):
    for j, keyj in enumerate(zoop_traits):
        if i>j:
            ax[i,j].scatter(t_all_log[keyj], t_all_log[keyi], s = s, alpha = 0.1,
                            color = "blue")
            ax[i,j].scatter(t_coex_log[keyj], t_coex_log[keyi], s = s, alpha = 0.1,
                            color = "orange")
    ax_hist = fig.add_subplot(n,n,1 + (n+1)*i)
    ax_hist.hist(t_all_log[keyi], bins, density = True, color = "blue")
    ax_hist.hist(t_coex_log[keyi], bins, density = True, color = "orange", 
                 alpha = 0.5)
    ax_hist.set_xticklabels([])
    ax_hist.set_yticklabels([])

    ax[i,0].set_ylabel(zoop_traits[i])
    ax[-1,i].set_xlabel(zoop_traits[i])
    

ax[0,0].set_ylim(ax[0,0].get_xlim())
ax[-1,-1].set_xlim(ax[-1,-1].get_ylim())

fig.savefig("Figure_equi_zooplankton_traits.pdf")

###############################################################################
# zooplankton trait distribution
phyto_traits = gp.pt.trait_names

n = len(phyto_traits)
fig, ax = plt.subplots(n,n,figsize = (9,9), sharex = "col", sharey = "row")
 
s = 5
for i, keyi in enumerate(phyto_traits):
    for j, keyj in enumerate(phyto_traits):
        if i>j:
            ax[i,j].scatter(t_all_log[keyj], t_all_log[keyi], s = s, alpha = 0.1,
                            color = "blue")
            ax[i,j].scatter(t_coex_log[keyj], t_coex_log[keyi], s = s, alpha = 0.1,
                            color = "orange")
    ax_hist = fig.add_subplot(n,n,1 + (n+1)*i)
    ax_hist.hist(t_all_log[keyi], bins, density = True, color = "blue")
    ax_hist.hist(t_coex_log[keyi], bins, density = True, color = "orange", 
                 alpha = 0.5)
    ax_hist.set_xticklabels([])
    ax_hist.set_yticklabels([])

    ax[i,0].set_ylabel(phyto_traits[i])
    ax[-1,i].set_xlabel(phyto_traits[i])
    

ax[0,0].set_ylim(ax[0,0].get_xlim())
ax[-1,-1].set_xlim(ax[-1,-1].get_ylim())

fig.savefig("Figure_equi_phytoplankton_traits.pdf")