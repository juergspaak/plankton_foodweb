import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import viridis

try:
    diversity_org
except NameError:
    from sim_diversity_full_coms import diversity as diversity_org, n_spec_max

diversity = diversity_org.copy()
div_arr = diversity[["r_spec_rel_{}".format(2*i)
                                for i in 1+np.arange(n_spec_max)]].values
fig = plt.figure(figsize = (14,11))
ax_org = fig.add_axes([0.2,0.1, 0.75,0.8])

def plot(save_id = [0]):
    fig.savefig("PP_slides/PP_diversity_full{}.png".format(save_id[0]))
    save_id[0] += 1
    
diversity = diversity[diversity.n_comb<=1]
diversity = diversity[diversity.comb != "size_P"]
diversity = diversity[diversity.comb != "size_Z"]
diversity.index = diversity["comb"]
    
ax_org.set_xlim([0,1.05])
ax_org.set_xticks([0,1])
ax_org.set_xticklabels([0,1], fontsize = 16)
ax_org.set_xlabel("Fraction of communities", fontsize = 24)

loc = np.arange(len(diversity))
diversity["identity"] = np.arange(len(diversity))
labels = diversity.comb
ax_org.set_ylim(loc[[0,-1]]+[-0.5,0.5])



colors = viridis(np.linspace(0,1,n_spec_max))

def plot_bar(ind_name, plot_all = False, mean = True):
    iden = diversity.loc[ind_name,"identity"].values
    print(ind_name)
    ax_org.set_yticks(loc[iden])
    ax_org.set_yticklabels(diversity.loc[ind_name, "full_names"],
                           fontsize = 20)
    
    #ax_mean.set_ylim(ax_org.get_ylim())
    for j in range(n_spec_max):
        ax_org.barh(loc[iden], div_arr[iden,j], left = np.sum(div_arr[iden,:j], axis = 1),
                 color = colors[j])
        if plot_all:
            plot()
    if mean:
        ax_mean.plot(diversity.rel_mean[iden], iden, 'ro')
    plot()

plot_bar([""], True, False)
ax_mean = ax_org.twiny()
ax_mean.set_xlim([0.9,1.5])
ax_mean.set_xticks([1, 1.5])
ax_mean.set_xticklabels([1, 1.5], fontsize = 16, color = "red")
ax_mean.set_yticks(loc)
#ax_mean.set_yticklabels(diversity.full_names, fontsize = 16)
ax_mean.set_xlabel("Average species richness", fontsize = 24,
                   color = "red")
iden = diversity.loc[[""],"identity"].values
ax_mean.plot(diversity.rel_mean[iden], iden, 'ro')
plot()

err = 1e-2
diversity["rel_div"] = diversity.rel_mean/diversity.loc["", "rel_mean"]-1
plot_bar(diversity.index[np.abs(diversity.rel_div)<err].values)
plot_bar(diversity.index[diversity.rel_div>-err].values)

plot_bar(diversity.index.values[1:])
plot_bar(diversity.index.values)