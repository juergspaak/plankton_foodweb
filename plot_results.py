import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    sim_data
except NameError:
    sim_data = pd.read_csv("First_simulation.csv")
    sim_data = sim_data[sim_data.prob_high==0]
    
fig = plt.figure(figsize = (9,9))
ax_res = fig.add_subplot(2,2,1)
cases_r = ["NPL", "NP", "NL", "PL"]


boxs_r = [sim_data[sim_data.lim_factors == case].richness for case in cases_r]
ax_res.boxplot(boxs_r, showmeans = True)
ax_res.set_xticklabels(cases_r)

ax_const = fig.add_subplot(2,2,2)
cases_t = ["None", "mu", "c", "k"]


boxs_t = [sim_data[sim_data.const_traits == case].richness for case in cases_t]
ax_const.boxplot(boxs_t, showmeans = True)
ax_const.set_xticklabels(cases_t)
ax_const.set_ylabel("richness")

ax_comb = fig.add_subplot(2,2,3)

mean_rich  = np.empty((len(cases_t), len(cases_r)))
for i, case_r in enumerate(cases_r):
    for j, case_t in enumerate(cases_t):
        mean_rich[i,j] = np.mean(sim_data[
                (sim_data.const_traits == case_t) &
                (sim_data.lim_factors == case_r)].richness)
        print(case_r, case_t, mean_rich[i,j])

cmap = ax_comb.imshow(mean_rich, origin = "lower")
fig.colorbar(cmap, ax = ax_comb)
ax_comb.set_xticks(np.arange(len(cases_t)))
ax_comb.set_xticklabels(cases_t)

ax_comb.set_yticks(np.arange(len(cases_r)))
ax_comb.set_yticklabels(cases_r)

fig.savefig("Figure_const_traits.pdf")
###############################################################################
env_keys = ["I_in", "N", "P", "loss_rate", "zm"]
sim_data2 = sim_data[(sim_data.const_traits == "None") &
                     (sim_data.lim_factors == "NPL")]
sim_data2 = sim_data.copy()
fig, ax = plt.subplots(2,3, figsize = (9,6))
ax = ax.flatten()
for j, key in enumerate(env_keys):
    ranges,dr = np.linspace(min(sim_data2[key]), max(sim_data2[key]), 21,
                            retstep = True)
    boxs = [sim_data2.richness[(sim_data2[key]>ranges[i]).values &
                       (sim_data2[key] < ranges[i+1]).values]
                for i in range(len(ranges)-1)]
    ax[j].plot(ranges[:-1] -dr/2, [np.mean(box) for box in boxs])
    ax[j].plot(ranges[:-1] -dr/2, [np.percentile(box, 75) for box in boxs])
    ax[j].plot(ranges[:-1] -dr/2, [np.percentile(box, 25) for box in boxs])
    ax[j].set_title(key)
    
fig.tight_layout()
fig.savefig("Figure_env_effects.pdf")