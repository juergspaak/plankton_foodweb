import numpy as np
import matplotlib.pyplot as plt

import gaussian_traits as gt

def plot(i = [1]):
    fig.savefig("PP_slides/PP_species_gen_{}.png".format(i[0]))
    i[0] += 1
    
"""    
for l in ax.lines:
    l.set_alpha(0.3)
    l.set_linewidth(lw_old)
    """
bins = 20

col_simple = "purple"
col_emp = "orange"
col_full = "blue"
    
fig = plt.figure(figsize = (10,10))
itera = int(1e4)
size = np.random.normal(gt.mean_size, gt.std_size, itera)
ax_size = fig.add_subplot(331)
ax_size.hist(size, bins = bins, density=True, label = "simulation",
             color = col_full)
ax_size.hist(gt.raw_data["size"], density = True, alpha = 0.5,
             label = "empirical", color = col_emp)
ax_size.set_xlabel("log(volume)")
ax_size.set_ylabel("frequency")
ax_size.legend(fontsize = 8)
plot()

ax_size_t1 = fig.add_subplot(334)
t1 = "k_p"
t1_simple = (size*gt.gaussians.loc[t1, "beta_min"] +
                gt.gaussians.loc[t1, "alpha"])
ax_size_t1.scatter(size, t1_simple, s = 2, color = col_simple)
ax_size_t1.set_xlabel("log(volume)")
ax_size.set_xlabel("")
ax_size_t1.set_ylabel("log({})".format(t1))
plot()

ax_t1 = fig.add_subplot(335)
ax_t1.hist(t1_simple, bins = bins, density = True, color = col_simple,
           alpha = 0.5)
ax_t1.set_xlabel("log({})".format(t1))
plot()

ax_t1.hist(gt.raw_data[t1], bins = bins, density = True, alpha = 0.5,
           color = col_emp)
plot()

t1_full = t1_simple + np.random.normal(0, gt.gaussians.loc[t1, "std_err"], itera)
ax_size_t1.scatter(size, t1_full, s = 2, color = col_full, alpha = 0.1)
ax_t1.hist(t1_full, bins = bins, color = col_full, density = True, alpha = 0.5)
plot()

t1_raw = "k_p"
ax_size_t1.scatter(gt.phyto_traits["size"], gt.phyto_traits[t1_raw], 
                   color = col_emp)
plot()


###############################################################################
# add another trait
t2 = "k_n"
ax_size_t2 =fig.add_subplot(337)
ax_size_t2.set_xlabel("log(volume)")
ax_size_t1.set_xlabel("")
ax_size_t2.set_ylabel("log({})".format(t2))

t2_full = (size*gt.gaussians.loc[t2, "beta_min"] +
                gt.gaussians.loc[t2, "alpha"]
                + np.random.normal(0, gt.gaussians.loc[t2, "std_err"], itera))
ax_size_t2.scatter(size, t2_full, s = 2, color = col_full, alpha = 0.1)

ax_t2 = fig.add_subplot(339)
ax_t2.hist(t2_full, bins = bins, color = col_full, density = True, alpha = 0.5)
ax_t2.set_xlabel("log({})".format(t2))
ax_t2.hist(gt.raw_data[t2], alpha = 0.5, color = col_emp, density = True)
ax_size_t2.scatter(gt.phyto_traits["size"], gt.phyto_traits[t2], 
                   color = col_emp)
plot()

ax_t1_t2 = fig.add_subplot(338)
ax_t1_t2.scatter(t1_full, t2_full, color = col_full, alpha = 0.1, s = 2)
ax_t1_t2.set_xlabel(ax_t1.get_xlabel())
ax_t1.set_xlabel("")

plot()
ax_t1_t2.scatter(gt.phyto_traits[t1_raw], gt.phyto_traits[t2], 
                   color = col_emp)
plot()


