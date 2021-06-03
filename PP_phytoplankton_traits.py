import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


import phytoplankton_traits as pt

fig = plt.figure(figsize = (10,10)) 

def plot(save_id = [0]):
    fig.savefig("PP_slides/PP_phytoplankton_traits_{}.png".format(save_id[0]))
    save_id[0] += 1
    
bins = 20

col_simple = "purple"
col_emp = "orange"
col_full = "blue"
    
traits = pt.generate_phytoplankton_traits(10,1000)
traits = {key: np.log(traits[key].flatten()) for key in traits.keys()}

itera = int(1e4)
ax_size = fig.add_subplot(331)

ax_size.hist(pt.raw_data["size_P"], density = True, alpha = 1,
             label = "empirical", color = col_emp)
ax_size.set_xlabel("log(volume)")
ax_size.set_ylabel("frequency")
ax_size.legend(fontsize = 8)
plot()

ax_size.hist(traits["size_P"], bins = bins, density=True, label = "simulation",
             color = col_full, alpha = 0.5)
plot()

ax_size_t1 = fig.add_subplot(334)
t1 = "k_p"
size_id = 0
t1_id = t1 == pt.trait_names
size_range = np.percentile(traits["size_P"], [5,95])
alpha = pt.mean_traits[t1_id] - pt.mean_traits[size_id]*pt.gaussians.loc[t1, "beta_min"]
t1_simple = (size_range*pt.gaussians.loc[t1, "beta_min"] + alpha)
ax_size_t1.plot(size_range, t1_simple, color = col_simple)
ax_size_t1.set_xlabel("log(volume)")
ax_size.set_xlabel("")
ax_size_t1.set_ylabel("log({})".format(t1))
plot()


ax_t1 = fig.add_subplot(335)
ax_t1.hist(traits["size_P"]*pt.gaussians.loc[t1, "beta_min"] + alpha,
           bins = bins, density = True, color = col_simple,
           alpha = 0.5)
ax_t1.set_xlabel("log({})".format(t1))
plot()

ax_t1.hist(pt.raw_data[t1], bins = bins, density = True, alpha = 0.5,
           color = col_emp)
plot()

ax_size_t1.scatter(traits["size_P"], traits[t1],
                   s = 2, color = col_full, alpha = 0.1)
ax_t1.hist(traits[t1], bins = bins, color = col_full, density = True,
           alpha = 0.5)
plot()

ax_size_t1.scatter(pt.phyto_traits["size_P"], pt.phyto_traits[t1], 
                   color = col_emp)
plot()


###############################################################################
# add another trait
t2 = "k_n"
ax_size_t2 =fig.add_subplot(337)
ax_size_t2.set_xlabel("log(volume)")
ax_size_t1.set_xlabel("")
ax_size_t2.set_ylabel("log({})".format(t2))

t1_id = t1 == pt.trait_names
size_range = np.percentile(traits["size_P"], [5,95])
alpha = pt.mean_traits[t1_id] - pt.mean_traits[size_id]*pt.gaussians.loc[t1, "beta_min"]

ax_size_t2.scatter(traits["size_P"], traits[t2], 
                   s = 2, color = col_full, alpha = 0.1)

ax_t2 = fig.add_subplot(339)
ax_t2.hist(traits[t2], bins = bins, color = col_full, density = True, alpha = 0.5)
ax_t2.set_xlabel("log({})".format(t2))
ax_t2.hist(pt.raw_data[t2], alpha = 0.5, color = col_emp, density = True)
ax_size_t2.scatter(pt.phyto_traits["size_P"], pt.phyto_traits[t2], 
                   color = col_emp)
plot()

ax_t1_t2 = fig.add_subplot(338)
ax_t1_t2.scatter(traits[t1], traits[t2], color = col_full, alpha = 0.1, s = 2)
ax_t1_t2.set_xlabel(ax_t1.get_xlabel())
ax_t1.set_xlabel("")

plot()
ax_t1_t2.scatter(pt.phyto_traits[t1], pt.phyto_traits[t2], 
                   color = col_emp)
plot()



