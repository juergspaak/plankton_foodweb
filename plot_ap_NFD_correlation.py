import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


traits = ["c_Z", "k_Z", "e_P", "R_P", "h_zp"]
path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"
n_prec = 9
# with no changes in traits
file = "sim_NFD_mean_combined_{}_{}.npz".format("{}", n_prec//2)

NFD_all = [[],[]]
for trait in traits:
    data = np.load(path + file.format(trait))
    NFD_all[0].append(data["ND"][np.isfinite(data["ND"][:,1])])
    NFD_all[1].append(data["F"][np.isfinite(data["F"][:,1])])
    
NFD_all = np.array([np.concatenate(x) for x in NFD_all])
a = NFD_all.copy()
b = NFD_all.copy()
a = np.sort(a[1], axis = 1)
NFD_all = NFD_all[:, np.arange(NFD_all.shape[1])[:,np.newaxis], np.argsort(NFD_all[1],axis = 1)]

fig = plt.figure(figsize = (9,9))
s = 2
alpha = 0.2
q = [5,95]
ax = [fig.add_subplot(223), fig.add_subplot(224)]
for i in range(2):
    ax[i].scatter(NFD_all[0,:,i], NFD_all[1,:,i], s = s, alpha = alpha)

    ax[i].set_xlim(np.nanpercentile(NFD_all[0,:,i], q))
    ax[i].set_ylim(np.nanpercentile(NFD_all[1,:,i], q))
    ax[i].set_ylabel("Fitness differences")
    ax[i].set_xlabel("Niche differecnes")

data_mean = pd.read_csv("data/assembly_mean.csv")
NFD_mean = pd.read_csv("data/NFD_mean.csv")

data_corr = pd.read_csv("data/assembly_corr.csv")
NFD_corr = pd.read_csv("data/NFD_corr.csv")

data_corr["trait"] = data_corr["tradeoff"]
NFD_corr["trait"] = NFD_corr["corr"]

data_rich = [data_mean, data_corr]
data_NFD = [NFD_mean, NFD_corr]
style = [dict(marker = "o", color = "b", ls = ""),
         dict(marker = "^", color = "red", ls = "")]
label = ["Changes in mean", "Changes in correlations"]

ax.append(fig.add_subplot(211))
for i, change in enumerate(data_NFD):
    ax[2].plot(data_NFD[i].ND_mean_1, data_NFD[i].FD_median_1, **style[i],
             label = label[i])
ax[2].legend()

ax[2].set_xlabel("Niche differences")
ax[2].set_ylabel("Fitness differences")

ax[0].set_title("B", loc = "left")
ax[1].set_title("C", loc = "left")
ax[2].set_title("A", loc = "left")

ax[0].set_title("Inferior competitor")
ax[1].set_title("Superior competitor")
ax[2].set_title("Average of niche and fitness differences")

fig.tight_layout()
fig.savefig("Figure_ap_association_NFD.pdf")