import numpy as np
import pandas as pd

n_prec = 9
add_mean = np.linspace(-1, 1, n_prec)
cutoff = 750
##############################################################################
# changes in mean

traits = ["c_Z", "k_Z", "e_P", "R_P", "h_zp"]

cols = ["{}_{}_{}".format(x, y, z) for x in ["ND", "FD"] for y in ["mean", "median"]
        for z in range(1, 3)]
df_mean = pd.DataFrame(np.nan, index=np.arange(n_prec * len(traits)),
                       columns=["trait", "change", "n", "portion"] + cols)

counter = 0
path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"
file = "sim_NFD_mean_combined_{}_{}.npz"

for i in range(n_prec):
    for trait in traits:
        data = np.load(path + file.format(trait, i))
        
        ND = np.sort(data["ND"], axis=1)
        FD = np.sort(data["F"], axis=1)

        ind = np.isfinite(ND[:, 0])

        df_mean.loc[counter] = [trait, add_mean[i],
                                np.sum(ind),
                                np.mean(ind),
                                *np.nanmean(ND[ind][:cutoff], axis=0),
                                *np.nanmedian(ND[ind][:cutoff], axis=0),
                                *np.nanmean(FD[ind][:cutoff], axis=0),
                                *np.nanmedian(FD[ind][:cutoff], axis=0)]
        counter += 1

df_mean.to_csv('data/NFD_mean.csv', index=False)
#"""

##############################################################################
# changes in correlation

gleaner = ["c_Z:m_Z", "c_Z:k_Z", "c_Z:mu_Z", "c_n:k_n", "c_p:k_p"]#, "k_Z:mu_Z"]
defense = ["R_P:mu_P", "R_P:k_n"]
super_resource = ["R_P:a", "R_P:e_P"]

corrs = gleaner + defense + super_resource
tradeoff = [gleaner, defense, super_resource]
file = "sim_NFD_corr_combined_{}_{}_{}.npz"

df_corr = pd.DataFrame(np.nan, index = np.arange(n_prec*len(traits)),
                    columns = ["corr", "change","case", "n", "portion"] + cols)
counter = 0

for corr in corrs:
    case = np.argmax([corr in x for x in tradeoff])
    for i in range(n_prec):
        
        data = np.load(path + file.format(*corr.split(":"), i))
        
        ND = np.sort(data["ND"], axis=1)
        FD = np.sort(data["F"], axis=1)

        ind = np.isfinite(ND[:, 0])
        
        df_corr.loc[counter] = [corr, data["corr"], int(case),
                                np.sum(ind),
                                np.mean(ind),
                                *np.nanmean(ND[ind][:cutoff], axis = 0),
                                *np.nanmedian(ND[ind][:cutoff], axis = 0),
                                *np.nanmean(FD[ind][:cutoff], axis = 0),
                                *np.nanmedian(FD[ind][:cutoff], axis = 0)]
        counter += 1
        
df_corr.to_csv("data/NFD_corr.csv", index = False)
#"""

##############################################################################
# reference values

traits = ["c_Z", "k_Z", "e_P", "R_P", "h_zp"]

cols = ["{}_{}_{}".format(x, y, z) for x in ["ND", "FD"] for y in ["mean", "median"]
        for z in range(1, 3)]
df_mean = pd.DataFrame(np.nan, index=np.arange(n_prec * len(traits)),
                       columns=["trait", "change", "n", "portion"] + cols)

counter = 0
path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"
file = "sim_NFD_mean_combined_{}_{}.npz"
i = n_prec // 2
ND, FD = np.empty((2, 0, 2))
for trait in traits:
    try:
        data = np.load(path + file.format(trait, i), allow_pickle=True)
        ND = np.append(ND, data["ND"], axis=0)
        FD = np.append(FD, data["F"], axis=0)
    except (FileNotFoundError, KeyError):
        continue

ND = np.sort(ND, axis=1)
FD = np.sort(FD, axis=1)

ind = np.isfinite(ND[:, 0])

ND = ND[ind][:(np.sum(ind) // cutoff) * cutoff].reshape((cutoff, -1, 2))
FD = FD[ind][:(np.sum(ind) // cutoff) * cutoff].reshape((cutoff, -1, 2))

df_ref = pd.DataFrame(np.nan, index=range(FD.shape[1]), columns=cols)

df_ref[["ND_mean_1", "ND_mean_2"]] = np.nanmean(ND, axis=0)
df_ref[["FD_mean_1", "FD_mean_2"]] = np.nanmean(FD, axis=0)

df_ref[["ND_median_1", "ND_median_2"]] = np.nanmedian(ND, axis=0)
df_ref[["FD_median_1", "FD_median_2"]] = np.nanmedian(FD, axis=0)

df_ref.to_csv("data/NFD_ref.csv", index=False)
#"""
