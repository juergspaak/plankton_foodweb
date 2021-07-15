import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.DataFrame()
for i in [0,3,4,5]:
    data_new = pd.read_csv("NFD_data_{}.csv".format(i))
    data = data.append(data_new, ignore_index=True)

data.to_csv("NFD_data_all.csv", index = False)
data = pd.read_csv("NFD_data_all_preselect.csv")
# remove intraspecific facilitation
data = data[np.all(data[["FD0", "FD1"]]<1, axis = 1)]
data["FD_inf"] = 1-1/(1-np.nanmin(data[["FD0", "FD1"]], axis = 1))
data = data[data["FD_inf"] >0]

data.loc[data.trait_comb != data.trait_comb, "trait_comb"] = "Ref"


plt.style.use('dark_background')

def plot(save_id = [0]):
    fig.savefig("PP_slides/PP_niche_and_fitness_differences_computed{}.png".format(save_id[0]))
    save_id[0] += 1

fig = plt.figure(figsize =(12,7))


lw = 5
xlim = [-0.5,1.5]
ylim = [-0.5,1]
col = "grey"
plt.plot(xlim, [0,0], linewidth = lw, color = col)
plt.plot([0,0], ylim, linewidth = lw, color = col)
plt.plot([1,1], ylim, linewidth = lw, color = col)
plt.plot(xlim, xlim, '--', linewidth = lw, color = col)
plt.xticks([0,1], fontsize = 16)
plt.yticks([0,1], fontsize = 16)
plt.gca().set_frame_on(False)
plt.axis([-0.5,1.5,-0.5,1])

plt.xlabel("Niche differences", fontsize = 20)
plt.ylabel("Fitness differences", fontsize = 20, rotation = 0, ha = "right")
fig.tight_layout()
plot()

n_coms = 500
s = 5
ind = (data.trait_comb == "Ref").values
plt.scatter(data.loc[ind,"ND0"][:n_coms], data.loc[ind,"FD_inf"][:n_coms],
            s = 5, color = "blue")
plt.plot(np.nanmedian(data.loc[ind, "ND0"]),
         np.nanmedian(data.loc[ind, "FD_inf"]), 'o', color = "blue",
         markersize = 15)
print(np.cov(data.loc[ind, ["ND0", "FD_inf"]].T))
plot()


ind = data.trait_comb == "s_zp"
plt.scatter(data.loc[ind,"ND0"][:n_coms], data.loc[ind,"FD_inf"][:n_coms],
            s = 5, color = "green")
plt.plot(np.nanmedian(data.loc[ind, "ND0"]),
         np.nanmedian(data.loc[ind, "FD_inf"]), 'o', color = "green",
         markersize = 15)
print(np.cov(data.loc[ind, ["ND0", "FD_inf"]].T))
plot()

ind = data.trait_comb == "k_n"
plt.scatter(data.loc[ind,"ND0"][:n_coms], data.loc[ind,"FD_inf"][:n_coms],
            s = 5, color = "orange")
plt.plot(np.nanmedian(data.loc[ind, "ND0"]),
         np.nanmedian(data.loc[ind, "FD_inf"]), 'o', color = "orange",
         markersize = 15)
print(np.cov(data.loc[ind, ["ND0", "FD_inf"]].T))
plot()