import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

def plot(save_id = [0]):
    fig.savefig("PP_slides/PP_niche_and_fitness_differences{}.png".format(save_id[0]))
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


###############################################################################
# one community
mu = {"Reference": np.array([0.8,0.6]), "Selectivity": np.array([0, 0.6]),
      "Halfsaturation": np.array([0.8, 0.3])}
col = {"Reference": "blue", "Selectivity": "green", "Halfsaturation": "orange"}

keys = ["Reference", "Halfsaturation", "Selectivity"]
ha = {"Halfsaturation": "left", "Selectivity": "center"}
va = {"Halfsaturation": "center", "Selectivity": "bottom"}

for key in keys:
    plt.plot(*mu[key], 'o', color = col[key], markersize = 15,
             zorder = 3)
    if key != "Reference":
        plt.annotate("", mu[key], mu["Reference"],
                     arrowprops=dict(color = col[key], width = 1, shrink = 0.1))
        plt.text(*((mu["Reference"] + mu[key])/2), "Constant\n" + key,
                 color = "white",
                 fontsize = 24, ha = ha[key], va = va[key], zorder = 4)
    plot()

scale = 100
sig = {"Reference": np.eye(2)/scale, "Halfsaturation": np.eye(2)/scale,
       "Selectivity": np.array([[0.05,0],[0,1]])/scale}

for key in keys:
    plt.scatter(*(np.random.multivariate_normal(mu[key],sig[key], 1000).T),
                s = 4, alpha = 0.5, color = col[key], zorder =3)
    plot()
    

plt.scatter(*(np.random.multivariate_normal([0.3,0.3],
                                            0.1*np.array([[1,-0.8],[-0.8,2]])/scale
                                            , 1000).T),
                s = 4, alpha = 0.5, color = "red", zorder =3)
plot()