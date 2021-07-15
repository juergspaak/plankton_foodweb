import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


import phytoplankton_traits as pt

fig = plt.figure(figsize = (13,13)) 

def plot(save_id = [0]):
    fig.savefig("PP_slides/PP_phytoplankton_traits_{}.png".format(save_id[0]))
    save_id[0] += 1
    
bins = 20
alpha = 0.5
fs = 24

traits = pt.generate_phytoplankton_traits(1,500)
traits = {key:np.log(traits[key]) for key in traits.keys()}

trait_names = ["Size", "Maximum\ngrowth", "Halfsaturation\nnitrogen",
               "Halfsatuation\nphosphorus", "Halfsaturation\nlight",
               "Nitrogen\nuptake", "Phosphorus\nuptake", "Light\nuptake",
               "Edibility", "Resource\nconcentration"]

traits_shown = ["size_P", "mu_P", "c_n", "R_P"]
trait_names = ["Size", "Maximum\ngrowth", "Nitrogen\nuptake",
               "Resource\nconcentration"]
n = len(traits_shown)
ax = np.empty((n,n),dtype = "object")
for i, trait in enumerate(traits_shown):
    ax[i,i] = fig.add_subplot(n,n,1+i*(n+1))
    ax[i,i].hist(pt.raw_data[trait], color = "orange",
             bins = bins, alpha = alpha, density = True)
    ax[i,i].set_xticks([])
    ax[i,i].set_yticks([])
    ax[i,i].set_title(trait_names[i], fontsize = fs)
    
plot()

for i, trait in enumerate(traits_shown):
    ax[i,i].hist(traits[trait], color = "blue",
             bins = bins, alpha = alpha, density = True)

plot()

for i, trait in enumerate(traits_shown):
    if i== 0:
        continue
    ax[i,0] = fig.add_subplot(n,n,1+i*n)
    ax[i,0].scatter(traits["size_P"], traits[trait], s = 1, color = "blue")
    ax[i,0].set_ylabel(trait_names[i], fontsize = fs+2)
    if i != n-1:
        ax[i,0].set_xticks([])
ax[-1,0].set_xlabel(trait_names[0], fontsize = fs)  

plot() 
   
for i, trait in enumerate(traits_shown):
    if i == 0:
            continue
    for j, traitj in enumerate(traits_shown):
        if j>=i or j==0:
            continue
        ax[i,j] = fig.add_subplot(n,n,1 + i*n + j)
        ax[i,j].scatter(traits[traitj], traits[trait], s = 1, color = "blue")
        ax[i,j].set_yticks([])
        #ax[i,j].set_ylabel(trait_names[i], fontsize = 16)
        if i != n-1:
            ax[i,j].set_xticks([])
        else:
            ax[i,j].set_xlabel(trait_names[j], fontsize = fs-2)
    
plot()

for i, trait in enumerate(traits_shown):
    for j, traitj in enumerate(traits_shown):
        if j>=i:
            continue
        ax[i,j].scatter(pt.raw_data[traitj], pt.raw_data[trait],
                        s = 5, color = "orange")
        
plot()