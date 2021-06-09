import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import linregress

uye = pd.read_csv("Uye_1989.csv")

fig, ax = plt.subplots(3,1, figsize = (7,7))
for spec in list(set(uye.Species)):
    
    datc = uye[uye.Species == spec].copy()
    ax[0].scatter(datc["Length"], datc.N, s = 1, label = spec)
    ax[1].scatter(datc["Length"], datc.C, s = 1, label = spec)
    ax[2].scatter(datc["Length"], datc.DW, s = 1, label = spec)

range_l = np.array([2.5,20])
s, i, r, p, std = linregress(np.log(uye["Length"]), np.log(uye["N"]))
ax[0].plot(range_l, np.exp(np.log(range_l)*s + i), 'r-', label = np.round(r**2,3))
ax[0].legend([np.round(r**2,3),
              np.round(s,3),
              np.round(s-std*2,3)])

s, i, r, p, std = linregress(np.log(uye["Length"]), np.log(uye["C"]))
ax[1].plot(range_l, np.exp(np.log(range_l)*s + i), 'r-', label = np.round(r**2,3))
ax[1].legend([np.round(r**2,3), np.round(s,3)])

s, i, r, p, std = linregress(np.log(uye["Length"]), np.log(uye["DW"]))
ax[2].plot(range_l, np.exp(np.log(range_l)*s + i), 'r-', label = np.round(r**2,3))
ax[2].legend([np.round(r**2,3), np.round(s,3)])

ax[0].loglog()
ax[1].loglog()
ax[2].loglog()

ax[0].set_ylabel("Nitrogen [\mug]")
ax[1].set_ylabel("Carbon [\mug]")
ax[2].set_ylabel("DW [\mug]")
ax[2].set_xlabel("Length [\mum]")  
