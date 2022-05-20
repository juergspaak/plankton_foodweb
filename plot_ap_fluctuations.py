import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,1, sharex = True, sharey = True)

path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/fluctuations/"

years = np.arange(20)
for key in ["in_phase", "quarter_phase", "off_phase"]:
    try:
        file = np.load(path  + "data_fluctuations_N_P_{}.npz".format(key))
        dens = file["dens"]
    except (FileNotFoundError, KeyError):
            continue
    rich = np.mean(np.sum(dens>0, axis = 2), axis = 0)
    
    ax[0].plot(years, rich[0], label = key)
    ax[1].plot(years, rich[1], label = key)
    
ax[0].legend()



fig, ax = plt.subplots(5,2, sharex = True, sharey = "col", figsize = (9,9))

file = np.load("no_fluct.npz")
years = np.arange(20)
for i, key in enumerate(["N", "P", "I_in", "d", "zm"]):
    for period in [1,10,25,50,100]:
        try:
            file = np.load(path + "data_fluctuations_no_pred_{}_{}.npz".format(key, period))
            dens = file["dens"]
            print(key, period)
        except (FileNotFoundError, KeyError):
            continue
        dens = file["dens"]
        rich = np.mean(np.sum(dens>0, axis = 2), axis = 0)
        
        ax[i, 0].plot(years, rich[0], label = period)
        ax[i, 1].plot(years, rich[1], label = period)
    
ax[0,0].legend()

file = np.load(path + "data_fluctuations_no_pred_{}_{}.npz".format("N", 100))
file2 = np.load(path + "data_fluctuations_no_pred_{}_{}.npz".format("N", 10))

surv = file["dens"][:,0]>0
surv2 = file2["dens"][:,0]>0
# have the same traits for all species
#for key in file.files:
#    print(key, np.all(file[key] == file2[key]))
    
    
plt.figure()
plt.plot(years, np.mean(np.any(surv != surv2,axis = 2), axis = 0))
plt.axhline(0, color = "k")
