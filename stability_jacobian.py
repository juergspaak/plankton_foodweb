"""
find a localy stable plankton community
"""

from plankton_growth import convert_ode_to_log, plankton_growth
import numpy as np
import plankton_growth as pg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import generate_plankton as gp
from timeit import default_timer as timer
   
r_phyto = 4
traits, env = gp.generate_communities(r_phyto, int(1e5))
    
N = np.append(traits["N_star_P"], traits["N_star_Z"], axis = -1)

df = np.full((traits["n_coms"], 2, traits["r_phyto"] + traits["r_zoo"],
               traits["r_phyto"] + traits["r_zoo"]), np.nan)
for i in range(traits["r_phyto"] + traits["r_phyto"]):
    dN = N[:,i]*1e-7
    for ind_j,j in enumerate([-1,1]):
        N_dN = N.copy()
        N_dN[:,i] += j*dN
        df[:,ind_j,:,i] = plankton_growth(N_dN, traits, env)/dN[:,np.newaxis]
            
df[:,:,traits["r_phyto"]:, traits["r_phyto"]:] = 1e-16

eigvals = np.linalg.eigvals(df[:,1])
stable = np.amax(np.real(eigvals), axis = -1)<0.0

print(np.sum(stable), len(stable))


ti, envi, i = gp.select_i(traits, env, np.argmax(stable))
    
# add environmental noise
N = np.append(ti["N_star_P"], ti["N_star_Z"], axis = 0)
err = 1e-0
N *= np.random.uniform(1-err,1+err,N.shape)
   

time = [0,1000]

fig, ax = plt.subplots(2, sharex = True)


sol = solve_ivp(lambda t, logN: convert_ode_to_log(logN, ti, envi),
                     time, np.log(N), method = "LSODA")   

ax[0].semilogy(sol.t, np.exp(sol.y[:r_phyto]).T, '-')
ax[1].semilogy(sol.t, np.exp(sol.y[r_phyto:]).T, '--')