from plankton_growth import convert_ode_to_log, plankton_growth
import numpy as np
import plankton_growth as pg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import generate_plankton as gp
from timeit import default_timer as timer

r_phyto = 4
traits, env = gp.generate_communities(r_phyto, int(1e4))

err = 1e-1

time = [0,1000]
stable = []
start = timer()
for i in range(min(traits["n_coms"], 20)):
    ti, envi, i = gp.select_i(traits, env, i)
    Ni = np.append(ti["N_star_P"], ti["N_star_Z"], axis = -1)
    Ni *= 1 + np.random.uniform(-err, err, Ni.shape)

    soli = solve_ivp(lambda t, logN: pg.convert_ode_to_log(logN, ti, envi), time, np.log(Ni),
                    method = "LSODA")
    stable.append(np.amin(soli.y)>np.log(1e-5))
    print(i, stable[-1])
    if True or stable[-1]:
        plt.figure()
        plt.plot(soli.t, soli.y[:ti["r_phyto"]].T, '-')
        plt.plot(soli.t, soli.y[ti["r_phyto"]:].T, '--')
        plt.show()
    
print(sum(stable), timer()-start)
