# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:13:33 2021

@author: Juerg Spaak
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from timeit import default_timer as timer

import phyto_growth as pg

itera = 100
r_spec = 2
traits = pg.generate_phytoplankton(r_spec,itera)

z_max = 10 # meters
z_steps = 20 # depth precision
z, dz = np.linspace(0,z_max, z_steps, retstep = True)

t_end = 1000 # time integration length
t_precision = 2000
time = np.linspace(0, t_end, t_precision)

# environmental data
P_supply = np.random.uniform(0,3, itera)
N_supply = np.random.uniform(0,30, itera)
I_in_supply = np.random.uniform(30,200, itera)
loss_rate = np.random.uniform(0.05, 0.3, itera)
zm = np.random.uniform(1,10, itera)
diff_const = np.random.uniform(1,100,itera) # dilution rate

conv = np.array([1,-2,1]) # convolution kernel for second order differential

def growth(Y, time, t, env, Y_copy):
    # compute spatial explicit growth
    
    Y_copy[:,1:-1] = np.reshape(Y.copy(), (-1,z_steps,1))
    Y_copy[:,0] = Y_copy[:,1] # ensure nothing leaves at the top
    
    X, N, P = Y_copy[:-2,...,0].T, Y_copy[-2], Y_copy[-1]
    X[X<0] = 0
    N[N<0] = 0
    P[P<0] = 0
    
    # ensure boundary conditions
    N[-1] = env["N_supply"]
    P[-1] = env["P_supply"]
    X[-1] = X[-2] # ensure phytoplankton don't leave at the bottom
    
    I = env["I_in"]*np.exp(-np.cumsum(dz*np.sum(t["k"] * X, axis = -1)))
    I = I.reshape(-1,1)
    light_lim = t["mu_l"] * I / (I + t["mu_l"]/t["alpha"])
    
    N_lim = t["mu_n"] * N / (t["k_n"] + N)
    P_lim = t["mu_p"] * P / (t["k_p"] + P)
    
    phyto_growth = np.amin([light_lim, N_lim, P_lim], axis = 0) * X
    
    diffusion = env["diff"]*np.array([np.convolve(x, conv, "same") for x in X.T]).T
    #print(np.sum(diffusion[1:-1], axis = 0))
    
    dX_dt = phyto_growth - t["m"] * X + diffusion
    
    diffusion = env["diff"]*np.convolve(N[:,0], conv, "same")
    dN_dt = diffusion - np.amin([np.sum(X * t["c_n"], axis = -1),
                                 N[:,0]], axis = 0)
    
    diffusion = env["diff"]*np.convolve(P[:,0], conv, "same")
    dP_dt = diffusion - np.amin([np.sum(X * t["c_p"], axis = -1),
                                 P[:,0]], axis = 0)

    Y_copy[:-2,...,0] = dX_dt.T
    Y_copy[-2:,...,0] = [dN_dt, dP_dt]
    
    return Y_copy[:,1:-1].reshape(-1)
X_start = np.zeros((r_spec, z_steps))
X_0 = 1e8
r_specs = np.empty(itera)

def fun(t_ind, sol, env, t):


    Y_copy[:,1:-1] = np.reshape(sol[t_ind].copy(), (-1,z_steps,1))
    Y_copy[:,0] = Y_copy[:,1] # ensure nothing leaves at the top
    X, N, P = Y_copy[:-2,...,0].T, Y_copy[-2], Y_copy[-1]
    N[N<0] = 0
    P[P<0] = 0
    
    # ensure boundary conditions
    N[-1] = env["N_supply"]
    P[-1] = env["P_supply"]
    X[-1] = X[-2] # ensure phytoplankton don't leave at the bottom
    
    I = env["I_in"]*np.exp(-np.cumsum(dz*np.sum(t["k"] * X, axis = -1)))
    I = I.reshape(-1,1)
    light_lim = t["mu_l"] * I / (I + t["mu_l"]/t["alpha"])
    
    N_lim = t["mu_n"] * N / (t["k_n"] + N)
    P_lim = t["mu_p"] * P / (t["k_p"] + P)

    fig, ax = plt.subplots(3,2, figsize = (12,9))
    ax[0,0].plot(time[:t_ind+1], np.sum(sol[:t_ind + 1,:-2], axis = -1))
    ax[0,1].plot(time[:t_ind +1], np.sum(sol[:t_ind + 1,-2:], axis = -1))
    
    ax[1,0].plot(sol[t_ind,:-2].T, z)
    ax[1,1].plot(sol[t_ind,-2:].T, z)
    ax[1,0].invert_yaxis()
    ax[1,1].invert_yaxis()
    
    ax[2,0].plot(N_lim[1:-1], z)
    ax[2,0].plot(P_lim[1:-1], z, '--')
    ax[2,0].plot(light_lim[1:-1], z, ':')
    phyto_growth = np.amin([light_lim, N_lim, P_lim], axis = 0)
    ax[2,1].plot(phyto_growth[1:-1], z)
    ax[2,0].invert_yaxis()
    ax[2,1].invert_yaxis()
    plt.show()
start_all = timer()
for i in range(itera):
    N_start = np.linspace(0, N_supply[i], z_steps)
    P_start = np.linspace(0, P_supply[i], z_steps)
    
    X_start[:,-10:] = X_0/10

    Y_start = np.empty((r_spec + 2, z_steps))
    Y_start[:-2] = X_start
    Y_start[-2:] = [N_start, P_start]
    env = {"I_in": I_in_supply[i], "N_supply": N_supply[i], "P_supply": P_supply[i],
       "diff": diff_const[i]}
    t = {key:traits[key][...,i] for key in traits.keys()}
    Y_copy = np.zeros((r_spec + 2, z_steps + 2,1))
    start = timer()
    sol = odeint(growth, Y_start.reshape(-1), time, args = (t, env, Y_copy))

    sol = sol.reshape((len(time), -1, z_steps))
    
    for j in range(5):
        
        if ((np.abs(sol[-100] - sol[-1])/sol[-1] < 1e-3) |
            (sol[-1] < 1e3)).all():
            continue
        print("reload: ", j)
        sol = odeint(growth, sol[-1].reshape(-1), time, args = (t, env, Y_copy))

        sol = sol.reshape((len(time), -1, z_steps))

    r_specs[i] = np.sum(
        np.sum(sol[-1,:-2], axis = -1) > 1e-3*np.sum(sol[-1,:-2]), axis = 0)
    
    print(i, r_specs[i], timer()-start, timer()-start_all, "\n\n")
    if r_specs[i]>1:
        fun(t_precision-1, sol, env, t)