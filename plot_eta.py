import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq

from timeit import default_timer as timer

import generate_plankton as gp
import plankton_growth as pg

n_coms = 100
n_spec = 2
traits = gp.generate_plankton(n_spec, n_coms)
env = gp.generate_env(n_coms)
traits = gp.phytoplankton_equilibrium(traits, env)

def NO_fun(c, equi_dist, ti, envi, r_i, mu, exp = 1,
           ret = "diff"):
    print("NO_fun", c, exp)
    eta = [eta_growth(c**exp, equi_dist[0], ti, envi, 5),
           eta_growth(c**(-exp), equi_dist[1], ti, envi, 4)]
    NO = (mu - r_i)/(mu - eta)
    print(NO)
    if ret == "diff":
        return NO[0] - NO[1]
    else:
        return eta
    
def eta_growth(c, sol, ti, envi, id_inv, ret_all = False):
    
    time_new = sol.t[sol.t>=sol.t[-1]/2]
    id_other = 4 if id_inv == 5 else 5
    
    N_all = np.full(6, 1e-10)
    def help_fun(time, N_other):
        N_all[id_inv] = sol.sol(time)[id_other] + np.log(c)
        N_all[:4] = N_other
        return pg.convert_ode_to_log(N_all, ti, envi)[:4]
    
    N_start = np.log(sol.y[:4, len(sol.t)//2]).flatten()
    for i in range(2):
        sol_eta = solve_ivp(help_fun, time_new[[0,-1]], N_start,
                        t_eval = time_new, rtol = 1e-3)
        N_start = sol_eta.y[:,-1]
    
    sol_eta.y = np.exp(sol_eta.y)
    sol_eta.y[sol_eta.y < 1e-20] = 1e-20  
    
    sol_eta.y = np.append(sol_eta.y, np.zeros((2, len(sol_eta.t))), axis = 0)
    sol_eta.y[id_inv] = c*sol.y[id_other, sol.t>=sol.t[-1]/2]
    eta_i_all = np.empty(len(time_new))
    for k in range(len(time_new)):
        eta_i_all[k] = pg.per_cap_plankton_growth(sol_eta.y[:,k], ti, envi)[id_inv]
     
    
    if ret_all:
        return np.mean(eta_i_all), sol_eta.t, sol_eta.y
    return np.mean(eta_i_all)

save = np.random.randint(10000)

##############################################################################
# start computation

start  = timer()
issue = np.empty(n_coms, dtype = "object")
mu_all, r_all, eta_all = np.full((3,n_coms, 2), np.nan)
for i in range(n_coms):
    print(i, timer()-start)
    ti, envi, i = gp.select_i(traits, env, i)
    # one phyto species can't survive
    # do not consider as it's competition for one resource
    if np.any(np.isnan(ti["N_star_P_res"])):
        issue[i] = "phyto_cant_exist"
        continue
    
    # growth rates of species at phyto monoculture equilibrium
    r_phyto = np.empty((2,6))
    
    # compute intrinsic growth rate of zooplankton species
    for j in range(2):
        N = np.array([ti["R_star_n_res"][j], ti["R_star_p_res"][j]] + 4*[0])
        N[2+j] = ti["N_star_P_res"][j]
        r_phyto[j] = pg.per_cap_plankton_growth(N, ti, envi)
    
    coex_state_phyto = np.sum(r_phyto[[0,1],[3,2]]>0)
    
    # ignore priority effects of phytoplankton species as
    # initial state of zooplankton is not clearly defined
    if coex_state_phyto != 1:
        issue[i] = "coex_phyto" if coex_state_phyto == 2 else "priority_phyto"
        continue
    
    j_phyto = np.where(r_phyto[[0,1],[3,2]]>0)[0][0]
    mu_zoo = r_phyto[j_phyto, -2:]
    # one of the zooplankton can't survive in monoculture, do not consider
    if np.any(mu_zoo<0):
        issue[i] = "neg_mu_zoo"
        continue
    
    # compute monoculture equilibrium density of both zooplankton species
    mu_Z_org = ti["mu_Z"].copy()
    t_eval = np.linspace(-100,1000,1101)
    N_start = np.array([ti["R_star_n_res"][j_phyto], ti["R_star_p_res"][j_phyto]] + 4*[10])
    N_start[2+j_phyto] = ti["N_star_P_res"][j_phyto]
    stable = [False, False]
    equi_dist = [None, None]
    for j in range(2):
        # to ensure other species does not interfere
        ti["mu_Z"][1-j] = 1e-10
        for k in range(5):
            print(j, k)
            sol = solve_ivp(lambda t, logN: pg.convert_ode_to_log(logN, ti, envi),
                            t_eval[[0,-1]], np.log(N_start), t_eval = t_eval,
                            dense_output=True)
            sol.y = np.exp(sol.y)
            sol.y[sol.y <1e-20] = 1e-20
            N_start = sol.y[:,-1]
            
            # check whether equilibrium has been reached
            fluct = np.nanstd(sol.y[:,-100:], axis = 1)/np.nanmean(sol.y[:,-100:], axis = 1)
            absent = np.amax(sol.y[:,-100:], axis = 1)<1e-3
            
            if np.all(absent | (fluct<0.05)):
                stable[j] = True
                break

            if k == 4:
                plt.figure()
                plt.plot(sol.t, sol.y.T, label = "lab")
                plt.legend()
                plt.semilogy()
                plt.title("{} present".format(j))
                plt.show()
        # save equilibrium distribution
        equi_dist[j] = sol
        ti["mu_Z"] = mu_Z_org.copy()
        
    if all(stable):
        issue[i] = "stable equilibria"
        continue
    
    if np.nanmedian(equi_dist[0].y[-2]) <1e-19:
        issue[i] = "no zoop"
        continue
    if np.nanmedian(equi_dist[1].y[-1]) <1e-19:
        issue[i] = "no zoop"
        continue
    
    # given equilibrium distribution compute invasion growth rates
    r_i_all = np.empty((2, len(t_eval)//2))
    for j in range(2):
        for k in range(len(t_eval)//2):
            r_i_all[j,k] = pg.per_cap_plankton_growth(equi_dist[1-j].y[:,-k], ti, envi)[4+j]
    
    r_i = np.mean(r_i_all, axis = 1)
    
    # compute conversion factor
    diff_NO = np.sign(NO_fun(1, equi_dist, ti, envi, r_i, mu_zoo))
    
    xtol = 1e-3
    rtol = 1e-3
    cs = np.linspace(1e-3, 1, 20)
    etas = np.empty((len(cs), 2))
    sol_y, sol_t = np.empty((2,2,len(cs)), dtype = "object")
    start_plot = timer()
    for j in range(2):
        ti["mu_Z"][j] = 1e-10
        for k, c in enumerate(cs):
            print(j,k, timer()-start_plot)
            c_eff = 1/c if j else c
            c_eff = c_eff**diff_NO
            etas[k,j], sol_t[j,k], sol_y[j,k] = eta_growth(
                c_eff, equi_dist[j], ti, envi, 5-j, ret_all = True)
        ti["mu_Z"] = mu_Z_org.copy()
    path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"
    NO = (mu_zoo - r_i)/(mu_zoo-etas)
    
    np.savez(path + "eta_function_{}_{}".format(i, save),
             **ti, **envi, sol_t = sol_t, sol_y = sol_y, etas = etas, cs = cs,
             NO = NO, equi_dist = equi_dist)
    
    fig = plt.figure()
    plt.plot(cs, NO)
    fig.savefig("Figure_test_eta_function_{}_{}.png".format(i, save))
    plt.show()
    

print({key: sum(issue == key) for key in set(issue)})