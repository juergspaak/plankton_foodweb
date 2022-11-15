import numpy as np
from timeit import default_timer as timer
from NFD_definitions.numerical_NFD import NFD_model, InputError
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

import generate_plankton as gp
import plankton_growth as pg


def fast_NFD(traits, env, save = None):
    itera = len(env["N"])
    results = {key: np.full((itera, 2), np.nan)
               for key in ["ND", "F", "r_i", "mu", "eta", "N_star"]}
    issue = np.empty(itera, dtype = object)
    start = timer()
    for i in range(itera):
        #print(i)
        #print("itera", i)
        ti, envi, i = gp.select_i(traits, env, i)
        # check whether both species have positive intrinsic growth rate
        try:
            mu = zoop_growth([0,0], ti, envi)
        except (RuntimeError, OverflowError):
            issue[i] = "coex_int"
            continue
        if min(mu)<0:
            issue[i] = "mu<0"
            continue
        
        # find equilibrium densities
        pars = {"N_star": np.ones((2,2))}
        
        try:
            pars = NFD_model(zoop_growth, args = (ti, envi), pars = pars,
                         estimate_N_star_mono=True)
            issue[i] = "done" if np.any(pars["F"] < 1) else "eta"
            for key in results.keys():
                if key == "N_star":
                    continue
                results[key][i] = pars[key]
            results["N_star"][i] = pars["N_star"][[1,0],[0,1]]
        except InputError:
            for j in range(2):
                if np.abs(zoop_growth(pars["N_star"][j], ti, envi)[j]) > 1e-4:
                    issue[i] = "N_star"
            if issue[i] == None:
                issue[i] = "new issue"
        except ValueError:
            issue[i] = "brentq"
        except RuntimeError:
            issue[i] = "runtime"
        except OverflowError:
            issue[i] = "coex"
        
        if (i%100==99) and not (save is None):
            print(save, i, timer()-start)
            np.savez(save, i = i, **results, issue = issue, **traits, **env)
    """
    print("\n\n")
    for case in set(issue):
        print(case, sum(issue == case))"""
        
    return results, issue

def zoop_growth(N_zoo, ti, envi):
    N_start = np.append([envi["N"], envi["P"]], ti["N_star_P_res"]/2)
    N_temp = np.empty(6)
    N_temp[-2:] = N_zoo
    def help_fun(N):
        N_temp[:4] = N
        return pg.per_cap_plankton_growth(N_temp, ti, envi)[:4]
    t_eval = np.linspace(0, 100, 20)
    sol = solve_ivp(lambda t,N: N*help_fun(N), [0,100], N_start, t_eval = t_eval)
    stable = np.isclose(sol.y[:,-1], sol.y[:,-2], rtol = 1e-4)
    present = sol.y[:,-1]>1e-3
    if not np.all(stable & present):
        t_eval = np.linspace(0, 1000, 20)
        sol = solve_ivp(lambda t,N: N*help_fun(N), [0,1000], N_start,
                        t_eval = t_eval)
    stable = np.isclose(sol.y[:,-1], sol.y[:,-2], rtol = 1e-4)
    present = sol.y[:,-1]>1e-3
    if not np.all(stable & present):
        raise RuntimeError("No stable equlibrium point found")
    N_temp[:4] = sol.y[:,-1]
    return pg.per_cap_plankton_growth(N_temp, ti, envi)[-2:]
    
    
    

if __name__ == "__main__":
    itera = 200
    traits = gp.generate_plankton(2, itera)
    env = gp.generate_env(itera)
    traits = gp.phytoplankton_equilibrium(traits, env)
    
    results, issue = fast_NFD(traits, env)