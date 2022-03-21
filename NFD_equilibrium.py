import numpy as np
from timeit import default_timer as timer
from NFD_definitions.numerical_NFD import NFD_model, InputError

import generate_plankton as gp
import plankton_growth as pg

from NFD_zooplankton2 import zoop_growth


def fast_NFD(traits, env, save):
    itera = len(env["N"])
    results = {key: np.full((itera, 2), np.nan)
               for key in ["ND", "F", "r_i", "mu", "eta", "N_star"]}
    issue = np.empty(itera, dtype = object)
    start = timer()
    for i in range(itera):
        #print(i)
        #print("itera", i)
        ti, envi, i = gp.select_i(traits, env, i)
        t_sub = [gp.select_present(ti, ind_phyto = 0),
                 gp.select_present(ti, ind_phyto = 1)]
        # check whether both species have positive intrinsic growth rate
        try:
            mu = zoop_growth([0,0], ti, envi, t_sub)
        except (RuntimeError, OverflowError):
            issue[i] = "coex_int"
            continue
        if min(mu)<0:
            issue[i] = "mu<0"
            continue
        
        # find equilibrium densities
        pars = {"N_star": np.ones((2,2))}
        
        try:
            pars = NFD_model(zoop_growth, args = (ti, envi, t_sub), pars = pars,
                         estimate_N_star_mono=True)
            issue[i] = "done" if np.any(pars["F"] < 1) else "eta"
            for key in results.keys():
                if key == "N_star":
                    continue
                results[key][i] = pars[key]
            results["N_star"][i] = pars["N_star"][[1,0],[0,1]]
        except InputError:
            for j in range(2):
                if np.abs(zoop_growth(pars["N_star"][j], ti, envi, t_sub)[j]) > 1e-4:
                    issue[i] = "N_star"
            if issue[i] == None:
                issue[i] = "new issue"
        except ValueError:
            issue[i] = "brentq"
        except RuntimeError:
            issue[i] = "runtime"
        except OverflowError:
            issue[i] = "coex"
        
        if i%100==99:
            print(save, i, timer()-start)
            np.savez(save, i = i, **results, issue = issue, **traits, **env)
    """
    print("\n\n")
    for case in set(issue):
        print(case, sum(issue == case))"""
        
    return results, issue

if __name__ == "__main__":
    itera = 200
    traits = gp.generate_plankton(2, itera)
    env = gp.generate_env(itera)
    
    results, issue = fast_NFD(traits, env)