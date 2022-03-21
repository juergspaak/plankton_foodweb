import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer

import generate_plankton as gp
import plankton_growth as pg
from scipy.integrate import solve_ivp, simps
from scipy.optimize import brentq
from NFD_definitions.no_fixpoint_NFD import no_fixpoint_NFD, InputError

from assembly_time_fun import assembly_richness

def focal_fun_gen(pars, tech, sol, traits,
                  ind_res = -1):

    N_all = np.empty(len(tech["ind_non_foc"])
                     + len(tech["ind_foc"]))
    
    def focal_fun(time,N,c):
        N_all[tech["ind_non_foc"]] = sol.sol(time)[ind_res]*c
        N_all[tech["ind_foc"]] = N
        return pars["dN_dt"](N_all, traits)[tech["ind_foc"]]
    return focal_fun 

def growth(dens, time, tech):
    growth_time = pg.per_cap_plankton_growth(dens, tech["ti"],
                                             tech["envi"])
    return simps(growth_time, x = time, axis = 0)/(time[-1] - time[0])

def compute_equilibrium(logN_start, ti, envi, t_end = 300, dt = 1,
                        tol = 1e-2, t_max = 5000):
    t_eval = np.append(np.arange(-t_end, 0, dt),
                       np.arange(0, t_end, dt))
    time = t_eval[[0,-1]]
    sol = solve_ivp(lambda t,logN: pg.convert_ode_to_log(logN, ti, envi),
                    time, logN_start, t_eval = t_eval, dense_output=True)
    
    # cut off burn in time
    sol.y = sol.y[:,sol.t>=0]
    sol.t = sol.t[sol.t>=0]
    
    # set minimal value for densities
    y_compare = sol.y.copy()
    y_compare[y_compare<np.log(1e-8)] = np.log(1e-8)
    
    plt.semilogy(sol.t, np.exp(y_compare.T))
    plt.show()
    
    # check whether distribution is same in first and second half
    percent = [5,25,50,75,95]
    # distribution considering only first half
    percent_first = np.nanpercentile(y_compare[:, :len(sol.t)//2],
                                     percent, axis = 1)
    # distribution cosidering half the time steps
    percent_uneven = np.nanpercentile(y_compare[:,::2], percent, axis = 1)
    # most precise distribution
    percent_precise = np.nanpercentile(y_compare, percent, axis = 1)
    
    
    # distribution in first half does not differ from entire distribution
    if np.allclose(percent_first, percent_precise, atol = tol,
                   rtol = tol):
        increase_time_length = False
    else:
        increase_time_length = True
        t_end *= 2
    if t_end >= t_max:
        t_end = np.copy(t_max)
        increase_time_length = False
        
    # distribution rate with double time step equal to precise growth rate
    if np.allclose(percent_uneven, percent_precise, atol = tol
                   , rtol = tol):
        decrease_time_step = False
    else:
        decrease_time_step = True
        dt /= 2
    
    if increase_time_length or decrease_time_step:
        return compute_equilibrium(sol.y[:,-1], ti, envi, t_end, dt, tol,
                                   t_max)
    else:
        # change back to non-log space
        sol.y = sol.y
        #sol.y[sol.y<1e-8] = 1e-8
        return sol


def NO_fun(c, pars, tech):
    c = [c,1/c]
    for j in range(tech["n_spec"]):
        # remove invader j from commnity
        t_temp = gp.select_present(tech["ti"], ind_zoo = j)
        focal_fun = focal_fun_gen(pars, tech, pars["sol"][j],
                                  t_temp)
        time = pars["sol"][j].t
        start = timer()
        sol = solve_ivp(focal_fun, time[[0,len(time)//2-1]],
                        pars["sol"][j].y[:tech["n_env"],0],
                        t_eval = time[:len(time)//2], args = (c[j],))
        if timer()-start > 120:
            pars["no_compute"] = "time"
            raise InputError
        sol = solve_ivp(focal_fun, time[[0,-1]],
                        sol.y[:,-1],
                        t_eval = time, args = (c[j],))
        if timer()-start > 120:
            pars["no_compute"] = "time"
            raise InputError
        dens_insert = np.zeros((ti["r_zoo"], len(sol.t)))
        dens_insert[j] = c[j]*pars["N_star"][j][tech["n_env"] + id_other[j]]
        pars["N_eta"][j] = np.insert(np.exp(sol.y), tech["n_env"],
                          dens_insert, axis = 0)
        pars["eta"][j] = growth(pars["N_eta"][j].T, sol.t,
                        tech)[tech["n_env"] + j]
    
    if (pars["eta"]>pars["mu"]-1e-2).any():
        pars["no_compute"] = "allee"
        raise InputError
    
    pars["NO"] = (pars["mu"]-pars["r_i"])/(pars["mu"]-pars["eta"])
    pars["ND"] = 1-pars["NO"]
    pars["F"] = -pars["eta"]/(pars["mu"] -pars["eta"])
    return np.abs(pars["NO"])

def solve_c(pars, tech, sp = [0,1]):
    
    # test niche overlap with initial guess of c
    NO_start = NO_fun(pars["c"][sp[0],sp[1]], pars, tech)
    print("NO_start")
    sp = np.asarray(sp)
    if np.any(pars["eta"] >pars["mu"]-1e-2):
        raise InputError
    def inter_fun(c):
        NO = NO_fun(c, pars, tech)
        print("intermediate", NO, c)
        if np.isclose(NO[0], NO[1], rtol = tech["ND_tol"]):
            print("shortcut")
            return 0
        else:
            return NO[0]-NO[1]
        
    # find interval for brentq method
    a = pars["c"][sp[0],sp[1]]
    # find which species has higher NO for c0
    direction = np.sign(NO_start[0]-NO_start[1])
    if direction == 0: # starting guess for c is correct
        return a, 1/a
    fac = 2**direction
    if not np.isfinite(direction):
        raise InputError("function `f` seems to be returning nonfinite values")

    b = float(a*fac)
    # change searching range to find c with changed size of NO
    while np.sign(inter_fun(b)) == direction:
        a = b
        b *= fac
        print("ab",a,b)
        # test whether a and be behave as they should (e.g. nonfinite)
        if not((2*a == b) or (2*b == a)) or np.sign(b-a) != direction:
            raise InputError("Not able to find c_{}^{}.".format(*sp) +
                "Please pass a better guess for c_i^j via the `pars` argument"+
                ". Please also check for non-positive entries in pars[``c``]")
    # solve equation
    try:
        c = brentq(inter_fun,a,b, xtol = 1e-4, rtol = 1e-4)
    except ValueError:
        raise ValueError("f does not seem to be continuous."
                         + "Niche and fitness differences cannot be computed"
                         + "automatically")
    # test whether c actually is correct
    # c = 0 implies issue with brentq
    if ((c==0)  or inter_fun(c)>tech["gtol"]):
        pars["c"] = [c, 1/c]
        pars["ND"][:] = np.nan
        pars["F"][:] = np.nan
        pars["no_compute"] = "discontinuous"
        raise InputError("Not able to find c_{}^{}.".format(*sp) +
                "Please pass a better guess for c_i^j via the `pars` argument"+
                ". Please also check for non-positive entries in pars[``c``]")
    pars["c"] = [c, 1/c]
    return c, 1/c # return c_i and c_j = 1/c_i

# generate communities
n_com = 100
r_phyto = 2
r_res = 2
try:
    traits_coex
except NameError:
    n_coms = 100
    traits = gp.generate_plankton(5, n_coms)
    env_org = gp.generate_env(n_coms)
    traits_org = gp.phytoplankton_equilibrium(traits, env_org)
            
    rich_all, id_survive, res_equi, dens_equi = assembly_richness(traits_org, env_org, plot_until = 0)
    
    ind = np.all(rich_all == 2, axis = 1)
    traits_coex, env, ind = gp.select_i(traits_org, env_org, ind)
    
    ind_phyto = np.where(id_survive[ind, 0])[1].reshape(-1,2)
    ind_zoo = np.where(id_survive[ind, 1])[1].reshape(-1,2)
    
    traits_select = {}
    for key in gp.pt.phyto_traits:
        traits_select[key] = np.take_along_axis(traits_coex[key],ind_phyto,
                                                axis = -1)
    for key in gp.zt.zoop_traits:
        traits_select[key] = np.take_along_axis(traits_coex[key],ind_zoo,
                                                axis = -1)
    
    # combined traits
    for key in ["h_zp", "s_zp"]:
        traits_select[key] = np.take_along_axis(traits_coex[key],
                                        ind_phyto[:,np.newaxis], axis = -1)
        traits_select[key] =  np.take_along_axis(traits_select[key],
                                        ind_zoo[...,np.newaxis], axis = -2)
    traits_select["r_phyto"] = 2
    traits_select["r_zoo"] = 2
    
    traits = traits_select
    
#traits = gp.generate_plankton(r_phyto, n_com)
n_spec = 2 + traits["r_phyto"] + traits["r_zoo"]
n_com = len(traits_coex["mu_P"])

# compute equilibrium density of phytoplankton in monoculture
traits = gp.phytoplankton_equilibrium(traits, env)

# focus on communities where both phytoplankton can survive in monoculture
ind = np.all(np.isfinite(traits["N_star_P_res"]), axis = 1)
env = {key: env[key][ind] for key in env.keys()}
traits = {key: traits[key][ind] for key in gp.select_keys(traits)}
traits["r_phyto"] = r_phyto
traits["r_zoo"] = r_phyto
n_com = traits["n_coms"] = len(traits["mu_P"])

# compute invasion growth rates of phytoplankton (and zoo plankton)
inv_phyto = np.empty((traits["r_phyto"], n_com, n_spec))
inv_phyto2 = np.empty((traits["r_phyto"], n_com, n_spec))
invader = [1,0]
for i in range(traits["r_phyto"]):
    N_resident = np.zeros((traits["n_coms"],
                               2 + traits["r_phyto"] + traits["r_zoo"]))
    # monoculture equilibrium conditions
    N_resident[:,0] = traits["R_star_p_res"][:,i]
    N_resident[:,1] = traits["R_star_n_res"][:,i]
    N_resident[:,2+i] = traits["N_star_P_res"][:,i]
    for j in range(traits["n_coms"]):
        ti, envi, j = gp.select_i(traits, env, j)
        inv_phyto[i,j] = pg.per_cap_plankton_growth(N_resident[j], ti, envi)
        
        inv_phyto2[i] = inv_phyto[i].copy()
        # if other phytoplankton species could invade, remove equilibrium
    inv_phyto[i, inv_phyto[i,:,invader[i]]>0] = np.nan
 
# find the competitive winner
id_phyto = np.full(traits["n_coms"], np.nan)
# select cases where one phytoplankton won, cases with priority see below
id_phyto = np.argmin(np.isfinite(inv_phyto[...,-1]), axis = 0)
    
# cases of priority effects, randomly select one phytoplankton as winner
ind_prio = np.all(np.isfinite((inv_phyto[...,-1])), axis = 0)
id_phyto[ind_prio] = np.random.binomial(1, 0.5, np.sum(ind_prio))

# how often can we compute niche and fitness differences
NFD_comp = np.all(inv_phyto[id_phyto, np.arange(traits["n_coms"]), -2:]>0,
                  axis = -1)

# cases where both phytoplankton coexist
id_coex = -100
id_phyto[np.all(np.isnan(inv_phyto[...,-1]), axis = 0)] = id_coex
 
result = {key: np.full((traits["n_coms"], 2), np.nan)
          for key in ["ND", "F", "eta", "r_i", "mu", "c"]}
result["N_star"] = np.empty((traits["n_coms"], 2), dtype = object)
non_continuous = []
computed = []
id_other = [1,0]
all_start = timer()
time = np.full(traits["n_coms"], np.nan)
for i in range(traits["n_coms"]):
    start = timer()
    # can NFD be computed?
    if not NFD_comp[i]:
        result["ND"][i] = 1
        result["F"][i] = 1
        computed.append("no_monoculture")
        continue
    print(i, id_phyto[i])
    if id_phyto[i] == id_coex:
        computed.append("coexistence")
        continue # xxx to be added
    ti, envi, i = gp.select_i(traits, env, i)
    tech = {"ti": ti, "envi": envi, "gtol": 1e-2, "ind_foc": np.arange(4),
        "ind_non_foc": 4 + np.arange(1),
        "n_env": 4, "n_spec": 2, "ND_tol": 1e-3}
    pars = {"N_star": np.empty(ti["r_zoo"], dtype = object),
            "sol": np.empty(ti["r_zoo"], dtype = object),
            "mu": inv_phyto[id_phyto[i], i, -ti["r_zoo"]:],
            "r_i": np.empty(ti["r_zoo"])}
    
    # compute the resident equilibrium densities
    for j in range(ti["r_zoo"]):
        
        # remove invader j from commnity
        t_temp = gp.select_present(ti, ind_zoo = id_other[j])
        
        N_start = np.empty(tech["n_env"] + t_temp["r_zoo"])
        # resource concentrations
        N_start[:r_res] = [ti["R_star_n_res"][id_phyto[i]],
                            ti["R_star_p_res"][id_phyto[i]]]
        # competitive dominant at equilibrium density
        N_start[id_phyto[i] + r_res] = ti["N_star_P_res"][id_phyto[i]]
        # other phytoplankton at low density
        N_start[r_res + id_other[id_phyto[i]]] = 10 
        
        # introduce resident zooplankton
        N_start[tech["n_env"]:] = 10

        pars["sol"][j] = compute_equilibrium(np.log(N_start), t_temp, envi)
        pars["N_star"][j] = np.insert(np.exp(pars["sol"][j].y),
                                      tech["n_env"] + j,
                                      0, axis = 0)
        
        pars["r_i"][j] = growth(pars["N_star"][j].T, pars["sol"][j].t,
                                 tech)[tech["n_env"] + j]
        print(i,j, "equi computed", len(pars["sol"][j].t), pars["sol"][j].t[-1])
    pars["N_eta"] = np.empty(ti["r_zoo"], dtype = "object")
    pars["eta"] = np.empty(ti["r_zoo"])
    pars["dN_dt"] = lambda logN, traits = ti: pg.convert_ode_to_log(logN,
                                                                    traits, envi)
    pars["c"] = np.ones((2,2))
    
    tech = {"ti": ti, "envi": envi, "gtol": 1e-2, "ind_foc": np.arange(4),
            "ind_non_foc": 4 + np.arange(1),
            "n_env": 4, "n_spec": 2, "ND_tol": 1e-3}
    try:
        c = solve_c(pars, tech)
        
        for key in result.keys():
            result[key][i] = pars[key]
        print("c", c)
        print("NO", (pars["mu"]-pars["r_i"])/(pars["mu"]-pars["eta"]))
        print("time", timer()-start,timer()-all_start)
        time[i] = timer()-start
        computed.append("done")
    except InputError:
        non_continuous.append(i)
        computed.append(pars["no_compute"])
        print("no-compute", pars["no_compute"])
        
    
#"""
  
   
        
    
