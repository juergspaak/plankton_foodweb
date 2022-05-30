import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from scipy.optimize import minimize, curve_fit, fsolve
"""

# parameters of the eppeley curve, cite from Norberg 2004
a_epp = 0.59
b_epp = 0.0633
temp = np.linspace(0,40,1001)

# load empirical data
temp_traits = pd.read_csv("empirical_data/temperature_traits.csv",
                          encoding = 'ISO-8859-1')
# rename columns
temp_traits = temp_traits[["Optimum", "Niche.width"]]
temp_traits.columns = ["T_opt", "T_sig"]
fit_cols = ["a_{}".format(i) for i in range(4)]
for key in fit_cols:
    temp_traits[key] = np.nan

# exclude very wide temporal niches
temp_traits = temp_traits[temp_traits["T_sig"]<=40]


x0 = [a_epp,b_epp,a_epp/2,b_epp/2]

def fun_to_minimize(a, T):
    return a[0]*np.exp(a[1]*T) - a[2]*np.exp(a[3]*T)

for i, row in temp_traits.iterrows():
    

    mu = a_epp*np.exp(b_epp*temp)*(1-4*(temp - row["T_opt"])**2/row["T_sig"]**2)


    T_max = temp[np.argmax(mu)]
    # zero growth should be at same temperature
    T_zero = row["T_opt"] + 0.5*row["T_sig"]
    # other zero location
    T_min = row["T_opt"] - 0.5*row["T_sig"]


    constraints = [
        # maximum growth rate at correct location
        dict(type = "eq", fun = lambda a:
             T_max - (np.log(a[0]/a[2]) + np.log(a[1]/a[3]))/(a[3]-a[1])),
        # maximum growth rate equal
        dict(type = "eq",  fun = lambda a:
             np.amax(mu) - (a[0]*np.exp(a[1]*T_max) - a[2]*np.exp(a[3]*T_max))),
        # mortality rate at equal temperature
        dict(type = "eq", fun = lambda a:
             T_zero - np.log(a[0]/a[2])/(a[3] - a[1])),
        dict(type = "ineq", fun = lambda a: a[3]-a[1])]
    with warnings.catch_warnings(record = True):
        a = minimize(fun_to_minimize, x0, args = (T_min, ),
                     constraints=constraints,
                 bounds = 4*[[0,None]])

    temp_traits.loc[i, fit_cols] = a["x"]
    temp_traits.loc[i, "success"] = a["success"]"""
    
fig, ax = plt.subplots(4,4, sharex = "col", sharey = "row", figsize = (9,9))

for i in range(4):
    for j in range(4):
        if i == j:
            continue
            ax[i,i].hist(temp_traits[fit_cols[i]], bins = 100)
            ax[i,i].hist(temp_traits[fit_cols[i]][temp_traits["success"]], bins = 100)
        else:
            ax[i,j].scatter(temp_traits[fit_cols[j]], temp_traits[fit_cols[i]],
                            s = 2, c = temp_traits["success"])
        