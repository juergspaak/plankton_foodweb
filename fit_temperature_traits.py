import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.stats import linregress

from scipy.optimize import minimize


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
fit_cols = ["a_1", "a_2", "b_1", "b_2"]
for key in fit_cols:
    temp_traits[key] = np.nan

# exclude very wide temporal niches
#temp_traits = temp_traits[temp_traits["T_sig"]<=40]
temp_traits.index = np.arange(len(temp_traits))


x0 = [a_epp,b_epp,a_epp/2,b_epp/2]

def fun_to_minimize(a, mu, temp):
    mu_fitted = a[0]*np.exp(a[1]*temp) - a[2]*np.exp(a[3]*temp)
    return np.nansum((mu_fitted - mu)**2)
    

for i, row in temp_traits.iterrows():
    

    mu = a_epp*np.exp(b_epp*temp)*(1-4*(temp - row["T_opt"])**2/row["T_sig"]**2)
    #mu[mu<0] = np.nan

    T_max = temp[np.nanargmax(mu)]
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
             np.nanmax(mu) - (a[0]*np.exp(a[1]*T_max) - a[2]*np.exp(a[3]*T_max))),
        # mortality rate at equal temperature
        dict(type = "eq", fun = lambda a:
             T_zero - np.log(a[0]/a[2])/(a[3] - a[1])),
        dict(type = "ineq", fun = lambda a: a[3]-a[1])]
    #constraints = None
    #constraints = []
    with warnings.catch_warnings(record = True):
        a = minimize(fun_to_minimize, x0, args = (mu, temp ),
                     constraints=constraints,
                 bounds = 4*[[0,None]])

    temp_traits.loc[i, fit_cols] = a["x"]
    temp_traits.loc[i, "success"] = a["success"]

#"""
# remove outliers from the data
temp_traits["good"] = (temp_traits["T_sig"]<40) & temp_traits["success"]

# remove outliers for a_2
iqr = np.nanpercentile(temp_traits.loc[temp_traits["good"], "a_2"], [25,75])
iqr = iqr + np.array([-1.5, 1.5])*(iqr[1]-iqr[0])
temp_traits["good"] = ((temp_traits["a_2"] > iqr[0]) & 
                           (temp_traits["a_2"] < iqr[1]) &
                           temp_traits["good"])


# perform linear regression to obtain estimates of b_1 and b_2
s_1, i_1, R_1, p_1, std_1 = linregress(temp_traits.loc[temp_traits.good, "a_1"],
                                       temp_traits.loc[temp_traits.good, "b_1"])
s_2, i_2, R_2, p_2, std_2 = linregress(temp_traits.loc[temp_traits.good, "a_2"],
                                       temp_traits.loc[temp_traits.good, "b_2"])

# store relevant parameters for future runs
mu_a2 = np.mean(temp_traits.loc[temp_traits.good, "a_2"])
sig_a2 = np.std(temp_traits.loc[temp_traits.good, "a_2"])

def generate_temp_traits(mu_P):
    # given mu_P generates  a_1, a_2, b_1 and b_2
    
    a_2 = np.random.normal(mu_a2, sig_a2, mu_P.shape)
    
    b_2 = s_2*a_2 + i_2
    T_ref = 20 # reference temperature is 15°
    
    # solve the equation mu = a_1*exp(a_2*T_ref) - b_1*exp(b_2*T_ref)
    # using the information that b_1 = s_1*a_ + i_1
    a_1 = (mu_P + i_1*np.exp(b_2*T_ref))/(np.exp(a_2*T_ref) - s_1*np.exp(b_2*T_ref))
    b_1 = s_1*a_1 + i_1
    
    return a_1, a_2, b_1, b_2

if __name__ == "__main__":
    
    # plot examples of fitted curves
    fig, ax = plt.subplots(3,3, sharex = True, sharey = True, figsize = (9,9))
    
    for i in range(len(ax)):
        ax[i,0].set_ylabel("Growth rate")
        ax[-1,i].set_xlabel("Temperature")
    
    ax = ax.flatten()
    
    counter = 0
    for i, row in temp_traits.iterrows():
        
        if counter>=len(ax)-1:
            continue
        if not row["good"]:
            continue
        T_zero = row["T_opt"] + 0.5*row["T_sig"]
        #temp = np.linspace(0,T_zero+2, 1000)
        mu = a_epp*np.exp(b_epp*temp)*(1-4*(temp - row["T_opt"])**2/row["T_sig"]**2)
        T_max = temp[np.argmax(mu)]
        
        ax[counter].plot(temp, mu)
        ax[counter].plot(T_max, np.amax(mu), 'bo')
        ax[counter].plot(T_zero, 0, 'bo')
        ax[counter].plot(temp, a_epp*np.exp(b_epp*temp), "k", label = "Eppeley curve")
        
        mu_fit = (row["a_1"]*np.exp(row["a_2"]*temp)
                   - row["b_1"]*np.exp(row["b_2"]*temp))
        ax[counter].plot(temp, mu_fit,
                   color = "red", label = "Fitted")
        counter += 1
        ax[-1].plot(temp, mu_fit)
        #ax[i].set_ylim([0,np.amax(mu) +0.5])
    ax[-1].plot(temp, a_epp*np.exp(b_epp*temp), "k")  
    ax[0].set_ylim([0,4.2])
    fig.savefig("Figure_int_phytoplankton_temperature_niche.pdf")
        
    
    ##########################################################################
    # plot raw values
    from generate_plankton import generate_plankton
    traits = generate_plankton(1, np.sum(temp_traits["good"]))
    a_fitted = np.array(generate_temp_traits(traits["mu_P"].flatten()))
    
    fig, ax = plt.subplots(4,4,  figsize = (11,11))
    
    for i in range(4):
        ax[i,0].set_ylabel(fit_cols[i])
        ax[-1,i].set_xlabel(fit_cols[i])
        
        ax[i,i].set_xticklabels([])
        ax[i,i].set_yticks([])
        a, bins, p = ax[i,i].hist(temp_traits[fit_cols[i]], bins = 30,
                                  color = "green", label = "used")
        ax[i,i].hist(temp_traits.loc[~temp_traits["good"], fit_cols[i]],
                     bins = bins, color = "red", label = "outliers")
        ax[i,i].hist(temp_traits.loc[temp_traits["T_sig"]>40, fit_cols[i]],
                     bins = bins, color = "purple", label = "T_sig>40")
        ax[i,i].hist(temp_traits.loc[~temp_traits["success"], fit_cols[i]],
                     bins = bins, color = "blue", label = "Bad fit")
        ax[i,i].hist(a_fitted[i], bins = bins, color = "orange", alpha = 0.5,
                     label = "Fitted")
        
        ax[i,i].text(0.5,0.5, fit_cols[i], ha = "center", va = "center",
                     transform=ax[i,i].axes.transAxes, fontsize = 20)
        for j in range(4):
            if i == j:
                continue
    
            ax[i,j].scatter(temp_traits[fit_cols[j]], temp_traits[fit_cols[i]],
                                s = 2, color = "green")
            ax[i,j].scatter(temp_traits.loc[~temp_traits["good"], fit_cols[j]],
                            temp_traits.loc[~temp_traits["good"], fit_cols[i]],
                            s = 2, color = "red")
            ax[i,j].scatter(temp_traits.loc[temp_traits["T_sig"]>40, fit_cols[j]],
                            temp_traits.loc[temp_traits["T_sig"]>40, fit_cols[i]],
                            s = 2, color = "purple")
            ax[i,j].scatter(temp_traits.loc[~temp_traits["success"], fit_cols[j]],
                            temp_traits.loc[~temp_traits["success"], fit_cols[i]],
                            s = 2, color = "blue")
            ax[i,j].scatter(a_fitted[j], a_fitted[i], color = "orange",
                            s = 2, alpha = 0.5, label = "fitted")
            if j != 0:
                ax[i,j].set_yticklabels([])
            if i != 3:
                ax[i,j].set_xticklabels([])
                
    for a in [ax[0,2], ax[2,0], ax[1,-1], ax[-1,1]]:
        a.set_xlim(a.get_xlim())
        a.set_ylim(a.get_ylim())
        a.plot(a.get_xlim(), a.get_xlim(), zorder = 0, linestyle = "--")
        
    ax[0,0].legend()
                
    fig.suptitle(r"$\mu(T) = a_1\exp(a_2 T) - b_1\exp(b_2T)$", fontsize = 20)
        
    fig.savefig("Figure_int_fit_temperature_traits.pdf")
    
        