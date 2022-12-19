import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# parameters of the eppeley curve, cite from Norberg 2004
a_epp = 0.59
b_epp = 0.0633
temp = np.linspace(0,40,101)

# load empirical data
temp_traits = pd.read_csv("empirical_data/temperature_traits.csv",
                          encoding = 'ISO-8859-1')
# rename columns
temp_traits = temp_traits[["Optimum", "Niche.width"]]
temp_traits.columns = ["T_opt", "T_sig"]

# exclude very wide temporal niches
temp_traits = temp_traits[temp_traits["T_sig"]<=40]


temp_normal = pd.DataFrame(data = np.nan, index = ["mean", "sig"],
                           columns = temp_traits.columns)
# remove outliers
for key in temp_traits.columns:
    var = temp_traits[key].values
    iqr = np.nanpercentile(var, [25,75])
    iqr = iqr + 1.5*(iqr[1]-iqr[0])*np.array([-1,1])
    var[(iqr[0]>var) | (iqr[1]<var)] = np.nan
    temp_normal.loc["mean", key] = np.nanmean(var)
    temp_normal.loc["sig", key] = np.nanstd(var)

##############################################################################
# zooplankton temperature traits
Q_mu = 2.72-1.96*0.26 # minimal Q10 from Hansen & Bjornsen 1997
exp_mu = np.log(Q_mu)/10 # exponential factor for 

if __name__ == "__main__":
    fig, ax = plt.subplots(2,2, figsize = (9,9), sharex = "col")
    
    temp_traits_org = pd.read_csv("empirical_data/temperature_traits.csv",
                          encoding = 'ISO-8859-1')
    # rename columns
    temp_traits_org = temp_traits_org[["Optimum", "Niche.width"]]
    temp_traits_org.columns = temp_traits.columns
    
    for i, key in enumerate(temp_traits.columns):
        bins = np.linspace(min(temp_traits_org[key]), max(temp_traits_org[key]),51)
        ax[i,i].hist(temp_traits_org[key], label = "All measurments", bins = bins)
        ax[i,i].hist(temp_traits[key], label = "no_outliers", bins = bins,
                     alpha = 0.5)
        
        ax[i,i].hist(np.random.normal(temp_normal.loc["mean", key],
                                      temp_normal.loc["sig", key],
                                      np.sum(np.isfinite(temp_traits[key]))),
                     label = "resampled", alpha = 0.5, bins = bins)
        ax[i,i].set_xlabel(key, fontsize = 16)
        ax[i,i].set_ylabel("Frequency")
        ax[i,i].legend()
        
    ax[0,1].scatter(temp_traits_org["T_sig"], temp_traits_org["T_opt"], s = 2)
    ax[0,1].scatter(temp_traits["T_sig"], temp_traits["T_opt"], s = 2,
                    alpha = 0.5)
    ax[0,1].set_xlabel("T_sig")
    ax[0,1].set_ylabel("T_opt")
    
    temp_range = np.linspace(0,50, 101)
    for i, row in temp_traits.iterrows():
        if i >20:
            continue
        ax[1,0].plot(temp, a_epp*np.exp(b_epp*temp)*
                     (1-4*(temp - row["T_opt"])**2/row["T_sig"]**2))
    ax[1,0].plot(temp, a_epp*np.exp(b_epp*temp), 'k')
    
    ax[1,0].set_ylim([0, None])
    ax[1,0].set_xlabel("Temperature")
    ax[1,0].set_ylabel("Maximum growth rate")
    
    fig.tight_layout()
    
    fig.savefig("Figure_int_temp_traits.pdf")
