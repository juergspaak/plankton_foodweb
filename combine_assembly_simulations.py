import numpy as np
import pandas as pd

import generate_plankton as gp

def d_mean(data):
    if data["present"].shape != (1000,2,20,21):
        raise
    return np.mean(np.sum(data["present"][...,-1],axis = -1), axis = 0)

n_prec = 9

##############################################################################
# changes in mean
"""
traits = np.concatenate((gp.pt.phyto_traits[1:], gp.zt.zoop_traits[1:], ["h_zp"]))

add_mean = np.linspace(-1,1,n_prec)
df_mean = pd.DataFrame(np.nan, index = np.arange(n_prec*len(traits)),
                    columns = ["trait", "richness_phyto", "richness_zoo", "change"])

counter = 0
path = "C:/Users/Juerg Spaak/Documents/Science backup/Trophic network_data/"
#path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"
file2 = "assembly_mean_{}_{}.npz"
file = "../TND/assebly_mean_20_1000_{}_{}.npz"
for i in range(n_prec):
    for trait in traits:
        data = np.load(path + file.format(trait, i))
        
        
        df_mean.loc[counter] = [trait, *d_mean(data), data["change_mean"]]
        
        # from old data saving method
        #data = np.load(path + file2.format(trait, i))
        #df_mean.loc[counter] = [trait, *d_mean(data), add_mean[i]]       
        #np.savez(path2 + file2.format(trait, i), **data, change_mean = add_mean[i])
        counter += 1
       
df_mean.to_csv("data/assembly_mean.csv", index = False)

#"""

##############################################################################
# changes in var

traits = np.concatenate((gp.pt.phyto_traits[1:], gp.zt.zoop_traits[1:], ["h_zp", "s_zp"]))

df_mean = pd.DataFrame(np.nan, index = np.arange(n_prec*len(traits)),
                    columns = ["trait", "richness_phyto", "richness_zoo", "change"])

counter = 0
path = "C:/Users/Juerg Spaak/Documents/Science backup/Trophic network_data/"
#path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"
file = "../TND/assebly_var_20_1000_{}_{}.npz"
for i in range(n_prec):
    for trait in traits:
        try:
            data = np.load(path + file.format(trait, i))
            data["change_var"]
        except FileNotFoundError:
            continue
        except KeyError:
            continue
        
        
        df_mean.loc[counter] = [trait, *d_mean(data), data["change_var"]]
        counter += 1
       
df_mean.to_csv("data/assembly_var.csv", index = False)
"""
##############################################################################
# simplified phytoplankton correlations


file2 = "../TND/assembly_corr_20_1000_{}_{}_{}.npz"
traits = gp.pt.phyto_traits[1:]

df_corr_phyto = pd.DataFrame(np.nan, index = np.arange(len(traits)**2),
                    columns = ["traiti", "traitj", "richness_phyto", "richness_zoo", "corr"])

df_phyto_phyto = pd.DataFrame(np.nan, index = traits, columns = traits)
df_phyto_zoo = df_phyto_phyto.copy()

counter = 0
for i, trait in enumerate(traits):
    for j, traiti in enumerate(traits):
        if traiti == trait:
            df_corr_phyto.loc[counter] = [trait, traiti, np.nan, np.nan, np.nan]
            counter += 1
            continue       
        
        data = np.load(path + file2.format(*sorted([trait, traiti]), 0 if i>j else n_prec-1))
        '''
        file = "sim_simplified_corr_phyto_20_1000_{}_{}.npz"
        # old data saveing method
        data = np.load(path + file.format(trait, traiti))
        corr = brentq(fun, gp.pt.corr_phyto.loc[trait,traiti],
                                            -1 if i>j else 1,
                                    args = (trait, traiti, gp.pt.corr_phyto.copy()))
        corr_test = np.corrcoef(np.log(data[trait].flatten()), np.log(data[traiti].flatten()))
        if np.abs(corr-corr_test[0,1])>1e-2:
            print(corr, corr_test[0,1], trait, traiti, data["present"].shape)
        np.savez(path + file2.format(*sorted([trait, traiti]), 0 if i>j else 8), **data, corr = corr)
        '''
        
        richness = d_mean(data)
        df_corr_phyto.loc[counter] = [trait, traiti, *richness, np.nan]
        
        df_phyto_phyto.loc[trait, traiti] = richness[0]
        df_phyto_zoo.loc[trait, traiti] = richness[1]
        counter += 1
        
        #

df_phyto_phyto.to_csv("data/simplified_corr_phyto_traits_phyto.csv")
df_phyto_zoo.to_csv("data/simplified_corr_phyto_traits_zoo.csv")
df_corr_phyto.to_csv("data/assembly_corr_phyto.csv", index = False)
#"""
##############################################################################
# simplified zoop correlations
"""
file2 = "../TND/assembly_corr_20_1000_{}_{}_{}.npz"
traits = gp.zt.zoop_traits[1:]

df_corr_zoo = pd.DataFrame(np.nan, index = np.arange(len(traits)**2),
                    columns = ["traiti", "traitj", "richness_phyto", "richness_zoo", "corr"])

df_zoo_phyto = pd.DataFrame(np.nan, index = traits, columns = traits)
df_zoo_zoo = df_zoo_phyto.copy()

counter = 0
for i, trait in enumerate(traits):
    for j, traiti in enumerate(traits):
        if traiti == trait:
            df_corr_phyto.loc[counter] = [trait, traiti, np.nan, np.nan, np.nan]
            counter += 1
            continue       
        try:
            data = np.load(path + file2.format(*sorted([trait, traiti]), 0 if i>j else n_prec-1))
        except FileNotFoundError:
            counter += 1
            continue
        
        '''
        # old data saveing method
        data = np.load(path + file.format(trait, traiti))
        corr = brentq(fun, gp.pt.corr_phyto.loc[trait,traiti],
                                            -1 if i>j else 1,
                                    args = (trait, traiti, gp.pt.corr_phyto.copy()))
        corr_test = np.corrcoef(np.log(data[trait].flatten()), np.log(data[traiti].flatten()))
        if np.abs(corr-corr_test[0,1])>1e-2:
            print(corr, corr_test[0,1], trait, traiti, data["present"].shape)
        np.savez(path + file2.format(*sorted([trait, traiti]), 0 if i>j else 8), **data, corr = corr)
        '''
        
        richness = d_mean(data)
        df_corr_zoo.loc[counter] = [trait, traiti, *richness, np.nan]
        
        df_zoo_phyto.loc[trait, traiti] = richness[0]
        df_zoo_zoo.loc[trait, traiti] = richness[1]
        counter += 1
        
        #

df_zoo_phyto.to_csv("data/simplified_corr_zoo_traits_phyto.csv")
df_zoo_zoo.to_csv("data/simplified_corr_zoo_traits_zoo.csv")
df_corr_zoo.to_csv("data/assembly_corr_zoo.csv", index = False)
#"""

##############################################################################
# correlations

file = "assembly_cor_{}_{}_{}.npz"
file2 = "../TND/assembly_corr_20_1000_{}_{}_{}.npz"

zoop = ["c_Z:k_Z", "c_Z:mu_Z", "c_Z:m_Z", "k_Z:mu_Z", "m_Z:mu_Z"]
gleaner = ["c_n:k_n", "c_p:k_p", "a:k_l"]
resources = ["a:c_n", "a:c_p", "c_n:c_p", "k_n:k_p", "k_l:k_n", "k_l:k_p",]
growth_defense = ["R_p:e_P", "R_P:mu_P", "e_P:mu_P"]
weird = ["c_n:mu_P", "R_P:k_n"]
corrs = zoop + gleaner + resources + weird + growth_defense

df_corr = pd.DataFrame(np.nan, index = np.arange(len(traits)*n_prec),
                    columns = ["tradeoff", "richness_phyto", "richness_zoo", "corr"])
counter = 0
for trait in corrs:
    traiti, traitj = trait.split(":")
    for i in np.arange(n_prec):
        try:
            data = np.load(path + file2.format(traiti, traitj,i))
            #np.savez(path + file2.format(*sorted([traiti, traitj]), i), **data)
            df_corr.loc[counter] = [trait,  *d_mean(data), np.round(data["corr"],3)]
            counter += 1
        except FileNotFoundError:
            #df_corr.loc[counter] = [trait,  np.nan, np.nan, np.round(data["corr"],3)]
            counter += 1
            print(trait, i)
            continue
        
    
df_corr.to_csv("data/assembly_corr.csv", index = False)
#"""