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

traits = np.concatenate((gp.pt.phyto_traits[1:], gp.zt.zoop_traits[1:], ["h_zp"]))

add_mean = np.linspace(-1,1,n_prec)
df_mean = pd.DataFrame(np.nan, index = np.arange(n_prec*len(traits)),
                    columns = ["trait", "richness_phyto", "richness_zoo", "change"])

counter = 0
path = "C:/Users/Juerg Spaak/Documents/Science backup/TND/"

file = "assebly_mean_20_1000_{}_{}.npz"
for i in range(n_prec):
    for trait in traits:
        data = np.load(path + file.format(trait, i))
        
        
        df_mean.loc[counter] = [trait, *d_mean(data), data["change_mean"]]
        counter += 1
       
df_mean.to_csv("data/assembly_mean.csv", index = False)

#"""

##############################################################################
# changes in var

traits = np.concatenate((gp.pt.phyto_traits[1:], gp.zt.zoop_traits[1:], ["h_zp", "s_zp"]))

df_mean = pd.DataFrame(np.nan, index = np.arange(n_prec*len(traits)),
                    columns = ["trait", "richness_phyto", "richness_zoo", "change"])

counter = 0
file = "/assebly_var_20_1000_{}_{}.npz"
for i in range(n_prec):
    for trait in traits:

        data = np.load(path + file.format(trait, i))     
        
        df_mean.loc[counter] = [trait, *d_mean(data), data["change_var"]]
        counter += 1
       
df_mean.to_csv("data/assembly_var.csv", index = False)

##############################################################################
# simplified phytoplankton correlations


file = "assembly_corr_20_1000_{}_{}_{}.npz"
traits = gp.pt.phyto_traits[1:]

df_corr_phyto = pd.DataFrame(np.nan, index = np.arange(len(traits)*(len(traits)-1)),
                    columns = ["traiti", "traitj", "richness_phyto", "richness_zoo", "corr"])

df_phyto_phyto = pd.DataFrame(np.nan, index = traits, columns = traits)
df_phyto_zoo = df_phyto_phyto.copy()

counter = 0
for i, trait in enumerate(traits):
    for j, traiti in enumerate(traits):     
        if i==j:
            continue

        data = np.load(path + file.format(*sorted([trait, traiti]), 0 if i>j else n_prec-1))

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

file = "assembly_corr_20_1000_{}_{}_{}.npz"
traits = gp.zt.zoop_traits[1:]

df_corr_zoo = pd.DataFrame(np.nan, index = np.arange(len(traits)*(len(traits)-1)),
                    columns = ["traiti", "traitj", "richness_phyto", "richness_zoo", "corr"])

df_zoo_phyto = pd.DataFrame(np.nan, index = traits, columns = traits)
df_zoo_zoo = df_zoo_phyto.copy()

counter = 0
for i, trait in enumerate(traits):
    for j, traiti in enumerate(traits):
        if traiti == trait:
            continue       

        data = np.load(path + file.format(*sorted([trait, traiti]), 0 if i>j else n_prec-1))
        
        
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

file = "assembly_corr_20_1000_{}_{}_{}.npz"

corrs = ["{}:{}".format(*sorted([traiti, traitj]))
         for traiti in gp.pt.phyto_traits[1:]
         for traitj in gp.pt.phyto_traits[1:]]
corrs = corrs + ["{}:{}".format(*sorted([traiti, traitj]))
         for traiti in gp.zt.zoop_traits[1:]
         for traitj in gp.zt.zoop_traits[1:]]
corrs = list(set(corrs))

df_corr = pd.DataFrame(np.nan, index = np.arange(len(traits)*n_prec),
                    columns = ["tradeoff", "richness_phyto", "richness_zoo", "corr"])
counter = 0
for trait in corrs:
    traiti, traitj = trait.split(":")
    for i in np.arange(n_prec):
        try:
            data = np.load(path + file.format(traiti, traitj,i))
            df_corr.loc[counter] = [trait,  *d_mean(data), np.round(data["corr"],3)]
            counter += 1
        except FileNotFoundError:
            pass
        
    
df_corr.to_csv("data/assembly_corr.csv", index = False)
#"""

##############################################################################
# reference cases

traits = np.concatenate((gp.pt.phyto_traits[1:], gp.zt.zoop_traits[1:], ["h_zp"]))

df_ref = pd.DataFrame(np.nan, index = np.arange(2*len(traits)),
                    columns = ["richness_phyto", "richness_zoo"])

counter = 0

file = "assebly_{}_20_1000_{}_{}.npz"
for change in ["var", "mean"]:
    for trait in traits:
        data = np.load(path + file.format(change, trait, n_prec//2))
        
        
        df_ref.loc[counter] = d_mean(data)
        
        counter += 1
df_ref.to_csv("data/assembly_reference.csv", index = False)
