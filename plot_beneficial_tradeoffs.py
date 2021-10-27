import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import generate_plankton as gp
from phytoplankton_traits import corr_phyto, phyto_traits
from zoop_traits import corr_zoop, zoop_traits

# is it more beneficial to have large or low trait values
phyto_beneficial = pd.DataFrame({"size_P": np.nan,
                                 "mu_P": 1,
                                 "k_n": -1, "k_l": -1, "k_p": -1,
                                 "c_n": np.nan, "c_p": np.nan, "a": np.nan,
                                 "e_P": -1, "R_p": -1}, index = [1])

optimal_phyto = phyto_beneficial.values*phyto_beneficial.values.T
np.fill_diagonal(optimal_phyto, 0)
optimal_phyto = pd.DataFrame(-optimal_phyto, index = phyto_traits,
                             columns = phyto_traits)

# is it more beneficial to have large or low trait values
zoop_beneficial = pd.DataFrame({"size_Z": np.nan,
                                 "mu_Z": 1,
                                 "c_Z": 1, "m_Z": -1, "k_Z":-1}, index = [1])

optimal_zoop = zoop_beneficial.values*zoop_beneficial.values.T
np.fill_diagonal(optimal_zoop, 0)
optimal_zoop = pd.DataFrame(-optimal_zoop, index = zoop_traits,
                             columns = zoop_traits)

fig, ax = plt.subplots(2,2, sharex = "row", sharey = "row", figsize = (9,9))
ax[0,0].imshow(optimal_phyto, cmap = "RdBu")
ax[0,1].imshow(corr_phyto, cmap = "RdBu", vmin = -1, vmax = 1)


ax[0,0].set_xticks(np.arange(len(phyto_traits)))
ax[0,0].set_xticklabels(phyto_traits, rotation = 90)
ax[0,1].set_xticklabels(phyto_traits, rotation = 90)
ax[0,0].set_yticks(np.arange(len(phyto_traits)))
ax[0,0].set_yticklabels(phyto_traits)


ax[1,0].imshow(optimal_zoop, cmap = "RdBu")
ax[1,1].imshow(corr_zoop, cmap = "RdBu", vmin = -1, vmax = 1)


ax[1,0].set_xticks(np.arange(len(zoop_traits)))
ax[1,0].set_xticklabels(zoop_traits, rotation = 90)
ax[1,1].set_xticklabels(zoop_traits, rotation = 90)
ax[1,0].set_yticks(np.arange(len(zoop_traits)))
ax[1,0].set_yticklabels(zoop_traits)