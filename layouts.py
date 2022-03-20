from matplotlib.cm import Set2 as cmap
import numpy as np

labels = {'size_P': "$V_i^P$",
          'mu_P': "$\mu_i^P$",
          'k_n': "$k_{iN}^P$",
          'k_p': "$k_{iP}^P$",
          'k_l': "$k_{iL}^P$",
          'c_n': "$c_{iN}^P$",
          'c_p': "$c_{iP}^P$",
          'a': "$a_i^P$",
          'e_P': "$e_i^P$",
          'R_P': "$w_i^P$",
          'size_Z': "$V_j^P",
          'mu_Z': "$\mu_j^Z$",
          'c_Z': "$c_j^Z$",
          'm_Z': "$m_j^Z$",
          'k_Z': "$k_j^Z$",
          "h_zp": "$h_{ji}$",
          "s_zp": "$s_{ji}$"
          }


def color_fun(i, n):
    return cmap(np.linspace(0, 1, n))[i]


color = {}
ax_id = {}

growth_traits = ["mu_P", "mu_Z", "m_Z"]
resource_traits = ["a", "c_n", "c_p", "k_l", "k_n", "k_p"]
interaction_traits = ["R_P", "k_Z", "c_Z", "e_P", "h_zp", "s_zp"]

for i, x in enumerate([growth_traits, resource_traits, interaction_traits]):
    ax_id.update({trait: i for trait in x})
    color.update({trait: color_fun(j, len(x)) for j, trait in enumerate(x)})

gleaner = ["c_Z:m_Z", "c_Z:k_Z", "c_Z:mu_Z", "c_n:k_n", "c_p:k_p", "k_Z:mu_Z"]
defense = ["R_P:mu_P", "R_P:k_n"]
super_resource = ["R_P:a", "R_P:e_P"]
resources = ["c_n:c_p", "a:c_n", "a:c_p", "k_n:k_p", "k_l:k_n", "k_l:k_p"]
corrs = gleaner + defense + super_resource + resources
for i, x in enumerate([gleaner, defense, super_resource, resources]):
    ax_id.update({corr: i for corr in x})
    color.update({trait: color_fun(j, len(x)) for j, trait in enumerate(x)})
