import numpy as np
import matplotlib.pyplot as plt
import phytoplankton_traits as pt
import pandas as pd

# define gaussian covariance matrix and mean
means = pt.gaussians.loc[["size_P", "k_n", "k_p", "mu_P"],
                                     "mean_trait"]

# contains the traits size, k_n, k_p and mu
A = np.zeros((4,4))
np.fill_diagonal(A, pt.gaussians.loc[["size_P", "k_n", "k_p", "mu_P"],
                                     "std_trait"])

# fill in allometric scaling
A[1:,0] = A[0,0]*pt.gaussians.loc[["k_n", "k_p", "mu_P"], "beta_min"]
A[0,1:] = A[0,0]*pt.gaussians.loc[["k_n", "k_p", "mu_P"], "beta_min"]
A[[1,2],[2,1]] = A[0,0]*np.prod(pt.gaussians.loc[["k_n", "k_p"], "beta_min"])
A[[1,3],[3,1]] = A[0,0]*np.prod(pt.gaussians.loc[["k_n", "mu_P"], "beta_min"])
A[[2,3],[3,2]] = A[0,0]*np.prod(pt.gaussians.loc[["k_p", "mu_P"], "beta_min"])
itera = int(1e4)
data = np.random.multivariate_normal(means, A, itera).T

ax_1 = np.array([0.17, 0.65, -0.74]) # first pca of data
ax_2 = np.array([0.46, -0.86, 0.21]) # second pca of data

# change to normal vector for equation format of plane a*x + b*y + c*z + d = 0
normal_vec = np.cross(ax_1, ax_2)
normal_vec = np.append(normal_vec, -normal_vec[0] -normal_vec[1])

###############################################################################
# given these parameters, find the optimal multivariate gaussian

itera = 100
A_all = np.empty((itera, itera, itera, len(A), len(A)))
A_all[:,:,:] = A.copy()

# covariance of k_n and k_p
A_all[...,1,2] = np.linspace(-A[1,1]*A[2,2],A[1,1]*A[2,2], itera)
A_all[...,2,1] = A_all[...,1,2]
# covariance of k_n, mu
A_all[...,1,3] = np.linspace(-A[1,1]*A[3,3], A[1,1]*A[3,3]
                             , itera)[:,np.newaxis]
A_all[...,3,1] = A_all[...,1,3]
# covariance of k_p and mu
A_all[...,2,3] = np.linspace(-A[2,2]*A[3,3], A[2,2]*A[3,3],
                             itera)[:,np.newaxis,np.newaxis]
A_all[...,3,2] = A_all[...,2,3]


norm = np.nansum(normal_vec[:,np.newaxis]*A_all*normal_vec, axis = (-1,-2))
semi_pos = np.amin(np.linalg.eigvalsh(A_all), axis = -1)>0
norm[~semi_pos] = 1e5 # not to be selected
best = np.where(norm == np.amin(norm))

A_tradeoff = A_all[best][0]

data = np.random.multivariate_normal(means, A_tradeoff, int(1e4)).T
data2 = np.random.multivariate_normal(means, A, int(1e4)).T

A_tradeoff = pd.DataFrame(A_tradeoff, index = ["size_P", "k_n", "k_p", "mu_P"],
                          columns = ["size_P", "k_n", "k_p", "mu_P"])

A_tradeoff.to_csv("Three_way_tradeoff.csv", index = True)

def plot_multivariate(X, names = None, figure = None):
    n_var = len(X)
    if n_var > 10:
        return
    if names is None:
        names = np.arange(n_var)
    if figure is None:    
        fig, ax = plt.subplots(n_var, n_var, sharex = "col", sharey = "row")
    else:
        fig, ax = figure
    
    for i in range(n_var):
        for j in range(n_var):
            if i<j:
                ax[i,j].scatter(X[j], X[i], s = 2, alpha = 0.01)
        ax_hist = fig.add_subplot(n_var, n_var, (n_var + 1)*i + 1)
        ax_hist.hist(X[i], density = True, bins = 15, alpha = 0.5)
        ax_hist.set_xticks([])
        ax_hist.set_yticks([])
        ax[i,0].set_ylabel(names[i])
        ax[-1,i].set_xlabel(names[i])
    return (fig, ax)

