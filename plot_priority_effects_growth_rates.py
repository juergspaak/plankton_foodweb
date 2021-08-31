import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 1.2],[1.2, 1]])
mu = np.array([2, 3])*2


d = np.array([0.8,2])*5


z_crit = (A[0]*mu[1]-A[1]*mu[0])/(A[0]*d[1]-A[1]*d[0])
equi_crit = (mu-d*z_crit[:,np.newaxis])/np.diag(A)
equi_crit = np.diag(equi_crit)
print(z_crit)
Z = np.linspace(0,1, 100)
equilibria = (mu-d*Z[:,np.newaxis])/np.diag(A)
def LV_prey(Z):
    try:
        N = np.zeros((len(Z),2,2))
    except:
        N = np.zeros((2,2))
    N[...,0,0] = ((mu -d*Z)/np.diag(A))[...,0].T
    N[...,1,1] = ((mu -d*Z)/np.diag(A))[...,1].T
    return N,mu -d*Z-np.einsum("ij, ...nj->...ni", A, N)

plt.plot(Z, equilibria, '--')


N, dN_dt = LV_prey(Z.reshape((-1,1,1)))

N[dN_dt[:,0,1]>0,0] = np.nan
N[dN_dt[:,1,0]>0,1] = np.nan
N[N<0] = 0


fig, ax = plt.subplots(3,sharex = True, figsize = (4,9))

ax[0].plot(Z, N[:,0,0], 'r')
ax[0].plot(Z, N[:,1,1], 'b')
ax[0].plot(Z, equilibria[:,0], "r--")
ax[0].plot(Z, equilibria[:,1], "b--")
ax[0].axhline(0, color = "k")
ax[-1].set_xlabel("Zooplankton density")
ax[0].set_ylabel("Phytoplankton\ndensity")
ax[0].plot(z_crit, equi_crit, 'ko')

e = np.array([1,1])-0.5
zoop_growth = np.sum(N*e, axis = -1) -0.5
ax[1].plot(Z, zoop_growth[:,0], 'r')
ax[1].plot(Z, zoop_growth[:,1], "b")
ax[1].axhline(0, color = "k")
ax[1].set_ylabel("Zoop_growth")
ax[1].set_title("Non-monotone growth rate")

e = np.array([0.3,1])
zoop_growth = np.sum(N*e, axis = -1)-1.5
ax[2].plot(Z, zoop_growth[:,0], 'r')
ax[2].plot(Z, zoop_growth[:,1], "b")
ax[2].axhline(0, color = "k")
ax[2].set_ylabel("Zoop_growth")
ax[2].set_title("No equilibrium density")

fig.savefig("Figure_priority_effects.pdf")