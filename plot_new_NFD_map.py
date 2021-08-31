import numpy as np
import matplotlib.pyplot as plt


# create random dots on the sphere
itera = 1000
theta = np.random.uniform(0, np.pi, itera) # polar angle
phi = np.random.uniform(0, 2*np.pi, itera) # azimutal angle

sphere = np.array([np.cos(phi)*np.sin(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(theta)])

# create dots where mu = 0
circles = {}
circle_prec = 91
theta = np.linspace(0, 2*np.pi, circle_prec)
phi = np.pi/2

circles["mu_0"] = np.array([np.cos(phi)*np.sin(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(theta)])

theta = np.linspace(0, 2*np.pi, circle_prec)
phi = 0

circles["ri_0"] = np.array([np.cos(phi)*np.sin(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(theta)])

theta = np.full(circle_prec, np.pi/2)
phi = np.linspace(0,2*np.pi,circle_prec)

circles["eta_0"] = np.array([np.cos(phi)*np.sin(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(theta)])

# circle where mu = ri
theta = np.linspace(0, 2*np.pi, circle_prec)
phi = np.pi/4
circles["mu_ri"] = np.array([np.cos(phi)*np.sin(theta),
                            np.sin(phi)*np.sin(theta),
                            np.cos(theta)])

# circle where mu = eta
angle = -np.pi/4
A = np.array([[np.cos(angle),0,np.sin(angle)],
             [0,1,0],
             [-np.sin(angle), 0, np.cos(angle)]])
circles["mu_eta"] = A.dot(circles["eta_0"])

# circle where ri = eta
angle = -np.pi/4
A = np.array([[1,0,0],
              [0,np.cos(angle),np.sin(angle)],
             [0,-np.sin(angle), np.cos(angle)]])
circles["ri_eta"] = A.dot(circles["eta_0"])

fig = plt.figure()
ax_3d = fig.add_subplot(221, projection = "3d")
ax_3d.set_ylim([1,-1])
ax_3d.set_xlabel("x")
ax_3d.set_ylabel("y")

colors = {"mu_0":"red", "ri_0":"yellow", "eta_0":"blue",
          "mu_ri": "orange", "mu_eta": "purple", "ri_eta":"green"}

#ax_3d.scatter(*sphere, s = 1, alpha = 0.5)
for key in circles.keys():
    ax_3d.scatter(*circles[key], s = 1, color = colors[key])

def project_general(dots, anker = np.array([0,0,-1]), x = np.array([1,0,0]),
                y = np.array([0,1,0])):
    x = x/np.linalg.norm(x)
    y = y/np.linalg.norm(y)
    anker = anker/np.linalg.norm(anker)
    s = 1-anker.dot(dots)
    return np.array([x.dot(dots), y.dot(dots)])/s

def project_new(dots, anker = np.array([0,0,-1]), x = np.array([1,0,0]),
                y = np.array([0,1,0])):
    
    x = x/np.linalg.norm(x) # project onto sphere
    y = y/np.linalg.norm(y) # project onto spere
    anker = anker/np.linalg.norm(anker)
    
    # project onto plane through origin
    dots = np.append(np.array([x,y]).T,dots, axis = 1)
    tau = 1/(1-anker.dot(dots))
    project_plane = anker[:,np.newaxis] + tau*(dots-anker[:,np.newaxis])
    A = np.append(project_plane[:,:2], anker[:,np.newaxis], axis = 1)
    # project plane onto R2
    return np.linalg.solve(A, project_plane)[:2,2:]



ax_NF2 = fig.add_subplot(222)
for key in circles.keys():
    ax_NF2.scatter(*project_general(circles[key]), s = 1, color = colors[key])
ax_NF2.set_xlim([-3,3])
ax_NF2.set_ylim([-3,3])

for key in circles.keys():
    ax_NF2.scatter(*project_new(circles[key], anker = [0,0,-1],
                                x = [1,0,0], y = [0,1,0]), s = 1, color = colors[key])
ax_NF2.set_xlim([-3,3])
ax_NF2.set_ylim([-3,3])

ax_NF = fig.add_subplot(223)
for key in circles.keys():
    ax_NF.scatter(*project_new(circles[key], anker = [-1,-1,-1],
                                x = [0,1,1], y = [1,0,1]),
                  s = 1, color = colors[key])
x = 10
ax_NF.set_xlim([-x,x])
ax_NF.set_ylim([-x,x])

ax_NF = fig.add_subplot(224)
for key in circles.keys():
    ax_NF.scatter(*project_new(circles[key], anker = [0.5,0.1,0.3],
                                x = [1,0,0], y = [0,1,0]),
                  s = 1, color = colors[key])
x = 10
ax_NF.set_xlim([-x,x])
ax_NF.set_ylim([-x,x])