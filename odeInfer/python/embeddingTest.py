#%%
import numpy as np
import pysindy as ps
import itertools

import julia
from julia import DynamicalSystems
from julia import Main

#%%

# LORENZE MODEL
def lorenz(x, y, z):
    sigma, rho, beta = 10, 8/3, 28
    xDot = sigma * (y - x)
    yDot = (x * (rho - z)) - y
    zDot = (x * y) - (beta * z)
    return xDot, yDot, zDot

# %%

x1s = np.linspace(-2, 2, 40)
y1s = np.linspace(-2, 2, 40)
zs = np.linspace(-2, 2, 40)
grid = [np.array((x1, y1, z)) for x1 in x1s for y1 in y1s for z in zs]

x = [i.reshape((1,3)) for i in grid]
x_dot = [np.array(lorenz(*i)).reshape((1,3)) for i in grid]

# %%

model = ps.SINDy()
model.fit(x, t = None, x_dot=x_dot, multiple_trajectories=True)
model.print()

# %%
