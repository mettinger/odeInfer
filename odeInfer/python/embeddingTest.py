#%%
import numpy as np
import pysindy as ps
from julia import Main
import pandas as pd
import plotly.express as px
from epileptor import epileptor
from scipy.integrate import solve_ivp

#%%
def plot3d(array, title):
    df = pd.DataFrame(array)
    fig = px.line_3d(df, x=0, y=1, z=2, title=title)
    fig.show()

#%%
y0 = [0,0,0,0,0,0]
timeStart = 0
timeEnd = 100

numPoint = 10**4
t_eval = np.linspace(timeStart, timeEnd, numPoint)

odeSolution = solve_ivp(epileptor, [timeStart, timeEnd], y0, t_eval=t_eval)
trajectory = odeSolution.y.transpose()

#%%
Main.include("../julia/delayEmbed.jl")

w = 0
Tmax = 100
embeddedY, τ_vals, ts_vals, traj = Main.embed(trajectory[:,0:6], w, Tmax)

#%%
plot3d(trajectory[:,0:3], 'test')



#%%
Main.include("../julia/delayEmbed.jl")
modelList = [None, None, None]
for obs in [1,2,3]:
    embedding, delay_values, ts_indices, trajectory, dt  = Main.lorenzEmbed(obs)
    sparse_regression_optimizer = ps.STLSQ(threshold=0)
    modelList[obs-1] = ps.SINDy(optimizer=sparse_regression_optimizer)
    modelList[obs-1].fit(embedding, t=dt)

