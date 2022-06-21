#%%  IMPORTS

import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import ToTensor


import julia
#julia.install()
from julia import DynamicalSystems
from julia import Main

#%%  NEURAL NETWORK CLASS AND FUNCTIONS

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        yHat = self.linear_relu_stack(x)
        return yHat

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def cycleFuturePoint(x0, y0, t):

    alpha = 1

    rho0 = np.abs(complex(x0, y0))
    phi0 = np.angle(complex(x0, y0))

    rho = ( (1/alpha) + (((1/rho0**2) - (1/alpha)) * np.exp(-2 * alpha * t)) ) ** -.5
    phi = phi0 + t

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def derivativeApprox(xlag, deltaTime):
    xdot = np.diff(xlag, axis = 0)/deltaTime
    xdot = np.array([(xdot[i,:] + xdot[i + 1,:])/2 for i in range(xdot.shape[0] - 1)])
    return xlag[1:-1,:], xdot

def flowGet(x0, y0, t, futureFunction, plotFlag=False):
    flow = np.array([futureFunction(x0, y0, i) for i in t])

    if plotFlag:
        plt.figure()
        plt.plot(flow[:,0], flow[:,1], '.')
        plt.pause(.001)

    return flow

def combineFlows(initialPoints, t, futureFunction):
    deltaTime = t[1] - t[0]
    flows = [flowGet(i[0], i[1], t, futureFunction) for i in initialPoints]
    flowVectorPairs = [derivativeApprox(i, deltaTime) for i in flows]
    xlag = np.vstack([i[0] for i in flowVectorPairs])
    xdot = np.vstack([i[1] for i in flowVectorPairs])

    return xlag, xdot


def limitCycleData(initialPoints, finalTime, timeSteps, plotFlag=True):

    t = np.linspace(0, finalTime, timeSteps)

    xlag, xdot = combineFlows(initialPoints, t, cycleFuturePoint)

    if plotFlag:
        plt.figure()
        plt.plot(xlag[:,0], xlag[:,1], '.')
        #plt.gca().set_aspect('equal')
        plt.title('Initial Data in Phase Space')
        plt.pause(.001)
    
    return xlag, xdot

#%%  CREATE SIMULATED DATA

finalTime = 10
timeSteps = 1000
batch_size = 64

initialPoints = [(0., 2.), (.5, 0.), (0., -2.), (-.5, 0.)]
x, xdot = limitCycleData(initialPoints, finalTime, timeSteps)

tensor_x = torch.Tensor(x)
tensor_y = torch.Tensor(xdot)
training_data = TensorDataset(tensor_x,tensor_y)
train_dataloader = DataLoader(training_data, batch_size=batch_size)

#%% USE JULIA TO FIND EMBEDDING

Main.s = x[:,0]
Main.eval('theiler = DynamicalSystems.estimate_delay(s, "mi_min")')
Main.eval("Tmax = 100")
Y, t_vals, ts_vals, Ls, eps = Main.eval("DynamicalSystems.pecuzal_embedding(s; τs = 0:Tmax , w = theiler, econ = true)")

#%%
YY = np.array(Y.data)
plt.plot(YY[:,0], YY[:,1])

#%%   FIT THE MODEL TO THE DATA

epochs = 1000
epochReportPeriod = 200

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NeuralNetwork().to(device)
loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for t in range(epochs):
    loss = train(train_dataloader, model, loss_fn, optimizer)
    if t % epochReportPeriod == 0:
        print("loss: " + str(loss))
        print(f"Epoch {t}\n-------------------------------")

print("Done!")

predictions = model(tensor_x.to(device)).cpu().detach().numpy()
resid = predictions - xdot

# %%

# GRAPH THE ESTIMATED FIELD

xRange = np.linspace(-2,2,21)
yRange = np.linspace(-2,2,21)

grid = np.array([(i,j) for i in xRange for j in yRange])
gridPredictions = model(torch.Tensor(grid).to(device)).cpu().detach().numpy()

xx = grid[:,0]
yy = grid[:,1]
uu = gridPredictions[:,0]
vv = gridPredictions[:,1]

fig = ff.create_quiver(xx, yy, uu, vv) 
fig.update_layout(title="Estimated Phase Portrait")
fig.show()

fig.write_image("estimatedPortrait.png")
torch.save(model.state_dict(), "limitCycle.pt")

#%%
'''
def simpleSineData():
    numSamples = 10000
    numPeriod = 100
    dim = 2
    lag = int((numSamples/(numPeriod * 2 * np.pi)) * np.pi/2)

    t = np.linspace(0, numPeriod * 2 * np.pi, numSamples)
    deltaT = t[1] - t[0]
    x = np.sin(t)

    xlag = np.vstack([x[(dim - 1 - i) * lag : len(x) - (i * lag)] for i in range(dim)]).transpose()
    xdot = np.diff(xlag, axis = 0)/ deltaT
    xdot = np.array([(xdot[i,:] + xdot[i + 1,:])/2 for i in range(xdot.shape[0] - 1)])

    plt.figure()
    plt.plot(x[0:200])
    plt.xlabel('Time')
    plt.ylabel('Observation')
    plt.title('Original Signal')

    gap = 5

    xx = xlag[1:-1:gap,0]
    yy = xlag[1:-1:gap,1]
    uu = xdot[::gap,0]
    vv = xdot[::gap,1]

    fig = ff.create_quiver(xx, yy, uu, vv) 
    fig.update_layout(title="Phase Portrait Input Data")
    fig.show()

    return xdot, xlag 

#xdot, xlag = simpleSineData()
'''