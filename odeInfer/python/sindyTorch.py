#%%
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import itertools

# %%

# MAIN AND ACCESSORY FUNCTIONS FOR EPILEPTOR MODEL
def f1(x1, x2, z):
    if x1 < 0:
        temp = x1**3 - (3 * x1**2)
        return temp
    else:
        temp = (x2 - (.6 * ((z - 4)**2))) * x1 
        return temp

def f2(x2):
    if x2 < -.25:
        return 0
    else:
        temp = 6 * (x2 + .25)
        return temp
        
def epileptor(x1, y1, z, x2, y2, u):

    x0 = -1.6
    y0 = 1
    tao0 = 2857
    tao1 = 1
    tao2 = 10
    Irest1 = 3.1
    Irest2 = .45
    gamma = .01

    x1Dot = y1 - f1(x1, x2, z) - z + Irest1
    y1Dot = y0 -(5 * x1**2) - y1
    zDot = (1/tao0) * ((4 * (x1 - x0)) - z)
    x2Dot = -y2 + x2 - x2**3 + Irest2 + (2*u) - (.3 * (z - 3.5))
    y2Dot = (1/tao2) * (-y2 + f2(x2))
    uDot = -gamma * (u - (.1 * x1))

    return x1Dot, y1Dot, zDot, x2Dot, y2Dot, uDot

def lorenz(x, y, z):
    sigma, rho, beta = 10, 8/3, 28
    xDot = sigma * (y - x)
    yDot = (x * (rho - z)) - y
    zDot = (x * y) - (beta * z)
    return xDot, yDot, zDot

def gridGet(bounds, steps):
    x = [np.linspace(bounds[i][0], bounds[i][1], steps[i]) for i in range(len(steps))]
    return list(itertools.product(*x))

class epileptorDataset(Dataset):
    def __init__(self, odeSystem, bounds, step, basisFunctions):

        systemDim = len(step)
        numBasis = len(basisFunctions)
        grid = gridGet(bounds, step)
        xDot = [np.array(odeSystem(*i)).reshape((1, systemDim)) for i in grid]

        self.x = torch.Tensor([np.array([f(i) for f in basisFunctions]).reshape((1, numBasis)) for i in grid])
        self.xDot = torch.Tensor(xDot)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:], self.xDot[idx,:]

basisFunctions = [
    lambda x : 1,
    lambda x : x[0],
    lambda x : x[1],
    lambda x : x[2],
    lambda x : x[0] ** 2,
    lambda x : x[0] * x[1],
    lambda x : x[0] * x[2],
    lambda x : x[1] ** 2,
    lambda x : x[1] * x[2],
    lambda x : x[2] ** 2
]

def train_loop(dataloader, models, loss_fn, optimizers):
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        for modelIndex, thisModel in enumerate(models):
            thisModel.train()
            optimizer = optimizers[modelIndex]

            # Compute prediction and loss
            pred = thisModel(Variable(X.cuda()))
            loss = loss_fn(pred, Variable(y[:,:, modelIndex].cuda()))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def dataloaderGet(odeIndex, basisFunctions):

    if odeIndex == 0:
        odeSystem = epileptor
        bounds = [(0,5) for i in range(6)]
        steps = [6 for i in range(6)]
        batchSize = 64
    elif odeIndex == 1:
        odeSystem = lorenz
        bounds = [(-2,2), (-2,2), (-2,2)]
        steps = [40, 40, 40]
        batchSize = 64


    dataset = epileptorDataset(odeSystem, bounds, steps, basisFunctions)
    dataloader = DataLoader(dataset, batch_size=batchSize)

    systemDim = len(steps)
    return dataloader, systemDim

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out

# %%

odeIndex = 1
learningRate = .01
epochs = 10

dataloader, systemDim = dataloaderGet(odeIndex, basisFunctions)
numBasisFunctions = len(basisFunctions)
models = [linearRegression(numBasisFunctions, 1).cuda() for i in range(systemDim)]

# change loss function for l1!
lossFunction = torch.nn.MSELoss() 
optimizers = [torch.optim.SGD(thisModel.parameters(), lr=learningRate) for thisModel in models]

#%%

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloader, models, lossFunction, optimizers)
print("Done!")


# %%