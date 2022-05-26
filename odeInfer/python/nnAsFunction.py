#%%
import torch
from torch import nn

#%%
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

#%%
path = './limitCycle.pt'
device = 'cuda'

model = NeuralNetwork()
model.load_state_dict(torch.load(path))
model.to(device)
model.eval()

# %%
def modelEval(testPoint):
    result = model(torch.Tensor(testPoint).to(device)).cpu().detach().numpy()
    return result
