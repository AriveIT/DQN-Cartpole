import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# pro tip: the target network and the network network need to have the 
# exact same architecture otherwise you cannot copy the weights between them.

class Model(nn.Module):
    def __init__(self,n_observation=4, n_actions=2):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(n_observation,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,128)
        self.fc4 = nn.Linear(128,n_actions)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x