# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:45:01 2018

@author: daniel
"""

#%% Define the network structure
import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(3, 5),
            nn.Sigmoid(),
            nn.Linear(5, 4),
        )

    def forward(self, x):
        x = self.hidden(x)
        return x

net = Net()
print(net)

#%% Define loss and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1, momentum=0.9)

#%% Data
X = torch.tensor([[0,0,0],
                  [0,0,1],
                  [0,1,0],
                  [1,0,0],
                  [0,1,1],
                  [1,0,1],
                  [1,1,0],
                  [1,1,1]], dtype=torch.float)
y = torch.tensor([0, 1, 1, 1, 2, 2, 2, 3], dtype=torch.long)

#%% Training
for epoch in range(5000):    
    # Reset the grads
    optimizer.zero_grad()
    
    # Forward + backward
    outputs = net(X)
    loss = criterion(outputs, y)
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    if epoch % 100 == 99:
        print('[{}] loss: {}'.format(epoch + 1, loss.item()))
        
#%% Testing
predict = lambda x: torch.max(net(x), 1)[1][0].item()

x1 = torch.tensor([[0, 1, 0]], dtype=torch.float)
x2 = torch.tensor([[1, 0, 1]], dtype=torch.float)
x3 = torch.tensor([[1, 1, 1]], dtype=torch.float)

print("[0, 1, 0] ->", predict(x1)) # [0, 1, 0] -> 1
print("[1, 0, 1] ->", predict(x2)) # [1, 0, 1] -> 2
print("[1, 1, 1] ->", predict(x3)) # [1, 1, 1] -> 3