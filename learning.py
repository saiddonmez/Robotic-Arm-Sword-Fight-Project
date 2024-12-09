import robotic as ry
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
from ddpgCode.model import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(folder):
    datas = []
    for path in os.listdir(folder):
        if path.endswith('.npy'):
            path = np.load(os.path.join(folder, path))
            for i in range(len(path)-1):
                state = path[i]
                goal = path[i+1]
                datas.append((state, goal))
    return np.array(datas)
    
def train(model, dataLoader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for i, (state, goal) in enumerate(dataLoader):
            state = state.to(device)
            goal = goal.to(device)
            optimizer.zero_grad()
            action = model(state)
            
            loss = F.mse_loss(output, goal)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Epoch: {}/{}, Iter: {}/{}, Loss: {}'.format(epoch, epochs, i, len(dataLoader), loss.item()))

datas = load_data('real_spline_attackPaths')
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print(np.min(datas[:,0,0,:], axis=0))
print(np.max(datas[:,0,0,:], axis=0))

print(np.min(datas[:,0,1,:], axis=0))
print(np.max(datas[:,0,1,:], axis=0))
#attackModel = Actor(28,7,400,300,0.003).to(device)


