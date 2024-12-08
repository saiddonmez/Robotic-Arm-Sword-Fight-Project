import rai
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import gymnasium_robotics as gymr
from torch.utils.data import DataLoader

class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DefenceModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def loadPathData(file):
    # load data from file
    data = torch.load(file)
    return data

def generateTrainLoader(data, batch_size=64):
    # generate data loader with batch size 32
    trainLoader = DataLoader(data, batch_size, shuffle=True)
    return trainLoader
    

def trainModel(trainLoader, epochs=10, lr=0.01):
    # train the model
    model = AttackModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for i, data in enumerate(trainLoader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
    return model

