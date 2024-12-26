import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
from .replayBuffer import ReplayBuffer
import numpy as np
import torch.nn.functional as F
import time

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim=10, action_dim=4, hidden_sizes=[256,256], minActions = None, maxActions = None):
        super(Actor, self).__init__()

        layers = []
        current_size = state_dim

        # Add hidden layers with activation
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        
        # Combine all layers in a Sequential module
        self.baseModel = nn.Sequential(*layers)

        self.meanLayer = nn.Linear(hidden_sizes[-1], action_dim)
        self.stdLayer = nn.Linear(hidden_sizes[-1], action_dim)
        self.max_log_std = 2
        self.min_log_std = -10

        self.minActions = minActions
        self.maxActions = maxActions

        self.actionScale = self.action_scale = torch.FloatTensor(
                (self.maxActions - self.minActions) / 2.)
        
        self.actionBias = torch.FloatTensor(
                (self.maxActions + self.minActions) / 2.)

    def forward(self, state):
        x = self.baseModel(state)

        mean = self.meanLayer(x)
        mean = F.tanh(mean) # mean is the mean of the gaussian distribution
        mean = mean * (self.maxActions - self.minActions) / 2.0 + (self.maxActions + self.minActions) / 2.0
        log_std = self.stdLayer(x) # log_std is the log of the standard deviation
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std) # std is the standard deviation
        gaussianDist = Normal(mean, std) # Gaussian distribution with mean and std
        return gaussianDist
    

    def sampleAction(self, state): # Sample an action from the gaussian distribution
        gaussianDist = self.forward(state) # Get the gaussian distribution
        action = gaussianDist.rsample() # Sample an action from the distribution
        action_y = torch.tanh(action)

        action = self.actionScale * action_y + self.actionBias

        logProbs = gaussianDist.log_prob(action) # Get the log probability of the action

        #action = action.clamp(min = self.minActions, max = self.maxActions)

        # Enforcing Action Bound
        logProbs -= torch.log(self.action_scale * (1 - action_y.pow(2)) + 1e-6)
        logProbs = logProbs.sum(-1,keepdims=True) # Sum the log probabilities
        #mean = torch.tanh(mean) * self.action_scale + self.action_bias  # for evaluation      
        return action, logProbs # Return the action and the log probabilities
    
    def save_model(self, path):
        torch.save(self.state_dict(), path) # Save the model to the path

class CriticQ(nn.Module):
    def __init__(self, state_dim=10, action_dim=4, hidden_sizes=[256,256]):
        super(CriticQ, self).__init__()

        layers = []
        current_size = state_dim + action_dim # The input size is the state size + action size

        # Add hidden layers with activation
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        
        # Add the output layer
        layers.append(nn.Linear(current_size, 1))
        
        # Combine all layers in a Sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1) # Concatenate the state and action
        return self.model(state_action) # Return the output of the model
    
    def save_model(self, path):
        torch.save(self.state_dict(), path) # Save the model to the path
    
class SAC:
    def __init__(self, state_dim=10, action_dim=4, lr=1e-3, gamma=0.99, tau=0.005, alpha=0.2, buffer_size=1000000, batch_size=512, device='cuda', load_file = None,minActions=None,maxActions=None):
        
        # Initialize parameters
        self.device = device    
        self.gamma = gamma
        self.tau = tau
        self.alpha = torch.Tensor([alpha]).to(self.device)
        self.alpha.requires_grad = True
        self.batch_size = batch_size

        self.replayBuffer = ReplayBuffer(buffer_size, state_dim, action_dim) # Initialize the replay buffer

        # Initialize the actor and critic networks
        self.actor = Actor(state_dim, action_dim, minActions=torch.Tensor(minActions), maxActions=torch.Tensor(maxActions)).to(device)
        self.criticQ1 = CriticQ(state_dim, action_dim).to(device)
        self.criticQ2 = CriticQ(state_dim, action_dim).to(device)
        self.criticQ1Target = CriticQ(state_dim, action_dim).to(device)
        self.criticQ2Target = CriticQ(state_dim, action_dim).to(device)

        # If a load file is provided, load the models
        if load_file:
            self.actor.load_state_dict(torch.load(f"{load_file}_actor.pth"))
            self.criticQ1.load_state_dict(torch.load(f"{load_file}_criticQ1.pth"))
            self.criticQ2.load_state_dict(torch.load(f"{load_file}_criticQ2.pth"))
            self.criticQ1Target.load_state_dict(torch.load(f"{load_file}_criticQ1Target.pth"))
            self.criticQ2Target.load_state_dict(torch.load(f"{load_file}_criticQ2Target.pth"))
            self.alpha = torch.load(f"{load_file}_alpha.pt")

        # Initialize the optimizers                               
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.criticQ1_optimizer = optim.Adam(self.criticQ1.parameters(), lr=lr)
        self.criticQ2_optimizer = optim.Adam(self.criticQ2.parameters(), lr=lr)

        self.criticQ1Target.load_state_dict(self.criticQ1.state_dict())
        self.criticQ2Target.load_state_dict(self.criticQ2.state_dict())

        self.alpha_optimizer = optim.Adam([self.alpha], lr=lr)
        
        # Initialize the loss functions
        self.criticQ1Loss = nn.MSELoss()
        self.criticQ2Loss = nn.MSELoss()

        self.entropyTarget = -action_dim # Entropy target is the negative of the action dimension

    def train(self, env, episodeNo, episodeLength=100, allPaths=None, attackPaths=None):
        #losses = {'criticQ1':[], 'criticQ2':[], 'actor':[], 'alpha':[]}
        rewards = [] # List to store the rewards
        success = 0 # Number of successful episodes
        pathsGiven = False

        if allPaths is not None:
            pathsGiven = True
            episodeNo = len(allPaths)

        for episode in range(episodeNo): # Loop over the episodes
            if pathsGiven:
                state,_ = env.reset(initial_state = attackPaths[episode][0])
                episodeLength = len(allPaths[episode])
            else:
                state,_ = env.reset(randomize=True) # Reset the environment
            done = False # Initialize done as False
            episodeData = [] # List to store the episode data
            episodeReward = 0 # Initialize the episode reward
            totalCriticQ1Loss = 0 # Initialize the total criticQ1 loss
            totalCriticQ2Loss = 0 # Initialize the total criticQ2 loss
            totalActorLoss = 0 # Initialize the total actor loss
            totalAlphaLoss = 0 # Initialize the total alpha loss
            for stepNo in range(episodeLength): # Loop over one episode
                state = torch.Tensor(state).to(self.device) # Convert the state to tensor and move it to the device
                action, _ = self.actor.sampleAction(state) # Sample an action from the actor network
                action = action.cpu().detach().numpy() # Convert the action to numpy
                if pathsGiven:
                    nextState, reward, done, info = env.step(action, allPaths[episode][min(episodeLength-1,1+stepNo)])
                else:
                    nextState, reward, done, info = env.step(action) # Take a step in the environment
                episodeData.append((state, action, reward, nextState, done)) # Append the episode data

                # render at each 25 episode
                if episode % 25 == 0:
                    env.render()

                state = nextState # Update the state

                episodeReward += reward # Add the reward to the the episode reward

                if done:
                    if info['is_success']:
                        success += 1
                    break

            self.replayBuffer.recordEpisode(episodeData) # Record the episode data in the replay buffer

            if self.replayBuffer.full or self.replayBuffer.lastIndex > self.batch_size: # If the replay buffer is full or the last index is greater than the batch size, start training
                for k in range(len(episodeData)):
                    # Sample a batch from the replay buffer and convert the data to tensors and move them to the device
                    batch = self.replayBuffer.sample(self.batch_size)
                    batch['state'] = torch.Tensor(batch['state']).to(self.device)
                    batch['action'] = torch.Tensor(batch['action']).to(self.device)
                    batch['reward'] = torch.Tensor(batch['reward']).to(self.device)
                    batch['nextState'] = torch.Tensor(batch['nextState']).to(self.device)
                    batch['terminated'] = torch.Tensor(batch['terminated']).to(self.device)

                    criticQ1_loss, criticQ2_loss, actor_loss, alpha_loss = self.gradientStep(batch) # Calculate losses and perform a gradient step over every network for the batch
                    
                    # Add the losses to the total losses
                    totalCriticQ1Loss += criticQ1_loss
                    totalCriticQ2Loss += criticQ2_loss
                    totalActorLoss += actor_loss
                    totalAlphaLoss += alpha_loss

            rewards.append(episodeReward)
            # losses['criticQ1'].append(totalCriticQ1Loss/stepNo)
            # losses['criticQ2'].append(totalCriticQ2Loss/stepNo)
            # losses['actor'].append(totalActorLoss/stepNo)
            # losses['alpha'].append(totalAlphaLoss/stepNo)

            #if episode % int(episodeNo/10) == int(episodeNo/10) - 1: # Print the results every half of the episodes
            #if episode % 5 == 4:
                #print(f"Episode: {1+episode}/{episodeLength}, Reward: {np.array(rewards).mean().item():.2f}, CriticQ1 Loss: {np.array(losses['criticQ1']).mean().item():.2f}, CriticQ2 Loss: {np.array(losses['criticQ2']).mean().item():.2f}, Actor Loss: {np.array(losses['actor']).mean().item():.2f}, Alpha Loss: {np.array(losses['alpha']).mean().item():.2f}, alpha = {self.alpha.item():.2f}")
            print(f"Episode: {1+episode}/{episodeNo}, Reward: {np.array(rewards).mean().item():.2f}")
        
        successRate = success/episodeNo # Calculate the success rate

        return np.array(rewards).mean().item(), successRate
        #return np.array(rewards).mean().item(), np.array(losses['criticQ1']).mean().item(), np.array(losses['criticQ2']).mean().item(), np.array(losses['actor']).mean().item(), np.array(losses['alpha']).mean().item(), successRate

    def actorLoss(self, state, action, logProbs): # Calculate the actor loss
        q1 = self.criticQ1Target(state, action)
        q2 = self.criticQ2Target(state, action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha.item() * logProbs - q).mean()
        return actor_loss
    
    def criticLosses(self, state, action, reward, nextState, done): # Calculate the critic losses

        q1 = self.criticQ1(state, action)
        q2 = self.criticQ2(state, action)

        nextAction, nextLogProbs = self.actor.sampleAction(nextState)
        targetq1 = self.criticQ1Target(nextState, nextAction)
        targetq2 = self.criticQ2Target(nextState, nextAction)

        targetq = torch.min(targetq1, targetq2)

        target = reward.view(-1,1) + self.gamma * (targetq - self.alpha.item() * nextLogProbs)*(1-done.view(-1,1))
        target = target.detach()

        criticQ1_loss = self.criticQ1Loss(q1, target)
        criticQ2_loss = self.criticQ2Loss(q2, target)

        return criticQ1_loss, criticQ2_loss
    
    def alphaLoss(self, logProbs): # Calculate the alpha loss
        alpha_loss = (-self.alpha * logProbs - self.alpha * self.entropyTarget).mean()
        return alpha_loss

    def updateTargetNetworks(self): # Update the target networks using exponential moving average
        for target_param, param in zip(self.criticQ1Target.parameters(), self.criticQ1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.criticQ2Target.parameters(), self.criticQ2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def gradientStep(self, batch):
        # Training code

        # Extract the data from the batch
        state = batch['state']
        action = batch['action']
        reward = batch['reward']
        nextState = batch['nextState']
        done = batch['terminated']

        # Training code for the critic
        criticQ1_loss, criticQ2_loss = self.criticLosses(state, action, reward, nextState, done)
        
        self.criticQ1_optimizer.zero_grad()
        criticQ1_loss.backward()
        self.criticQ1_optimizer.step()

        self.criticQ2_optimizer.zero_grad()
        criticQ2_loss.backward()
        self.criticQ2_optimizer.step()

        # Training code for the actor
        action1, logProbs = self.actor.sampleAction(state)

        actor_loss = self.actorLoss(state, action1, logProbs)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Training code for the alpha
        alpha_loss = self.alphaLoss(logProbs.detach())

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.updateTargetNetworks() # Update the target networks using exponential moving average

        return criticQ1_loss.item(), criticQ2_loss.item(), actor_loss.item(), alpha_loss.item()

    def test(self, env, episodeNo=100, episodeLength=100, seed=100, render=False):
        # Evaluation code
        rewards = []
        success = 0
        for episode in range(episodeNo): # Loop over the episodes
            state,_ = env.reset(seed = seed, randomize=True)
            totalReward = 0
            for stepNo in range(episodeLength): # Loop over one episode
                state = torch.Tensor(state).to(self.device) # Convert the state to tensor and move it to the device
                action, _ = self.actor.sampleAction(state) # Sample an action from the actor network
                action = action.cpu().detach().numpy() # Convert the action to numpy
                nextState, reward, done, info = env.step(action) # Take a step in the environment
                state = nextState # Update the state
                if render:
                    env.render()
                if done:
                    if info['is_success']:
                        print('robot hit')
                        success += 1
                    break
            
                totalReward += reward # Add the reward to the total reward
           
            rewards.append(totalReward)

        successRate = success/episodeNo # Calculate the success rate
        return np.array(rewards).mean(), successRate