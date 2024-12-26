from raiSimulationEnv import *
import robotic as ry
import time
import numpy as np
import argparse
import gymnasium as gym # version 1.0.0
import torch # version 2.5.1+cu124
from pytorch_SAC.sac import SAC
from pytorch_SAC.replay_memory import ReplayMemory

import random
import numpy as np # version 1.26.3
import matplotlib.pyplot as plt
import os 


def load_data_spline(folder):
    datas = []
    for path in os.listdir(folder):
        if path.endswith('.npy'):
            path = np.load(os.path.join(folder, path)).reshape(-1,28)
            datas.append(path)
    return datas


def load_data(folder):
    datas = []
    for path in os.listdir(folder):
        if path.endswith('.npy') and "hitting" not in path:
            path = np.load(os.path.join(folder, path))
            datas.append(path)
    return datas

def trainModel(args, agent, env, memory, episodeNo, episodeLength=100, allPaths=None, attackPaths=None):
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

        for stepNo in range(episodeLength): # Loop over one episode
            #state = torch.Tensor(state).to(agent.device) # Convert the state to tensor and move it to the device
            if args.start_steps > len(memory):
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy
            #action = action.cpu().detach().numpy() # Convert the action to numpy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

            if pathsGiven:
                nextState, reward, done, info = env.step(action, allPaths[episode][stepNo])
            else:
                nextState, reward, done, info = env.step(action) # Take a step in the environment
            # episodeData.append((state, action, reward, nextState, done)) # Append the episode data

            # # render at each 25 episode
            if episode % 25 == 0:
                env.render()

            # state = nextState # Update the state

            episodeReward += reward # Add the reward to the the episode reward




            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if stepNo == episodeLength else float(not done)

            memory.push(state, action, reward, nextState, mask) # Append transition to memory

            state = nextState

            if done:
                if info['is_success']:
                    success += 1
                break


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

def testModel(agent, env, episodeNo=100, episodeLength=100, seed=100, render=False):
    # Evaluation code
    rewards = []
    success = 0
    for episode in range(episodeNo): # Loop over the episodes
        state,_ = env.reset(seed = seed, randomize=True)
        totalReward = 0
        for stepNo in range(episodeLength): # Loop over one episode
            state = torch.Tensor(state).to(agent.device) # Convert the state to tensor and move it to the device
            action = agent.select_action(state, evaluate=True)
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


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=100, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    
    parser.add_argument("--modelSavePath", type=str, default="SAC/modelsSaved/model_saved", help="Path to save the model")
    parser.add_argument("--load_file", type=str, default=None, help="Path to load the model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelSavePath = args.modelSavePath
    load_file = args.load_file
    train = True
    print(train)
    allPaths = load_data_spline('real_spline_attackPaths')
    attackPaths = load_data('attackPaths')

    env = RobotSimEnv()

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    print(nb_states)
    print(nb_actions)

    env.action_space
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    updates = 0


    meanTrainRewardsList = []
    meanTestRewardsList = []
    trainSuccessRateList = []
    testSuccessRateList = []

    if train:
        for epoch in range(100):
            for t in range(len(allPaths)//100):
                # Train model for 100 epochs each with 50 episodes
                meanTrainRewards, trainSuccessRate = trainModel(args, agent, env, memory, episodeNo=50, episodeLength=50, allPaths = allPaths[0:1],attackPaths=attackPaths[0:1]) #allPaths[t*100:(t+1)*100]
                env.close()
                # Test the model after each epoch of training
                render = True # "human" or None
                seed = random.randint(0, 1000) # Random seed for testing
                # Reinitialize the environment for testing
                env = RobotSimEnv(render=render)
                if epoch >90:
                    meanTestRewards, testSuccessRate = testModel(agent, env, episodeNo=1, episodeLength=100, seed=seed, render=render)
                    env.close()
                meanTestRewards = 0
                testSuccessRate = 0
                print(f"epoch {epoch}, meanTrainRewards: {meanTrainRewards}, meanTestRewards: {meanTestRewards}, trainSuccessRate: {trainSuccessRate}, testSuccessRate: {testSuccessRate}")

                meanTrainRewardsList.append(meanTrainRewards)
                meanTestRewardsList.append(meanTestRewards)
                trainSuccessRateList.append(trainSuccessRate)
                testSuccessRateList.append(testSuccessRate)

                env = RobotSimEnv()
                os.system('clear')

    else: # Test
        assert load_file is not False, "Please provide the path to the models to test"
        render = True
        seedList = [4, 8, 15, 16, 23, 42] # 6 different random seeds
        env = RobotSimEnv(render=render)
        for seed in seedList:
            meanTestRewards, testSuccessRate = agent.test(env, episodeNo=1, episodeLength=50, seed=seed, render=render)
        env.close()
        print(f"meanTestRewards: {meanTestRewards}, testSuccessRate: {testSuccessRate}")
    
    # Save the results
    np.save("meanTrainRewardsList.npy", meanTrainRewardsList)
    np.save("meanTestRewardsList.npy", meanTestRewardsList)
    np.save("trainSuccessRateList.npy", trainSuccessRateList)
    np.save("testSuccessRateList.npy", testSuccessRateList)

    # Plot the results and save them, do not show the plots
    meanTrainRewardsList = np.load("meanTrainRewardsList.npy")
    meanTestRewardsList = np.load("meanTestRewardsList.npy")
    trainSuccessRateList = np.load("trainSuccessRateList.npy")
    testSuccessRateList = np.load("testSuccessRateList.npy")
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(meanTrainRewardsList, label="Train")
    ax[0].plot(meanTestRewardsList, label="Test")
    ax[0].set_title("Mean rewards over epochs")
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Mean rewards")

    ax[1].plot(trainSuccessRateList, label="Train")
    ax[1].plot(testSuccessRateList, label="Test")
    ax[1].set_title("Success rate over epochs")
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Success rate")
    plt.savefig("results.png")
    plt.close()
    env.close()
    print("Done!")