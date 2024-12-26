from raiSimulationEnv import *
import robotic as ry
import time
import numpy as np
import argparse
import gymnasium as gym # version 1.0.0
import torch # version 2.5.1+cu124
from SAC.models import SAC
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



if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
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
    agent = SAC(state_dim=nb_states, action_dim=nb_actions, device=device, load_file=load_file,minActions=env.action_space.low, maxActions=env.action_space.high)

    meanTrainRewardsList = []
    meanTestRewardsList = []
    trainSuccessRateList = []
    testSuccessRateList = []

    if train:
        for epoch in range(100):
            for t in range(len(allPaths)//100):
                # Train model for 100 epochs each with 50 episodes
                meanTrainRewards, trainSuccessRate = agent.train(env, episodeNo=50, episodeLength=50, allPaths = allPaths[0:1],attackPaths=attackPaths[0:1]) #allPaths[t*100:(t+1)*100]
                #env.close()
                # Test the model after each epoch of training
                render = True # "human" or None
                seed = random.randint(0, 1000) # Random seed for testing
                # Reinitialize the environment for testing
                env = RobotSimEnv(render=render)
                if epoch >90:
                    meanTestRewards, testSuccessRate = agent.test(env, episodeNo=1, episodeLength=100, seed=seed, render=render)
                    env.close()
                meanTestRewards = 0
                testSuccessRate = 0
                print(f"epoch {epoch}, meanTrainRewards: {meanTrainRewards}, meanTestRewards: {meanTestRewards}, trainSuccessRate: {trainSuccessRate}, testSuccessRate: {testSuccessRate}")

                meanTrainRewardsList.append(meanTrainRewards)
                meanTestRewardsList.append(meanTestRewards)
                trainSuccessRateList.append(trainSuccessRate)
                testSuccessRateList.append(testSuccessRate)

                # Save the models and the alpha
                agent.actor.save_model(f"{modelSavePath}_actor.pth")
                agent.criticQ1.save_model(f"{modelSavePath}_criticQ1.pth")
                agent.criticQ2.save_model(f"{modelSavePath}_criticQ2.pth")
                agent.criticQ1Target.save_model(f"{modelSavePath}_criticQ1Target.pth")
                agent.criticQ2Target.save_model(f"{modelSavePath}_criticQ2Target.pth")
                torch.save(agent.alpha, f"{modelSavePath}_alpha.pt")

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