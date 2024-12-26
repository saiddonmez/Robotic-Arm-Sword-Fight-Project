import gymnasium_robotics # version 1.3.1
import gymnasium as gym # version 1.0.0
import torch # version 2.5.1+cu124
from models import SAC
import random
import numpy as np # version 1.26.3
import matplotlib.pyplot as plt
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--modelSavePath", type=str, default="models/model_saved", help="Path to save the model")
parser.add_argument("--load_file", type=str, default=None, help="Path to load the model")
parser.add_argument("--train", action="store_true", help="Train the model")
args = parser.parse_args()  

if __name__ == "__main__":
    # Set the device and the path to save the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # call argument parser
    modelSavePath = args.modelSavePath
    load_file = args.load_file
    train = args.train

    print(train)
    print(load_file)
    print(modelSavePath)


    gym.register_envs(gymnasium_robotics)
    env = gym.make('FetchReach-v3',max_episode_steps=50)
    state_dim = env.observation_space['observation'].shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.observation_space['desired_goal'].shape[0]

    # Initialize the SAC Model
    sac = SAC(state_dim=state_dim, action_dim=action_dim, goal_dim=goal_dim, device=device, load_file=load_file)

    meanTrainRewardsList = []
    meanTestRewardsList = []
    trainSuccessRateList = []
    testSuccessRateList = []
    if train:
        for epoch in range(100):
            # Train model for 100 epochs each with 50 episodes
            meanTrainRewards, meanLossq1, meanLossq2, meanLossActor, meanLossAlpha, trainSuccessRate = sac.train(env, episodeNo=50, episodeLength=50)
            env.close()
            # Test the model after each epoch of training
            render_mode = None # "human" or None
            seed = random.randint(0, 1000) # Random seed for testing
            # Reinitialize the environment for testing
            env = gym.make('FetchReach-v3', render_mode=render_mode)
            meanTestRewards, testSuccessRate = sac.test(env, episodeNo=20, episodeLength=50, seed=seed, render_mode=render_mode)
            env.close()

            print(f"epoch {epoch}, meanTrainRewards: {meanTrainRewards}, meanTestRewards: {meanTestRewards}, trainSuccessRate: {trainSuccessRate}, testSuccessRate: {testSuccessRate}")

            meanTrainRewardsList.append(meanTrainRewards)
            meanTestRewardsList.append(meanTestRewards)
            trainSuccessRateList.append(trainSuccessRate)
            testSuccessRateList.append(testSuccessRate)

            # Save the models and the alpha
            sac.actor.save_model(f"{modelSavePath}_actor.pth")
            sac.criticQ1.save_model(f"{modelSavePath}_criticQ1.pth")
            sac.criticQ2.save_model(f"{modelSavePath}_criticQ2.pth")
            sac.criticQ1Target.save_model(f"{modelSavePath}_criticQ1Target.pth")
            sac.criticQ2Target.save_model(f"{modelSavePath}_criticQ2Target.pth")
            torch.save(sac.alpha, f"{modelSavePath}_alpha.pt")

            env = gym.make('FetchReach-v3',max_episode_steps=50)

    else: # Test
        assert load_file is not None, "Please provide the path to the models to test"
        render_mode = "human"
        seedList = [4, 8, 15, 16, 23, 42] # 6 different random seeds
        env = gym.make('FetchReach-v3', render_mode=render_mode)
        for seed in seedList:
            meanTestRewards, testSuccessRate = sac.test(env, episodeNo=1, episodeLength=50, seed=seed, render_mode=render_mode)
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