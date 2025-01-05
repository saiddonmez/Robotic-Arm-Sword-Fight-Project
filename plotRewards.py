import matplotlib.pyplot as plt
import csv
import numpy as np

def plotRewards(file):
    f = open(file, 'r')
    reader = csv.reader(f)
    header = next(reader)
    rewards = np.array(list(reader)).astype(float)[:,1]
    # moving average of rewards
    window = 100
    rewards = [sum(rewards[i:i+window])/window for i in range(len(rewards)-window)]

    plt.plot(np.array(rewards))
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Episodes')
    plt.savefig('rewards_plot.png')


plotRewards("rewards_log_2M_trial1.csv")