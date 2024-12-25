from raiSimulationEnv import *
import robotic as ry
import time
import numpy as np
import torch

from ddpgCode.normalized_env import NormalizedEnv
from ddpgCode.evaluator import Evaluator
from ddpgCode.ddpg import DDPG
from ddpgCode.util import *
import argparse

import gymnasium as gym

def load_data(folder):
    datas = []
    for path in os.listdir(folder):
        if path.endswith('.npy'):
            path = np.load(os.path.join(folder, path)).reshape(-1,28)
            datas.append(path)
    return datas


def train(num_iterations, agent, env, evaluate, validate_steps, output, max_episode_length=None, debug=False):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='test', type=str, help='support option: train/test')
    parser.add_argument('--env', default='Pendulum-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=128, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=1000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.02, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=2, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=100, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=10000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()



    allPaths = load_data('real_spline_attackPaths')

    env = RobotSimEnv()

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]

    print(nb_states)
    print(nb_actions)

    agent = DDPG(nb_states, nb_actions, args)
    evaluate = Evaluator(args.validate_episodes, args.validate_steps, args.output, max_episode_length=args.max_episode_length)

    agent.actor.load_state_dict(torch.load('output/actor.pkl'))
    agent.critic.load_state_dict(torch.load('output/critic.pkl'))

    for path in allPaths:
        meanTestReward = evaluate(env,agent,visualize=True,save=False, initial_state=path[0])
        print(f"Mean test reward: {meanTestReward:.2f}")

