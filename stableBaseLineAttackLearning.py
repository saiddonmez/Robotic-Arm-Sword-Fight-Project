from raiSimulationEnv import *
import robotic as ry
import time
import numpy as np
import torch

from ddpgCode.normalized_env import NormalizedEnv
from ddpgCode.evaluator import Evaluator
from ddpgCode.ddpg import DDPG
from ddpgCode.util import *
from raiSimulationEnv2 import *
import argparse

import gymnasium as gym



if __name__ == "__main__":



    env = 