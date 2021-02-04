from utils import get_variant, argsparser
import gym
from panda_gym import gym_envs
import cv2
import argparse
import numpy as np

args = argsparser()
variant = get_variant(args)
env = gym.make(variant['env_name'], **variant['env_kwargs'])

while True:
    env.reset()
    for i in range(1000):
        print(i)
        ac = np.zeros(7)
        ac[3] = 0.001
        #ac = np.ones(7)*0.001
        env.step(ac)
        env.render()
