# Copyright 2020 (c) Aalto University - All Rights Reserved
# Author: Julius Hietala <julius.hietala@aalto.fi>
#
import numpy as np
import gym
from rlkit.samplers.rollout_functions import multitask_rollout
from generate.sideways_trajectory import items2
from pynput import mouse
import copy


if __name__ == "__main__":
    env_name = 'Franka-v1'
    env = gym.make(env_name)
    env.reset()

    for s in range(10000):
        print(s)
        action = np.zeros(9)
        stuff = env.step(action)
        env.render()