# Copyright 2020 (c) Aalto University - All Rights Reserved
# Author: Julius Hietala <julius.hietala@aalto.fi>
#
import numpy as np
import gym
from rlkit.samplers.rollout_functions import multitask_rollout
from generate.sideways_trajectory import items2
from pynput import mouse
import copy
from joint_angles import joint_angles
from sideways_positions import sideways_positions


if __name__ == "__main__":
    env_name = 'Franka-v1'
    env = gym.make(env_name)
    env.reset()
    jc = False

    if jc:
        for joint_control in joint_angles:
            action = joint_control + [1]
            action = np.array(action)
            stuff = env.step(action)
            env.render()
    else:

        
        for prestep in range(50):
            action = np.array([0.0, 0.0, 0.0, 1.0])
            stuff = env.step(action)
            env.render()

        '''
        for midsteps in range(1000):
            action = np.array([0, 0, 0.0002, 1.0])
            stuff = env.step(action)
            env.render()
        
        for s in range(10000):
            action = np.array([0.0, 0.0, 0.0, 1.0])
            if s < 60:
                action[0] = -0.0001
                action[1] = 0.0001
                action[2] = 0.0001
            elif s < 120:
                action[0] = -0.0001
                action[1] = 0.0001
                action[2] = -0.0001
            elif s < 150:
                action[2] = 0.0002
                action[3] = -1.0
        '''
        for step in sideways_positions:
            action = step + [1]
            action = np.array(action)/2 * 0.0016
            #action = np.array(action)/2 * 0.1
            stuff = env.step(action)
            env.render()

            stuff = env.step(action)
            env.render()
        
        for final_step in range(1000):
            action = np.array([0.0, 0.0, 0.000, 1.0])
            if final_step > 10 and final_step < 70:
                action[2] = 0.0002
                action[3] = -1
            stuff = env.step(action)
            env.render()