# Copyright 2020 (c) Aalto University - All Rights Reserved
# Author: Julius Hietala <julius.hietala@aalto.fi>
#
import numpy as np
import gym


if __name__ == "__main__":
    env_name = 'Franka-v1'
    env = gym.make(env_name)
    while True:
        env.reset()
        for s in range(300):
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
            stuff = env.step(action)
            env.render()
