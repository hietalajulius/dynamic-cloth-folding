from rlkit.envs.wrappers import NormalizedBoxEnv
import gym
from utils import get_variant, argsparser
import numpy as np
import time
import copy


args = argsparser()
variant = get_variant(args)

env1 = gym.make('Franka-v1', **variant['env_kwargs'])

start = time.time()
epis = 50
while True:
    env1.reset()
    for i in range(epis):
        print(i)
        # action = np.random.uniform(-10, 10, (7,))
        action = -np.zeros(7)
        #action = -np.ones(7)
        #action[3] = 0.075

        ret = env1.step(action)
        print("ret", ret)
        env1.render()
