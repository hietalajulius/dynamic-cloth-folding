from rlkit.envs.wrappers import NormalizedBoxEnv
import gym
from utils import get_variant, argsparser
import numpy as np
import time

args = argsparser()
variant = get_variant(args)

env = NormalizedBoxEnv(
    gym.make(variant['env_name'], **variant['env_kwargs']))

env2 = NormalizedBoxEnv(
    gym.make(variant['env_name'], **variant['env_kwargs']))

env.reset()
times = []
for _ in range(500):
    step_start = time.time()
    action = np.random.uniform(-1, 1, (3,))
    next_o, r, d, env_info = env.step(action)
    step_total_time = time.time() - step_start
    print("Took to step with action", action, step_total_time)
    times.append(step_total_time)
    # if d:
    # env.reset()

print("Average step time", np.mean(np.array(times)))
