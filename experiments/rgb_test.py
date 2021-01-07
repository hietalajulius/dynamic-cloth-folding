import numpy as np
from rlkit.envs.wrappers import NormalizedBoxEnv
from utils import argsparser, get_variant
import gym

def experiment(variant):
    eval_env = NormalizedBoxEnv(
        gym.make(variant['env_name'], **variant['env_kwargs']))


    
    eval_env.reset()
    
    while True:
        eval_env.render()
        obs, r, d, info = eval_env.step(np.array([0,0,0]))
        print("Step")




if __name__ == "__main__":
    args = argsparser()
    variant = get_variant(args)
    experiment(variant)