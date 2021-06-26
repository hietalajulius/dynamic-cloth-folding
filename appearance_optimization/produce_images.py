import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import KeyPathCollector, VectorizedKeyPathCollector
from rlkit.torch.sac import policies
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhScriptPolicy, CustomScriptPolicy, CustomTanhScriptPolicy, ScriptPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.her.cloth.her import ClothSacHERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.data_management.future_obs_dict_replay_buffer import FutureObsDictRelabelingBuffer
import gym
import mujoco_py
import torch
import cProfile
from rlkit.envs.wrappers import SubprocVecEnv
from gym.logger import set_level
from clothmanip.utils.utils import get_variant, argsparser, get_randomized_env, dump_commit_hashes, get_keys_and_dims, dump_goal
import copy
from clothmanip.utils import reward_calculation
import numpy as np
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
import os
from rlkit.core import logger
import json
from rlkit.samplers.eval_suite.success_rate_test import SuccessRateTest
from rlkit.samplers.eval_suite.blank_images_test import BlankImagesTest
from rlkit.samplers.eval_suite.real_corner_prediction_test import RealCornerPredictionTest

from rlkit.samplers.eval_suite.base import EvalTestSuite

import tracemalloc
import cv2
import albumentations as A

def main(variant):
    variant['output_max'] = 1

    def make_env():
        env = ClothEnv(**variant['env_kwargs'], save_folder=variant['save_folder'], has_viewer=variant['image_training'])
        env = NormalizedBoxEnv(env)
        env = get_randomized_env(env, variant)
        return env

    env_fns = [make_env for _ in range(variant['num_processes'])]
    vec_env = SubprocVecEnv(env_fns)
    zero_action = np.zeros((variant['num_processes'], 3))
    rollouts = 0
    while True:
        demos = np.genfromtxt(np.random.choice(variant['demo_paths']), delimiter=',')
        vec_env.reset()
        
        for i, action in enumerate(demos):
            #print("\n")
            zero_action[:] = action
            #print(zero_action)

            o, _, _, info = vec_env.step(zero_action)
            #print("Img", o['image'], i)
            images = o['image']

            for j, image in enumerate(images):
                c = info[j]['corner_positions']
                #print("image",image.shape)
                reshaped_image = image.reshape((100,100,1)).astype('float32')
               # print("reshapes image", reshaped_image.shape, reshaped_image)
                corner_pos_str = f"{c[0]}_{c[1]}_{c[2]}_{c[3]}_{c[4]}_{c[5]}_{c[6]}_{c[7]}"

                cv2.imwrite(f"{variant['folder']}/{rollouts}_{i}_{j}_{corner_pos_str}.png", reshaped_image*255)

            rollouts += 1
            
            ###DEBUG
            '''
            corn = i[-1]['corner_positions']
            for corn_idx in range(int(corn.shape[0]/2)):
                aux_u = int(corn[corn_idx*2]*100)
                aux_v = int(corn[corn_idx*2+1]*100)
                cv2.circle(gray_image, (aux_u, aux_v), 2, (0, 255, 0), -1)
            cv2.imshow("env", gray_image)
            cv2.waitKey(100)
            '''
        




if __name__ == "__main__":
    args = argsparser()
    variant, arg_str = get_variant(args)

    variant["save_folder"] = variant["folder"]


    main(variant)