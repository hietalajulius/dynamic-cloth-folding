"""
Script to showcase domain randomization functionality.
"""

import robosuite.utils.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import DomainRandomizationWrapper

from gym.envs.robotics import task_definitions
import cv2
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhCNNGaussianPolicy, GaussianPolicy, GaussianCNNPolicy
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from ctrl_optimization import eval_settings
from utils import get_variant, argsparser, get_robosuite_env, ATE, get_tracking_score, plot_trajectory, render_env, get_obs_processor


if __name__ == "__main__":
    save_new = True
    save_idx = 4
    args = argsparser()
    variant = get_variant(args)

    env = get_robosuite_env(variant)

    added_fc_input_size = env.observation_space.spaces['desired_goal'].low.size

    action_dim = env.action_space.low.size
    obs_dim = env.observation_space.spaces['observation'].low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    robot_obs_dim = env.observation_space.spaces['robot_observation'].low.size
    added_fc_input_size += robot_obs_dim

    policy_obs_dim = obs_dim + goal_dim + robot_obs_dim

    if variant['image_training']:
        policy_name = 'current_policy_3cm.mdl'
        agent = TanhCNNGaussianPolicy(
            output_size=3,
            added_fc_input_size=added_fc_input_size,
            aux_output_size=12,
            **variant['policy_kwargs'],
        )
    else:
        M = variant['layer_size']
        policy_name = '../policies/sync/ctrl-opt-mod_current_policy.mdl'
        agent = TanhGaussianPolicy(
            obs_dim=policy_obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M],
            **variant['policy_kwargs']
        )

    agent.load_state_dict(torch.load(
        policy_name, map_location=torch.device('cpu')))

    agent = MakeDeterministic(agent)

    obs_processor = get_obs_processor(
        'observation', ['robot_observation'], 'desired_goal')

    tracking_score, ate, s, current_ee_positions = eval_settings(
        variant, agent, render=True, plot=True, max_steps=15, obs_processor=obs_processor, plot_predefined=False)

    if save_new:
        np.save(
            f"traj_opt/ee_reached_{str(save_idx)}.npy", current_ee_positions)

    print(f"Tracking: {tracking_score}, ATE: {ate}")