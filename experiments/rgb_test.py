"""
Script to showcase domain randomization functionality.
"""

import robosuite.utils.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import DomainRandomizationWrapper

from gym.envs.robotics import task_definitions
from utils import get_variant, argsparser
import cv2
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhCNNGaussianPolicy, GaussianPolicy, GaussianCNNPolicy
import torch
import numpy as np
import time


DEFAULT_CAMERA_ARGS = {
    'camera_names': None,  # all cameras are randomized
    'randomize_position': True,
    'randomize_rotation': True,
    'randomize_fovy': True,
    'position_perturbation_size': 0.01,
    'rotation_perturbation_size': 0.087,
    'fovy_perturbation_size': 5.,
}


def get_robosuite_env(variant):
    options = {}
    options["env_name"] = variant["env_name"]
    options["robots"] = "Panda"
    controller_name = variant["ctrl_name"]
    options["controller_configs"] = load_controller_config(
        default_controller=controller_name)
    options["controller_configs"]["interpolation"] = "linear"
    env = suite.make(
        **options,
        **variant['env_kwargs'],
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=False,
        use_camera_obs=False,
    )
    return env


def obs_processor(o):
    obs = o['image']
    for additional_key in ['robot_observation']:
        obs = np.hstack((obs, o[additional_key]))

    return np.hstack((obs, o['desired_goal']))

# We'll use instance randomization so that entire geom groups are randomized together


if __name__ == "__main__":

    args = argsparser()
    variant = get_variant(args)

    env = get_robosuite_env(variant)

    added_fc_input_size = env.observation_space.spaces['desired_goal'].low.size

    action_dim = env.action_space.low.size
    robot_obs_dim = env.observation_space.spaces['robot_observation'].low.size
    added_fc_input_size += robot_obs_dim

    agent = TanhCNNGaussianPolicy(
        output_size=3,
        added_fc_input_size=added_fc_input_size,
        aux_output_size=12,
        **variant['policy_kwargs'],
    )

    agent.load_state_dict(torch.load(
        '/Users/juliushietala/Desktop/current_policy.mdl', map_location=torch.device('cpu')))

    #agent = MakeDeterministic(agent)

    camera_id = env.sim.model.camera_name2id('sideview')

    # do visualization
    traj = 0
    # while True:
    traj += 1
    traj_deltas = []
    o = env.reset()
    for i in range(50):
        o_for_agent = obs_processor(o)
        a, agent_info, aux_output = agent.get_action(o_for_agent)
        traj_deltas.append(a)
        #print("Action", i+1, ": ", a)
        o, reward, done, _ = env.step(a)
        env.sim._render_context_offscreen.render(
            1000, 1000, camera_id)
        image_obs = env.sim._render_context_offscreen.read_pixels(
            1000, 1000, depth=False)

        image_obs = image_obs[::-1, :, :]

        image = image_obs.reshape((1000, 1000, 3)).copy()
        cv2.imshow('goal', image)
        cv2.waitKey(10)
        time.sleep(2)
    print("actual traj")
    print(env.robots[0].controller.ee_poss)
    print(np.array(env.robots[0].controller.ee_poss).shape)
    #traj_deltas = np.array(traj_deltas)
    #np.save("trajs/traj" + str(traj) + ".npy", traj_deltas)
