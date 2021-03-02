"""
Script to showcase domain randomization functionality.
"""

import robosuite.utils.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import DomainRandomizationWrapper

from gym.envs.robotics import task_definitions
from utils import get_variant, argsparser, get_robosuite_env
import cv2
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhCNNGaussianPolicy, GaussianPolicy, GaussianCNNPolicy
import torch
import numpy as np
import time
import matplotlib.pyplot as plt


DEFAULT_CAMERA_ARGS = {
    'camera_names': None,  # all cameras are randomized
    'randomize_position': True,
    'randomize_rotation': True,
    'randomize_fovy': True,
    'position_perturbation_size': 0.01,
    'rotation_perturbation_size': 0.087,
    'fovy_perturbation_size': 5.,
}


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
        'current_policy_3cm.mdl', map_location=torch.device('cpu')))

    # agent = MakeDeterministic(agent)

    camera_id_1 = env.sim.model.camera_name2id('sideview')
    camera_id_2 = env.sim.model.camera_name2id('frontview')

    cameras_to_render = ['frontview']

    error_distances = []
    current_ee_positions = []
    current_ee_goals = []

    substep_errors = []
    substep_positions = []
    substep_goals = []

    o = env.reset()
    for _ in range(15):
        del env.sim._render_context_offscreen._markers[:]

        o_for_agent = obs_processor(o)
        a, agent_info, aux_output = agent.get_action(o_for_agent)
        o, reward, done, info = env.step(a)

        error_distances.append(info['error_distance'])
        current_ee_positions.append(info['ee_pos'])
        current_ee_goals.append(info['ee_goal'])

        substep_errors += info['substep_errors']
        substep_positions += info['substep_positions']
        substep_goals += info['substep_goals']

        for idx, goal_marker in enumerate(current_ee_goals):
            env.sim._render_context_offscreen.add_marker(size=np.array(
                [.001, .001, .001]), pos=goal_marker, label=str(1+idx))

        for camera in cameras_to_render:
            camera_id = env.sim.model.camera_name2id(camera)
            env.sim._render_context_offscreen.render(
                1000, 1000, camera_id)
            image_obs = env.sim._render_context_offscreen.read_pixels(
                1000, 1000, depth=False)
            image_obs = image_obs[::-1, :, :]
            image = image_obs.reshape((1000, 1000, 3)).copy()
            cv2.imshow(camera, image)

        cv2.waitKey(10)
        time.sleep(1)

error_distances = np.array(error_distances)
current_ee_positions = np.array(current_ee_positions)
current_ee_goals = np.array(current_ee_goals)

substep_errors = np.array(substep_errors)
substep_positions = np.array(substep_positions)
substep_goals = np.array(substep_goals)

#data1 = current_ee_positions
#data2 = current_ee_goals

data1 = substep_positions
data2 = substep_goals

fig = plt.figure(figsize=(30, 30))
ax1 = fig.add_subplot(111, projection='3d')
ax1.view_init(10, -10)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax1.plot(data1[:, 0],
         data1[:, 1], data1[:, 2], linewidth=3)

ax1.plot(data2[:, 0],
         data2[:, 1], data2[:, 2], linewidth=3)

ax1.text(substep_positions[0, 0], substep_positions[0, 1], substep_positions[0, 2], '%s' %
         ("start"), size=20, zorder=1, color='k')

plt.show()
