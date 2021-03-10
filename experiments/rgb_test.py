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

from ctrl_optimization import trajectory_score


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


use_presets = True

if use_presets:
    print("using presets")
    actions = np.load("traj_opt/actions.npy")
    preset_positions = np.load("traj_opt/ee_reached.npy")
else:
    print("creating new actions")
    actions = []

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

    agent = MakeDeterministic(agent)

    camera_id_1 = env.sim.model.camera_name2id('sideview')
    camera_id_2 = env.sim.model.camera_name2id('frontview')

    cameras_to_render = ['frontview']

    trajectories = []

    if not use_presets:
        n_trajs = 1
    else:
        n_trajs = 1
    for _ in range(n_trajs):
        error_distances = []
        current_ee_positions = []
        current_ee_goals = []

        substep_starts = []
        substep_ends = []

        substep_positions = []
        substep_goals = []

        o = env.reset()

        for i in range(20):
            del env.sim._render_context_offscreen._markers[:]

            o_for_agent = obs_processor(o)
            a, agent_info, aux_output = agent.get_action(o_for_agent)

            if use_presets:
                a = actions[i]
            else:
                actions.append(a)

            o, reward, done, info = env.step(a)

            error_distances.append(info['error_distance'])
            current_ee_positions.append(info['ee_pos'])
            current_ee_goals.append(info['ee_goal'])

            substep_starts.append(info['substep_start'])
            substep_ends.append(info['substep_end'])

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
            # time.sleep(0.1)

        error_distances = np.array(error_distances)
        current_ee_positions = np.array(current_ee_positions)
        current_ee_goals = np.array(current_ee_goals)

        substep_starts = np.array(substep_starts)
        substep_ends = np.array(substep_ends)

        substep_positions = np.array(substep_positions)
        substep_goals = np.array(substep_goals)

        #print("Current ees at 0", current_ee_positions[0])
        #print("Current ee goals at 0", current_ee_goals[0])
        trajectories.append(substep_positions)

    #data1 = current_ee_positions
    #data2 = current_ee_goals

    if not use_presets:
        np.save("traj_opt/actions.npy", actions)
        np.save("traj_opt/ee_reached", current_ee_positions)
        print("saved actions and ee positions")

    fig = plt.figure(figsize=(30, 30))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.view_init(10, -10)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    #ax1.plot(data1[:, 0], data1[:, 1], data1[:, 2], linewidth=2)
    #ax1.plot(data2[:, 0], data2[:, 1], data2[:, 2], linewidth=2)
    #ax1.scatter(data5[:, 0], data5[:, 1], data5[:, 2], linewidth=1, marker='o')
    #ax1.scatter(data6[:, 0], data6[:, 1], data6[:, 2], linewidth=1, marker='x')

    #ax1.scatter(data2[:, 0], data2[:, 1], data2[:, 2],linewidth=1, c=col, marker='o')
    #ax1.scatter(data1[:, 0], data1[:, 1], data1[:, 2],linewidth=1, c=col, marker='x')

    # print(preset_actions_sum)
    #ax1.plot(preset_actions_sum[:, 0], preset_actions_sum[:, 1], preset_actions_sum[:, 2], linewidth=2)

    #ax1.plot(substep_positions[:, 0],substep_positions[:, 1], substep_positions[:, 2], linewidth=2)
    print("shapes", preset_positions.shape, current_ee_positions.shape)
    print("TRAJ SCORE", trajectory_score(
        preset_positions, current_ee_positions))

    if use_presets:
        ax1.plot(preset_positions[:, 0], preset_positions[:,
                                                          1], preset_positions[:, 2], linewidth=5)

        '''
        ax1.scatter(substep_goals[:, 0], substep_goals[:,
                                                    1], substep_goals[:, 2], color="red")
        ax1.scatter(substep_starts[:, 0], substep_starts[:, 1],
                    substep_starts[:, 2], color="green", linewidth=3)
        ax1.scatter(substep_ends[:, 0], substep_ends[:, 1],
                    substep_ends[:, 2], color="blue", linewidth=3)


        for idx in range(substep_starts.shape[0]):
            ax1.plot([substep_starts[idx, 0], substep_ends[idx, 0]], [
                substep_starts[idx, 1], substep_ends[idx, 1]], [
                substep_starts[idx, 2], substep_ends[idx, 2]], linewidth=1)
        '''

    for i, traj in enumerate(trajectories):
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2)
        ax1.text(traj[0, 0], traj[0, 1], traj[0, 2],
                 str(i), size=20, zorder=1, color='k')
    #ax1.scatter(substep_goals[:, 0], substep_goals[:, 1], substep_goals[:, 2], linewidth=2, marker='o')

    plt.show()
