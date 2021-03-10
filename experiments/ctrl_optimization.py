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


def obs_processor(o):
    obs = o['image']
    for additional_key in ['robot_observation']:
        obs = np.hstack((obs, o[additional_key]))

    return np.hstack((obs, o['desired_goal']))


def get_tracking_score(ee_positions, goals):
    norm = np.linalg.norm(ee_positions-goals, axis=1)
    return np.sum(norm)


def ATE(ee_positions, goals):
    squared_norms = np.linalg.norm(ee_positions-goals, axis=1)**2
    return np.sqrt(squared_norms.mean())


def plot_trajectories(ee_initial, current_ee_positions, ideal_positions, desired_starts, desired_ends):
    fig = plt.figure(figsize=(30, 30))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.view_init(10, -10)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.plot(current_ee_positions[:, 0], current_ee_positions[:,
                                                              1], current_ee_positions[:, 2], linewidth=3, label="achieved", color="blue")
    for i, (ds, de) in enumerate(zip(desired_starts, desired_ends)):
        ax1.plot([ds[0], de[0]], [ds[1], de[1]], [ds[2], de[2]],
                 linewidth=2, label="desired " + str(i), color="orange")

    ax1.plot(ideal_positions[:, 0], ideal_positions[:, 1],
             ideal_positions[:, 2], linewidth=3, label="predefined", color="green")

    ax1.text(ee_initial[0], ee_initial[1],
             ee_initial[2], "start", size=10, zorder=1, color='k')

    plt.legend()

    plt.show()


def render_env(env):
    cameras_to_render = ["birdview", "frontview"]

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
    time.sleep(0.1)


def eval_settings(variant, kp, dr, rr, deltas, render=False, plot=False):
    variant['ctrl_kwargs']['kp'] = kp
    variant['ctrl_kwargs']['damping_ratio'] = dr
    variant['ctrl_kwargs']['ramp_ratio'] = rr

    env = get_robosuite_env(variant)
    env.reset()

    start = env.sim.data.get_site_xpos(
        'gripper0_grip_site').copy()

    current_ee_positions = []
    desired_starts = []
    desired_ends = []

    ideal_positions = []

    current_ideal_pos = start.copy()

    for delta in deltas:
        if variant["ctrl_kwargs"]["control_delta"]:
            ee_pos = env.sim.data.get_site_xpos('gripper0_grip_site').copy()
            current_desired_start = ee_pos
            current_desired_end = ee_pos + delta
            _, reward, done, info = env.step(
                delta/variant['ctrl_kwargs']['output_max'])
        else:
            current_desired_start = current_ideal_pos
            current_desired_end = current_ideal_pos + delta
            _, reward, done, info = env.step(current_ideal_pos + delta)

        current_ideal_pos += delta

        current_ee_positions.append(
            env.sim.data.get_site_xpos('gripper0_grip_site').copy())
        ideal_positions.append(current_ideal_pos.copy())

        desired_starts.append(current_desired_start.copy())
        desired_ends.append(current_desired_end.copy())

        if render:
            render_env(env)

    current_ee_positions = np.array(current_ee_positions)
    desired_starts = np.array(desired_starts)
    desired_ends = np.array(desired_ends)
    ideal_positions = np.array(ideal_positions)

    tracking_score = get_tracking_score(
        current_ee_positions, desired_ends)

    ate = ATE(current_ee_positions, desired_ends)

    if plot:
        plot_trajectories(start, current_ee_positions,
                          ideal_positions, desired_starts, desired_ends)

    return tracking_score, ate


if __name__ == "__main__":

    args = argsparser()
    variant = get_variant(args)

    positions = np.load("traj_opt/ee_reached.npy")

    positions = [[-0.07334889, -0.03174962, 0.14356065]] + positions.tolist()

    deltas = [np.array(positions[i+1]) - np.array(positions[i])
              for i in range(len(positions)-1)]

    '''
    deltas = [np.array([0.03, 0.03, 0.03]) for i in range(1, 5)]

    rang = np.linspace(0.01*5, -0.01, 10)

    for ramp in range(1, 10):
        deltas.append(np.array([0.01, 0.01, rang[ramp]]))
    '''

    eval_ = True

    # New best settings {'kp': 475.0, 'ramp_ratio': 0.2, 'damping_ratio': 0.5, 'score': 0.06227582305065782}
    # New best settings {'kp': 650.0, 'ramp_ratio': 0.5, 'damping_ratio': 0.5, 'score': 0.054428931131875635}
    # {'kp': 800.0, 'ramp_ratio': 0.2, 'damping_ratio': 0.7, 'score': 0.05137783876823295}]

    # 'kp': 650.0, 'ramp_ratio': 0.5, 'damping_ratio': 0.5
    if eval_:
        eval_kp = 650
        eval_dr = 0.5
        eval_rr = 0.5

        variant['robosuite_kwargs']['offscreen_renderer'] = True

        tracking_score, ate = eval_settings(
            variant, eval_kp, eval_dr, eval_rr, deltas, render=True, plot=True)

        print(f"Tracking: {tracking_score}, ATE: {ate}")
    else:
        variant['robosuite_kwargs']['offscreen_renderer'] = False
        kp_range = np.linspace(2000, 4000, 10)
        damping_ratio_range = np.linspace(0.3, 3, 10)
        #ramp_ratio_range = [0.1, 0.2, 0.5, 0.7, 1]
        ramp_ratio_range = [1]

        best_tracking_score = np.inf
        best_settings = dict()

        new_bests = []
        tries = 0
        for kp in kp_range:
            for dr in damping_ratio_range:
                for rr in ramp_ratio_range:
                    tries += 1
                    tracking_score, ate = eval_settings(
                        variant, kp, dr, rr, deltas)

                    print(tries, "Tracking score (kp, dr, rr)",
                          kp, dr, rr, tracking_score)
                    if tracking_score < best_tracking_score:
                        best_settings['kp'] = kp
                        best_settings['ramp_ratio'] = rr
                        best_settings['damping_ratio'] = dr
                        best_settings['score'] = tracking_score

                        best_tracking_score = tracking_score
                        print("New best settings", best_settings, "\n")
                        new_bests.append(best_settings)

        print("FINAL BEST SETTINGS:", new_bests)
