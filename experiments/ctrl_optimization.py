"""
Script to showcase domain randomization functionality.
"""

import robosuite.utils.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import DomainRandomizationWrapper

from gym.envs.robotics import task_definitions
from utils import get_variant, argsparser, get_robosuite_env, ATE, get_tracking_score, plot_trajectory, render_env
import cv2
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhCNNGaussianPolicy, GaussianPolicy, GaussianCNNPolicy
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import pandas as pd


class TestAgent(object):
    def __init__(self, actions):
        self.actions = actions
        self.current_action = 0

    def get_action(self, obs=None):
        action = copy.deepcopy(self.actions[self.current_action])
        self.current_action += 1
        return action

    def reset(self):
        self.current_action = 0


def eval_settings(variant, agent, render=False, plot=False, max_steps=20, obs_processor=None):
    env = get_robosuite_env(variant, evaluation=render)
    o = env.reset()
    agent.reset()

    start = env.sim.data.get_site_xpos(
        'gripper0_grip_site').copy()

    current_ee_positions = []
    desired_starts = []
    desired_ends = []

    ideal_positions = []

    current_ideal_pos = start.copy()

    success = False

    for _ in range(max_steps):
        if not obs_processor is None:
            o = obs_processor(o)
            delta = agent.get_action(o)[0]
        else:
            delta = agent.get_action(o)

        delta_in_space = delta * variant['ctrl_kwargs']['output_max']

        ee_pos = env.sim.data.get_site_xpos('gripper0_grip_site').copy()
        current_desired_start = ee_pos
        current_desired_end = ee_pos + delta_in_space
        o, reward, done, info = env.step(delta)
        if reward >= 0:
            success = True

        current_ideal_pos += delta_in_space

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
        plot_trajectory(start, current_ee_positions,
                        ideal_positions, desired_starts, desired_ends)

    return tracking_score, ate, success, current_ee_positions


def get_actions(idx, variant):
    positions = np.load(f"../traj_opt/ee_reached_{str(idx)}.npy")
    positions = [[-0.07334889, -0.03174962, 0.14356065]] + positions.tolist()
    return np.array([np.array(positions[i+1]) - np.array(positions[i])
                     for i in range(len(positions)-1)]) / variant['ctrl_kwargs']['output_max']


if __name__ == "__main__":

    args = argsparser()
    variant = get_variant(args)

    if variant['ctrl_kwargs']['ctrl_eval']:

        actions = get_actions(
            variant['ctrl_kwargs']['ctrl_eval_file'], variant)
        agent = TestAgent(actions)

        tracking_score, ate, _, _ = eval_settings(
            variant, agent, render=True, plot=True, max_steps=actions.shape[0])

        print(f"Tracking: {tracking_score}, ATE: {ate}")
    else:
        stats_df = pd.DataFrame(
            columns=['kp', 'damping_ratio', 'ramp_ratio', 'output_max', 'action_file', 'score', 'ate'])
        kp_range = np.linspace(100, 2000, 10)
        damping_ratio_range = np.linspace(0.5, 2, 10)
        ramp_ratio_range = [0.1, 0.2, 0.5, 0.7, 1]
        output_max_range = [0.02, 0.03, 0.04, 0.05]
        action_files = [0, 1, 2, 3, 4, 5]

        best_tracking_score = np.inf
        settings = dict()
        best_settings = dict()

        tries = 0
        for af in action_files:
            for om in output_max_range:
                variant['ctrl_kwargs']['output_max'] = om
                actions = get_actions(af, variant)
                agent = TestAgent(actions)
                agent = TestAgent(actions)
                for kp in kp_range:
                    for dr in damping_ratio_range:
                        for rr in ramp_ratio_range:
                            tries += 1
                            variant['ctrl_kwargs']['kp'] = kp
                            variant['ctrl_kwargs']['damping_ratio'] = dr
                            variant['ctrl_kwargs']['ramp_ratio'] = rr

                            tracking_score, ate, success, _ = eval_settings(
                                variant, agent, max_steps=actions.shape[0])

                            settings['kp'] = kp
                            settings['ramp_ratio'] = rr
                            settings['damping_ratio'] = dr
                            settings['output_max'] = om
                            settings['action_file'] = af
                            settings['success'] = success
                            settings['score'] = tracking_score
                            settings['ate'] = ate

                            stats_df = stats_df.append(
                                settings, ignore_index=True)
                            stats_df.to_csv("../traj_opt/ctrl_stats.csv")

                            print(tries, "Tracking score/ate (kp, dr, rr, om, af, suc)",
                                  kp, dr, rr, om, af, success, tracking_score, ate)

                            if tracking_score < best_tracking_score:
                                best_tracking_score = tracking_score
                                best_settings = copy.deepcopy(settings)
                                print("New best settings", best_settings, "\n")

        print("Best settings", best_settings)
