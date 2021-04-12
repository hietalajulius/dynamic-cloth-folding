import tracemalloc
import linecache
import argparse
from gym.envs.robotics import task_definitions
import os
from rlkit.envs.wrappers import NormalizedBoxEnv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def plot_trajectory(ee_initial, current_ee_positions, desired_starts, desired_ends):
    fig = plt.figure()
    ax1 = Axes3D(fig)
    ax1.set_xlim3d(-0.02, 0.18)
    ax1.set_ylim3d(-0.1, 0.1)
    ax1.set_zlim3d(-0.02, 0.18)
    ax1.set_box_aspect((1, 1, 1))

    ax1.plot(current_ee_positions[:, 0], current_ee_positions[:,
                                                              1], current_ee_positions[:, 2], linewidth=3, label="achieved", color="blue")
    for i, (ds, de) in enumerate(zip(desired_starts, desired_ends)):
        ax1.plot([ds[0], de[0]], [ds[1], de[1]], [ds[2], de[2]],
                 linewidth=2, label="desired " + str(i), color="orange")

    ax1.text(0, 0,
             0, "start", size=10, zorder=1, color='k')

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


def get_obs_processor(observation_key, additional_keys, desired_goal_key):
    def obs_processor(o):
        obs = o[observation_key]
        for additional_key in additional_keys:
            obs = np.hstack((obs, o[additional_key]))

        return np.hstack((obs, o[desired_goal_key]))
    return obs_processor


def get_tracking_score(ee_positions, goals):
    norm = np.linalg.norm(ee_positions-goals, axis=1)
    return np.sum(norm)


def ATE(ee_positions, goals):
    squared_norms = np.linalg.norm(ee_positions-goals, axis=1)**2
    return np.sqrt(squared_norms.mean())

def argsparser():
    parser = argparse.ArgumentParser("Parser")
    # Generic
    parser.add_argument('--run',  default=1, type=int)
    parser.add_argument('--title', default="notitle", type=str)
    parser.add_argument('--computer', default="laptop", type=str)
    parser.add_argument('--num_processes', type=int, default=1)
    # TODO: no traditional logging at all
    parser.add_argument('--log_tabular_only', type=int, default=0)

    # Train
    parser.add_argument('--train_steps', default=1000, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--save_policy_every_epoch', default=1, type=int)
    parser.add_argument('--num_cycles', default=20, type=int)
    parser.add_argument('--min_expl_steps', type=int, default=0)
    parser.add_argument('--num_eval_rollouts', type=int, default=20)
    parser.add_argument('--num_eval_param_buckets', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--debug_same_batch', type=int, default=0)

    # Replay buffer
    # HER 0.8 from paper
    parser.add_argument('--her_percent', default=0.8, type=float)
    parser.add_argument('--buffer_size', default=1E6, type=int)

    # Collection
    parser.add_argument('--max_path_length', default=25, type=int)

    # Controller optimization
    parser.add_argument('--ctrl_eval_file', type=int, default=0)
    parser.add_argument('--ctrl_eval', type=int, default=0)

    # Env
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--filter', type=float, default=0.03)
    parser.add_argument('--sphere_clipping', type=int, default=0)
    parser.add_argument('--error_norm_coef', type=float, default=1.0)
    parser.add_argument('--stay_in_place_coef', type=float, default=1.0)
    parser.add_argument('--cosine_sim_coef', type=float, default=0.03)
    parser.add_argument('--output_max', type=float, default=0.07)
    parser.add_argument('--damping_ratio', type=float, default=1)
    parser.add_argument('--kp', type=float, default=1000.0)
    parser.add_argument('--constant_goal', type=int, default=1)
    parser.add_argument('--task', type=str, default="sideways_franka_easy")
    parser.add_argument('--velocity_in_obs', type=int, default=1)
    parser.add_argument('--image_training', default=0, type=int)
    parser.add_argument('--image_size', type=int, default=100)
    parser.add_argument('--randomize_params', type=int, default=0)
    parser.add_argument('--randomize_geoms', type=int, default=0)
    parser.add_argument('--uniform_jnt_tend', type=int, default=1)
    parser.add_argument('--sparse_dense', type=int, default=1)
    parser.add_argument('--goal_noise_range', type=tuple, default=(0, 0.01))
    parser.add_argument('--reward_offset', type=float, default=1.0)

    args = parser.parse_args()
    return args

def get_variant(args):
    variant = dict(
        algorithm="SAC",
        layer_size=256,
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        path_collector_kwargs=dict(),
        policy_kwargs=dict(),
        replay_buffer_kwargs=dict(),
        algorithm_kwargs=dict()
    )
    variant['random_seed'] = args.seed
    variant['version'] = args.title
    variant['image_training'] = bool(args.image_training)
    variant['num_processes'] = int(args.num_processes)
    variant['log_tabular_only'] = bool(args.log_tabular_only)

    variant['algorithm_kwargs'] = dict(
        num_epochs=args.num_epochs,
        num_trains_per_train_loop=args.train_steps,
        num_expl_steps_per_train_loop=args.train_steps,
        num_train_loops_per_epoch=int(args.num_cycles),
        max_path_length=int(args.max_path_length),
        num_eval_rollouts_per_epoch=args.num_eval_rollouts,
        num_eval_param_buckets=args.num_eval_param_buckets,
        save_policy_every_epoch=args.save_policy_every_epoch,
        min_num_steps_before_training=args.min_expl_steps,
        batch_size=args.batch_size,
        debug_same_batch=bool(args.debug_same_batch)
    )

    variant['replay_buffer_kwargs'] = dict(
        max_size=int(args.buffer_size),
        fraction_goals_rollout_goals=1 - args.her_percent
    )

    variant['env_kwargs'] = dict(
        stay_in_place_coef=args.stay_in_place_coef,
        ctrl_filter=args.filter,
        sphere_clipping=bool(args.sphere_clipping),
        kp=args.kp,
        damping_ratio=args.damping_ratio,
        computer=args.computer,
        cosine_sim_coef=args.cosine_sim_coef,
        error_norm_coef=args.error_norm_coef,
        reward_offset=args.reward_offset,
        constant_goal=bool(args.constant_goal),
        output_max=args.output_max,
        sparse_dense=bool(args.sparse_dense),
        constraints=task_definitions.constraints[args.task],
        pixels=bool(args.image_training),
        goal_noise_range=tuple(args.goal_noise_range),
        randomize_params=bool(args.randomize_params),
        randomize_geoms=bool(args.randomize_geoms),
        uniform_jnt_tend=bool(args.uniform_jnt_tend),
        image_size=args.image_size,
        random_seed=args.seed,
        velocity_in_obs=bool(args.velocity_in_obs)
    )


    if args.image_training:
        channels = 1
        variant['policy_kwargs'] = dict(
            input_width=args.image_size,
            input_height=args.image_size,
            input_channels=channels,
            kernel_sizes=[3, 3, 3, 3],
            n_channels=[32, 32, 32, 32],
            strides=[2, 2, 2, 2],
            paddings=[0, 0, 0, 0],
            hidden_sizes=[256, 256, 256, 256],
            init_w=1e-4
        )
        variant['path_collector_kwargs']['additional_keys'] = [
            'robot_observation']
        variant['replay_buffer_kwargs']['internal_keys'] = [
            'image', 'model_params', 'robot_observation']

    else:
        variant['path_collector_kwargs']['additional_keys'] = [
            'robot_observation']
        variant['replay_buffer_kwargs']['internal_keys'] = [
            'model_params', 'robot_observation']

    return variant
