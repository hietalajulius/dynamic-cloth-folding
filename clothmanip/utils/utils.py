import tracemalloc
import linecache
import argparse
from clothmanip.utils import task_definitions
import os
from rlkit.envs.wrappers import NormalizedBoxEnv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import mujoco_py
from robosuite.wrappers import DomainRandomizationWrapper

from scipy import spatial
import sys
from clothmanip.utils import mujoco_model_kwargs
from pathlib import Path


COLOR_ARGS = {
    'geom_names': None,  # all geoms are randomized
    'randomize_local': False,  # sample nearby colors
    'randomize_material': True,  # randomize material reflectance / shininess / specular
    'local_rgb_interpolation': 0.02,
    'local_material_interpolation': 0.05,
    # all texture variation types
    'texture_variations': ['rgb', 'checker', 'noise', 'gradient'],
    'randomize_skybox': False,  # by default, randomize skybox too
}


LIGHTING_ARGS = {
    'light_names': None,  # all lights are randomized
    'randomize_position': True,
    'randomize_direction': True,
    'randomize_specular': True,
    'randomize_ambient': True,
    'randomize_diffuse': True,
    'randomize_active': True,
    'position_perturbation_size': 0.05,
    'direction_perturbation_size': 0.05,
    'specular_perturbation_size': 0.01,
    'ambient_perturbation_size': 0.01,
    'diffuse_perturbation_size': 0.01,
}

CAMERA_ARGS = {
    'camera_names': None,  # all cameras are randomized
    'randomize_position': True,
    'randomize_rotation': True,
    'randomize_fovy': True,
    'position_perturbation_size': 0.05,
    'rotation_perturbation_size': 0.05,
    'fovy_perturbation_size': 1,
}

def get_randomized_env(env):
    return DomainRandomizationWrapper(
                env,
                randomize_on_reset=True,
                randomize_camera=True,
                randomize_every_n_steps=0,
                randomize_color=True,
                camera_randomization_args=CAMERA_ARGS,
                color_randomization_args=COLOR_ARGS,
                randomize_lighting=True,
                randomize_blur=True,
                custom_randomize_color=True,
                lighting_randomization_args=LIGHTING_ARGS)


def deltas_from_positions(positions):
    #Should be logged in traj instead...
    deltas = []
    for i in range(positions.shape[0] -1):
        delta = positions[i+1] -positions[i]
        deltas.append(delta)
    return np.array(deltas)

def get_obs_processor(observation_key, additional_keys, desired_goal_key):
    def obs_processor(o):
        obs = o[observation_key]
        for additional_key in additional_keys:
            obs = np.hstack((obs, o[additional_key]))

        return np.hstack((obs, o[desired_goal_key]))
    return obs_processor

def calculate_cosine_distances(deltas):
    distances = []
    print("delats sha", deltas.shape)
    for i in range(deltas.shape[0] -1):
        vec1 = deltas[i+1]
        vec2 = deltas[i]
        distance = spatial.distance.cosine(vec1, vec2)
        print("dist", distance)
        distances.append(distance)
    return np.array(distances)



def calculate_ate(ee_positions, goals):
    squared_norms = np.linalg.norm(ee_positions-goals, axis=1)**2
    return np.sqrt(squared_norms.mean())

def argsparser():
    parser = argparse.ArgumentParser("Parser")
    # Generic
    parser.add_argument('--run',  default=1, type=int)
    parser.add_argument('--title', default="notitle", type=str)
    parser.add_argument('--num_processes', type=int, default=1)
    # TODO: no traditional logging at all
    parser.add_argument('--log_tabular_only', type=int, default=0)

    # Train
    parser.add_argument('--legacy_cnn', default=0, type=int)
    parser.add_argument('--train_steps', default=1000, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--save_policy_every_epoch', default=1, type=int)
    parser.add_argument('--save_images_every_epoch', default=5, type=int)
    parser.add_argument('--num_cycles', default=20, type=int)
    parser.add_argument('--min_expl_steps', type=int, default=0)
    parser.add_argument('--num_eval_rollouts', type=int, default=20)
    parser.add_argument('--num_eval_param_buckets', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--debug_same_batch', type=int, default=0)
    parser.add_argument('--conv_normalization_type', type=str, default='none')
    parser.add_argument('--fc_normalization_type', type=str, default='none')
    parser.add_argument('--pool_type', type=str, default='none')
    parser.add_argument('--discount', type=float, default=0.99)

    # Replay buffer
    # HER 0.8 from paper
    parser.add_argument('--her_percent', default=0.8, type=float)
    parser.add_argument('--buffer_size', default=1E6, type=int)
    parser.add_argument('--use_demos', required=True, type=int)
    parser.add_argument('--num_demos', type=int, default=0)
    parser.add_argument('--num_demoers', type=int, default=0)
    parser.add_argument('--demo_coef', type=float, default=1.0)


    # Collection
    parser.add_argument('--max_path_length', default=50, type=int)
    parser.add_argument('--domain_randomization', required=True, type=int)

    # sim2real
    parser.add_argument('--eval_folder', type=str, required=False)

    # Env
    parser.add_argument('--ate_penalty_coef', type=float, default=0)
    parser.add_argument('--action_norm_penalty_coef', type=float, default=0)
    parser.add_argument('--cosine_penalty_coef', type=float, default=0)
    parser.add_argument('--image_obs_noise_mean', type=float, default=1.0)
    parser.add_argument('--image_obs_noise_std', type=float, default=0.0)

    parser.add_argument('--cloth_type', choices=["bath", "kitchen", "wipe"], required=True)
    parser.add_argument('--robot_observation', choices=["all", "joints", "ee", "ctrl"], default="all")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--filter', type=float, default=0.03)
    parser.add_argument('--clip_type', type=str, default="none")
    parser.add_argument('--output_max', type=float, required=True)
    parser.add_argument('--damping_ratio', type=float, default=1)
    parser.add_argument('--kp', type=float, default=1000.0)
    parser.add_argument('--constant_goal', type=int, default=1)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--image_training', default=0, type=int, required=True)
    parser.add_argument('--image_size', type=int, default=100)
    parser.add_argument('--depth_frames', type=int, default=0)
    parser.add_argument('--frame_stack_size', type=int, default=1)
    parser.add_argument('--sparse_dense', type=int, default=0)
    parser.add_argument('--goal_noise', type=float, default=0.01)
    parser.add_argument('--success_reward', type=int, required=True)
    parser.add_argument('--fail_reward', type=int, required=True)
    parser.add_argument('--extra_reward', type=int, required=True)
    parser.add_argument('--timestep', type=float, default=0.01)
    parser.add_argument('--control_frequency', type=int, default=10)


    args = parser.parse_args()
    return args

def get_variant(args):
    arg_str = ""
    for arg in sys.argv:
        arg_str += arg
        arg_str += " "

    utils_dir = Path(os.path.abspath(__file__))
    demo_path = os.path.join(utils_dir.parent.parent.parent.absolute(), "experiments", f"executable_deltas_{args.task}_{args.cloth_type}.csv")

    variant = dict(
        algorithm="SAC",
        layer_size=256,
        domain_randomization=bool(args.domain_randomization),
        trainer_kwargs=dict(
            discount=args.discount,
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
        algorithm_kwargs=dict(),
        eval_folder=args.eval_folder,
        use_demos=bool(args.use_demos),
        demo_path=demo_path,
        num_demos=args.num_demos,
        num_demoers=args.num_demoers,
        demo_coef=args.demo_coef
    )
    variant['random_seed'] = args.seed
    variant['version'] = args.title
    variant['image_training'] = bool(args.image_training)
    variant['num_processes'] = int(args.num_processes)
    variant['log_tabular_only'] = bool(args.log_tabular_only)
    variant['legacy_cnn'] = bool(args.legacy_cnn)
    variant['save_images_every_epoch'] = args.save_images_every_epoch

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

    if args.cloth_type == "bath":
        cloth_specific_kwargs = mujoco_model_kwargs.BATH_MODEL_KWARGS
    elif args.cloth_type == "kitchen":
        cloth_specific_kwargs = mujoco_model_kwargs.KITCHEN_MODEL_KWARGS
    elif args.cloth_type == "wipe":
        cloth_specific_kwargs = mujoco_model_kwargs.WIPE_MODEL_KWARGS
    else:
        raise("Bad cloth type provided")

    model_kwargs = dict(
            **cloth_specific_kwargs,
            timestep=args.timestep,
            domain_randomization=bool(args.domain_randomization)
        )

    

    variant['env_kwargs'] = dict(
        mujoco_model_kwargs=model_kwargs,
        robot_observation=args.robot_observation,
        control_frequency=args.control_frequency,
        ctrl_filter=args.filter,
        clip_type=args.clip_type,
        kp=args.kp,
        frame_stack_size=args.frame_stack_size,
        depth_frames=bool(args.depth_frames),
        damping_ratio=args.damping_ratio,
        action_norm_penalty_coef=args.action_norm_penalty_coef,
        ate_penalty_coef=args.ate_penalty_coef,
        cosine_penalty_coef=args.cosine_penalty_coef,
        success_reward=args.success_reward,
        fail_reward=args.fail_reward,
        extra_reward=args.extra_reward,
        constant_goal=bool(args.constant_goal),
        output_max=args.output_max,
        max_episode_steps=int(args.max_path_length),
        sparse_dense=bool(args.sparse_dense),
        constraints=task_definitions.constraints[args.task],
        pixels=bool(args.image_training),
        goal_noise_range=(0.0, args.goal_noise),
        num_eval_rollouts=args.num_eval_rollouts,
        image_obs_noise_mean=args.image_obs_noise_mean,
        image_obs_noise_std=args.image_obs_noise_std,
        
    )



    if args.image_training:
        variant['policy_kwargs'] = dict(
            input_width=args.image_size,
            input_height=args.image_size,
            input_channels=args.frame_stack_size*(1+int(args.depth_frames)),
            kernel_sizes=[3, 3, 3, 3],
            n_channels=[32, 32, 32, 32],
            strides=[2, 2, 2, 2],
            paddings=[0, 0, 0, 0],
            hidden_sizes_aux=[256, 8],
            hidden_sizes_main=[256, 256, 256, 256],
            init_w=1e-4,
            conv_normalization_type=args.conv_normalization_type,
            fc_normalization_type=args.fc_normalization_type,
            pool_type=args.pool_type,
            pool_sizes=[],
            pool_paddings=[],
            pool_strides=[],
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

    return variant, arg_str


def remove_distance_welds(sim):
    """Removes the mocap welds that we use for actuation.
    """
    for i in range(sim.model.eq_data.shape[0]):
        if sim.model.eq_type[i] == mujoco_py.const.EQ_DISTANCE:
            sim.model.eq_active[i] = False
    sim.forward()


def enable_distance_welds(sim):
    """Removes the mocap welds that we use for actuation.
    """
    for i in range(sim.model.eq_data.shape[0]):
        if sim.model.eq_type[i] == mujoco_py.const.EQ_DISTANCE:
            sim.model.eq_active[i] = True
    sim.forward()