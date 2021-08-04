import argparse
from clothmanip.utils import task_definitions
import os
import numpy as np
from robosuite.wrappers import DomainRandomizationWrapper
import sys
from git import Repo
import json

def get_keys_and_dims(variant, env):
    obs_dim = env.observation_space.spaces['observation'].low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    action_dim = env.action_space.low.size
    policy_obs_dim = obs_dim + goal_dim
    value_input_size = obs_dim + action_dim + goal_dim
    added_fc_input_size = goal_dim

    if 'robot_observation' in env.observation_space.spaces:
        robot_obs_dim = env.observation_space.spaces['robot_observation'].low.size
        policy_obs_dim += robot_obs_dim
        added_fc_input_size += robot_obs_dim

    image_training = variant['image_training']
    if image_training:
        path_collector_observation_key = 'image'
    else:
        path_collector_observation_key = 'observation'

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    keys = {
        'path_collector_observation_key': path_collector_observation_key,
        'observation_key': observation_key,
        'desired_goal_key': desired_goal_key,
        'achieved_goal_key': achieved_goal_key
    }

    dims = {
        'value_input_size': value_input_size,
        'action_dim': action_dim,
        'added_fc_input_size': added_fc_input_size,
        'policy_obs_dim': policy_obs_dim,
    }

    return keys, dims

def dump_commit_hashes(save_folder):
    robotics_dir = os.getenv('ROBOTICS_PATH')
    repos = ['mujoco-py', 'rlkit', 'clothmanip', 'gym', 'robosuite']
    commit_hashes = dict()
    for repo_name in repos:
        repo = Repo.init(os.path.join(robotics_dir, repo_name))
        repo_value = dict(hash=str(repo.head.commit), message=repo.head.commit.message)
        commit_hashes[repo_name] = repo_value

    with open(f"{save_folder}/commit_hashes.json", "w") as outfile:
            json.dump(commit_hashes, outfile)

def dump_goal(save_folder, goal):
    np.savetxt(f"{save_folder}/goal.txt", goal, delimiter=",", fmt='%f')

    

LIGHTING_ARGS = {
    'light_names': None,  # all lights are randomized
    'randomize_position': True,
    'randomize_direction': True,
    'randomize_specular': True,
    'randomize_ambient': True,
    'randomize_diffuse': True,
    'randomize_active': True,
    'position_perturbation_size': 0.99,
    'direction_perturbation_size': 0.99,
    'specular_perturbation_size': 0.99,
    'ambient_perturbation_size': 0.99,
    'diffuse_perturbation_size': 0.99,
}

CAMERA_ARGS = {
    'camera_names': None,  # all cameras are randomized
    'randomize_position': True,
    'randomize_rotation': True,
    'randomize_fovy': True,
    'position_perturbation_size': 0.2,
    'rotation_perturbation_size': 0.75,
    'fovy_perturbation_size': 0.05,
}



def get_randomized_env(env, variant):
    r = variant['env_kwargs']['randomization_kwargs']
    return DomainRandomizationWrapper(
                env,
                xml_randomization_kwargs=r,
                randomize_on_reset=True,
                randomize_camera=r['lights_randomization'],
                randomize_every_n_steps=0,
                randomize_color=False,
                camera_randomization_args=CAMERA_ARGS,
                randomize_lighting=r['camera_position_randomization'],
                lighting_randomization_args=LIGHTING_ARGS
                )



def argsparser():
    parser = argparse.ArgumentParser("Parser")
    # Generic
    parser.add_argument('--title', type=str, default="default")
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--folder', type=str)

    # Train
    parser.add_argument('--train_steps', default=1000, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--save_policy_every_epoch', default=1, type=int)
    parser.add_argument('--save_images_every_epoch', default=10, type=int)
    parser.add_argument('--num_cycles', default=20, type=int)
    parser.add_argument('--min_expl_steps', type=int, default=0)
    parser.add_argument('--num_eval_rollouts', type=int, default=20)
    parser.add_argument('--use_eval_suite', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--corner_prediction_loss_coef', type=float, default=0.001)
    parser.add_argument('--fc_layer_size', type=int, default=512)
    parser.add_argument('--fc_layer_depth', type=int, default=5)

    # Replay buffer
    # HER 0.8 from paper
    parser.add_argument('--her_percent', default=0.8, type=float)
    parser.add_argument('--buffer_size', default=1E6, type=int)
    parser.add_argument('--num_pre_demos', type=int, default=0)
    parser.add_argument('--num_demoers', type=int, default=0)


    # Collection
    parser.add_argument('--max_path_length', default=25, type=int)
    parser.add_argument('--max_close_steps', default=5,type=int)

    parser.add_argument('--lights_randomization', default=1, type=int)
    parser.add_argument('--materials_randomization', default=1, type=int)
    parser.add_argument('--camera_position_randomization', default=1, type=int)
    parser.add_argument('--lookat_position_randomization_radius', default=0.03, type=float)
    parser.add_argument('--lookat_position_randomization', default=1, type=int)
    parser.add_argument('--albumentations_randomization', default=1, type=int)
    parser.add_argument('--dynamics_randomization', default=1, type=int)

    parser.add_argument('--camera_type', choices=["up", "side", "front", "all"], default="side")
    parser.add_argument('--camera_config', choices=["small", "large"], default="small")
    parser.add_argument('--cloth_size', default=0.2, type=float)

    # Env
    parser.add_argument('--image_obs_noise_mean', type=float, default=0.5)
    parser.add_argument('--image_obs_noise_std', type=float, default=0.5)

    parser.add_argument('--smallest_key', choices=["corner_0_distance", "corner_sum_distance"], default="corner_sum_distance")
    parser.add_argument('--robot_observation', choices=["ee", "ctrl", "none"], default="ctrl")
    parser.add_argument('--filter', type=float, default=0.03)
    parser.add_argument('--output_max', type=float, default=0.03)
    parser.add_argument('--damping_ratio', type=float, default=1)
    parser.add_argument('--kp', type=float, default=1000.0)
    parser.add_argument('--constant_goal', type=int, default=0)
    parser.add_argument('--task', type=str, default="sideways")
    parser.add_argument('--success_distance', type=float, default=0.05)
    parser.add_argument('--image_training', default=1, type=int)
    parser.add_argument('--image_size', type=int, default=100)
    parser.add_argument('--frame_stack_size', type=int, default=1)
    parser.add_argument('--sparse_dense', type=int, default=1)
    parser.add_argument('--goal_noise', type=float, default=0.03)
    parser.add_argument('--success_reward', type=int, default=0)
    parser.add_argument('--fail_reward', type=int, default=-1)
    parser.add_argument('--extra_reward', type=int, default=1)
    parser.add_argument('--timestep', type=float, default=0.01)
    parser.add_argument('--control_frequency', type=int, default=10)


    args = parser.parse_args()
    return args

def get_variant(args):
    arg_str = ""
    for arg in sys.argv:
        arg_str += arg
        arg_str += " "
    
    variant = dict(
        algorithm="SAC",
        folder=args.folder,
        fc_layer_size=args.fc_layer_size,
        fc_layer_depth=args.fc_layer_depth,
        trainer_kwargs=dict(
            discount=args.discount,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            corner_prediction_loss_coef=args.corner_prediction_loss_coef
        ),
        path_collector_kwargs=dict(),
        policy_kwargs=dict(),
        replay_buffer_kwargs=dict(),
        algorithm_kwargs=dict(),
        num_pre_demos=args.num_pre_demos,
        num_demoers=args.num_demoers,
        demo_paths=[]
    )
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    envs_dir = os.path.join(utils_dir, "..", "envs")
 
    variant['demo_paths'].append(os.path.join(envs_dir, f"cloth_data", "scaled_executable_raw_actions.csv"))
        
    variant['random_seed'] = args.run
    variant['image_training'] = bool(args.image_training)
    variant['num_processes'] = int(args.num_processes)
    variant['save_images_every_epoch'] = args.save_images_every_epoch

    variant['num_eval_rollouts'] = args.num_eval_rollouts
    variant['use_eval_suite'] =bool(args.use_eval_suite)

    variant['algorithm_kwargs'] = dict(
        num_epochs=args.num_epochs,
        num_trains_per_train_loop=args.train_steps,
        num_expl_steps_per_train_loop=args.train_steps,
        num_train_loops_per_epoch=int(args.num_cycles),
        max_path_length=int(args.max_path_length),
        save_policy_every_epoch=args.save_policy_every_epoch,
        min_num_steps_before_training=args.min_expl_steps,
        batch_size=args.batch_size
    )

    variant['replay_buffer_kwargs'] = dict(
        max_size=int(args.buffer_size),
        fraction_goals_rollout_goals=1 - args.her_percent
    )
    
    if args.camera_config == "small":
        camera_config = dict(type="small", fovy_range=(13,15), height=100, width=848)
    else:
        camera_config = dict(type="large", fovy_range=(57,59), height=480, width=848)
    
    dummy_constraints = task_definitions.constraints[args.task](0, 4, 8, args.success_distance)
    constraint_distances = [c['distance'] for c in dummy_constraints]
        
    variant['env_kwargs'] = dict(
        timestep=args.timestep,
        task=args.task,
        smallest_key=args.smallest_key,
        success_distance=args.success_distance,
        image_size=args.image_size,
        randomization_kwargs=dict(
            lights_randomization=bool(args.lights_randomization),
            materials_randomization=bool(args.materials_randomization),
            camera_position_randomization=bool(args.camera_position_randomization),
            lookat_position_randomization=bool(args.lookat_position_randomization),
            lookat_position_randomization_radius=args.lookat_position_randomization_radius,
            dynamics_randomization=bool(args.dynamics_randomization),
            albumentations_randomization=bool(args.albumentations_randomization),
            cloth_size=args.cloth_size,
            camera_type=args.camera_type,
            camera_config=camera_config,
        ),
        robot_observation=args.robot_observation,
        control_frequency=args.control_frequency,
        ctrl_filter=args.filter,
        kp=args.kp,
        frame_stack_size=args.frame_stack_size,
        damping_ratio=args.damping_ratio,
        success_reward=args.success_reward,
        fail_reward=args.fail_reward,
        extra_reward=args.extra_reward,
        constant_goal=bool(args.constant_goal),
        output_max=args.output_max,
        max_close_steps=int(args.max_close_steps),
        sparse_dense=bool(args.sparse_dense),
        constraint_distances=constraint_distances,
        pixels=bool(args.image_training),
        goal_noise_range=(0.0, args.goal_noise),
        image_obs_noise_mean=args.image_obs_noise_mean,
        image_obs_noise_std=args.image_obs_noise_std,
        
    )



    if args.image_training:
        variant['policy_kwargs'] = dict(
            input_width=args.image_size,
            input_height=args.image_size,
            input_channels=args.frame_stack_size,
            kernel_sizes=[3, 3, 3, 3],
            n_channels=[32, 32, 32, 32],
            strides=[2, 2, 2, 2],
            paddings=[0, 0, 0, 0],
            hidden_sizes_aux=[256, 8],
            hidden_sizes_main=[256, 256, 256, 256],
            init_w=1e-4,
        )
        variant['path_collector_kwargs']['additional_keys'] = [
            'robot_observation']
        variant['replay_buffer_kwargs']['internal_keys'] = [
            'image', 'robot_observation']

    else:
        variant['path_collector_kwargs']['additional_keys'] = []
        variant['replay_buffer_kwargs']['internal_keys'] = []

    return variant, arg_str
