import argparse
import torch
import os
import numpy as np
from robosuite import wrappers
from git import Repo
import json
from rlkit.torch import pytorch_util
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


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

    path_collector_observation_key = 'image'

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
    repos = ['./submodules/mujoco-py/',
             './submodules/rlkit/', './submodules/robosuite/', "."]
    commit_hashes = dict()
    for repo_name in repos:
        repo_path = os.path.abspath(repo_name)
        repo = Repo.init(repo_path)
        repo_value = dict(hash=str(repo.head.commit),
                          message=repo.head.commit.message)
        commit_hashes[repo_name] = repo_value

    with open(f"{save_folder}/commit_hashes.json", "w") as outfile:
        json.dump(commit_hashes, outfile)


def dump_goal(save_folder, goal):
    np.savetxt(f"{save_folder}/goal.txt", goal, delimiter=",", fmt='%f')


def get_randomized_env(env, randomization_kwargs):
    cam_args = {
        'camera_names': None,  # all cameras are randomized
        'randomize_position': True,
        'randomize_rotation': True,
        'randomize_fovy': True,
        'position_perturbation_size': randomization_kwargs['position_perturbation_size'],
        'rotation_perturbation_size': randomization_kwargs['rotation_perturbation_size'],
        'fovy_perturbation_size':  randomization_kwargs['fovy_perturbation_size']
    }
    lighting_args = {
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
    return wrappers.DomainRandomizationWrapper(
        env,
        xml_randomization_kwargs=randomization_kwargs,
        randomize_on_reset=True,
        randomize_camera=randomization_kwargs[
            'camera_position_randomization'],
        randomize_every_n_steps=0,
        randomize_color=False,
        camera_randomization_args=cam_args,
        randomize_lighting=randomization_kwargs['lights_randomization'],
        lighting_randomization_args=lighting_args
    )


def argsparser():
    parser = argparse.ArgumentParser("Parser")
    # Generic
    parser.add_argument('--title', type=str, default="default")
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--num-processes', type=int, default=1)

    # Train
    parser.add_argument('--train-steps', default=1000, type=int)  # Per cycle
    parser.add_argument('--num-epochs', default=100, type=int)
    parser.add_argument('--save-policy-every-epoch', default=1, type=int)
    parser.add_argument('--num-cycles', default=20, type=int)  # Per epoch
    parser.add_argument('--num-eval-rollouts', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--corner-prediction-loss-coef',
                        type=float, default=0.001)

    # Sample images from evaluation
    parser.add_argument('--save-images-every-epoch', default=10, type=int)

    # FC network sizes
    parser.add_argument('--fc-layer-size', type=int, default=512)
    parser.add_argument('--fc-layer-depth', type=int, default=5)

    # Replay buffer
    parser.add_argument('--her-percent', default=0.8, type=float)
    parser.add_argument('--buffer-size', default=1E5, type=int)

    # How many of the envs only execute demos
    parser.add_argument('--num-demoers', type=int, default=0)

    # Maximum length for episode
    parser.add_argument('--max-path-length', default=50,
                        type=int)

    # Maximum number of steps the agent is within goal threshold i.e. the task is successful
    parser.add_argument('--max-close-steps', default=10, type=int)

    # Domain randomization
    parser.add_argument('--lights-randomization', default=1, type=int)
    parser.add_argument('--materials-randomization', default=1, type=int)
    parser.add_argument('--camera-position-randomization', default=1, type=int)
    parser.add_argument(
        '--lookat-position-randomization-radius', default=0.03, type=float)
    parser.add_argument('--lookat-position-randomization', default=1, type=int)
    parser.add_argument('--albumentations-randomization', default=1, type=int)
    parser.add_argument('--dynamics-randomization', default=1, type=int)

    parser.add_argument('--fovy-perturbation-size', default=0.05, type=float)
    parser.add_argument('--rotation-perturbation-size',
                        default=0.75, type=float)
    parser.add_argument('--position-perturbation-size',
                        default=0.2, type=float)

    # Camera
    parser.add_argument(
        '--camera-type', choices=["up", "side", "front", "all"], default="side")
    parser.add_argument('--camera-config',
                        choices=["small", "large"], default="small")

    # Env
    parser.add_argument('--cloth-size', default=0.2, type=float)

    # Image capture delays
    parser.add_argument('--image-obs-noise-mean', type=float,
                        default=0.5)
    parser.add_argument('--image-obs-noise-std', type=float, default=0.5)

    # Which observations from the robot to consider
    parser.add_argument('--robot-observation',
                        choices=["ee", "ctrl", "none"], default="ctrl")

    # Filter value to use for controller interpolation
    parser.add_argument('--filter', type=float, default=0.03)
    # Maximum distance that the goal is changed in any direction
    parser.add_argument('--output-max', type=float, default=0.03)
    # Controller daming ratio
    parser.add_argument('--damping-ratio', type=float, default=1)
    # Controller position gain
    parser.add_argument('--kp', type=float, default=1000.0)

    # Distance threshold for success for each corner
    parser.add_argument('--success-distance', type=float, default=0.05)

    parser.add_argument('--frame-stack-size', type=int, default=1)
    parser.add_argument('--sparse-dense', type=int, default=1)
    parser.add_argument('--goal-noise', type=float, default=0.03)
    parser.add_argument('--success-reward', type=int, default=0)
    parser.add_argument('--fail-reward', type=int, default=-1)
    # For sparse dense i.e. extra for being closer to the goal
    parser.add_argument('--extra-reward', type=int, default=1)
    # Mujoco timestep length
    parser.add_argument('--timestep', type=float, default=0.01)
    # Control frequncy in Hz
    parser.add_argument('--control-frequency', type=int, default=10)

    args = parser.parse_args()
    return args


def get_general_kwargs(args, save_folder, title):
    kwargs = dict(
        algorithm="SAC",
        title=title,
        save_folder=save_folder,
        random_seed=args.run,
    )
    return kwargs


def get_eval_kwargs(args, save_folder):
    eval_kwargs = dict(
        save_images_every_epoch=args.save_images_every_epoch,
        num_runs=args.num_eval_rollouts,
        max_path_length=int(args.max_path_length),
        additional_keys=[
            'robot_observation'],
        frame_stack_size=args.frame_stack_size,
        save_blurred_images=bool(args.albumentations_randomization),
        save_folder=save_folder)
    return eval_kwargs


def get_algorithm_kwargs(args, save_folder):
    algorithm_kwargs = dict(
        num_epochs=args.num_epochs,
        num_trains_per_train_loop=args.train_steps,
        num_expl_steps_per_train_loop=args.train_steps,
        num_train_loops_per_epoch=int(args.num_cycles),
        max_path_length=int(args.max_path_length),
        save_policy_every_epoch=args.save_policy_every_epoch,
        batch_size=args.batch_size,
        num_demoers=args.num_demoers,
        save_folder=save_folder
    )
    return algorithm_kwargs


def get_path_collector_kwargs(args):
    path_collector_kwargs = dict(
        additional_keys=[
            'robot_observation'],
        # The demo actions collected in the lab for the sideways fold)
        demo_paths=[os.path.join(f"./data", "demos.csv")],
        demo_divider=args.output_max,
        num_processes=int(args.num_processes),)
    return path_collector_kwargs


def get_value_function_kwargs(args):
    value_function_kwargs = dict(fc_layer_size=args.fc_layer_size,
                                 fc_layer_depth=args.fc_layer_depth,)
    return value_function_kwargs


def get_trainer_kwargs(args):
    trainer_kwargs = dict(
        discount=args.discount,
        soft_target_tau=5e-3,
        target_update_period=1,
        policy_lr=3E-4,
        qf_lr=3E-4,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
        corner_prediction_loss_coef=args.corner_prediction_loss_coef
    )
    return trainer_kwargs


def get_replay_buffer_kwargs(args):
    replay_buffer_kwargs = dict(
        max_size=int(args.buffer_size),
        fraction_goals_rollout_goals=1 - args.her_percent,
        internal_keys=['image', 'robot_observation']
    )
    return replay_buffer_kwargs


def get_env_kwargs(args, save_folder):
    env_kwargs = dict(
        save_folder=save_folder,
        timestep=args.timestep,
        success_distance=args.success_distance,
        robot_observation=args.robot_observation,
        control_frequency=args.control_frequency,
        ctrl_filter=args.filter,
        kp=args.kp,
        frame_stack_size=args.frame_stack_size,
        damping_ratio=args.damping_ratio,
        success_reward=args.success_reward,
        fail_reward=args.fail_reward,
        extra_reward=args.extra_reward,
        output_max=args.output_max,
        max_close_steps=int(args.max_close_steps),
        sparse_dense=bool(args.sparse_dense),
        goal_noise_range=(0.0, args.goal_noise),
        image_obs_noise_mean=args.image_obs_noise_mean,
        image_obs_noise_std=args.image_obs_noise_std,
        model_kwargs_path=os.path.join(f"./data", "model_params.csv")

    )
    return env_kwargs


def get_randomization_kwargs(args):
    if args.camera_config == "small":
        camera_config = dict(type="small", fovy_range=(
            13, 15), height=100, width=848)
    else:
        camera_config = dict(type="large", fovy_range=(
            57, 59), height=480, width=848)
    randomization_kwargs = dict(
        lights_randomization=bool(args.lights_randomization),
        materials_randomization=bool(args.materials_randomization),
        camera_position_randomization=bool(
            args.camera_position_randomization),
        lookat_position_randomization=bool(
            args.lookat_position_randomization),
        lookat_position_randomization_radius=args.lookat_position_randomization_radius,
        dynamics_randomization=bool(args.dynamics_randomization),
        albumentations_randomization=bool(
            args.albumentations_randomization),
        cloth_size=args.cloth_size,
        camera_type=args.camera_type,
        camera_config=camera_config,
        position_perturbation_size=args.position_perturbation_size,
        rotation_perturbation_size=args.rotation_perturbation_size,
        fovy_perturbation_size=args.fovy_perturbation_size
    )
    return randomization_kwargs


def get_policy_kwargs(args):
    policy_kwargs = dict(
        input_width=100,
        input_height=100,
        input_channels=args.frame_stack_size,
        kernel_sizes=[3, 3, 3, 3],
        n_channels=[32, 32, 32, 32],
        strides=[2, 2, 2, 2],
        paddings=[0, 0, 0, 0],
        hidden_sizes_aux=[256, 8],
        hidden_sizes_main=[256, 256, 256, 256],
        init_w=1e-4,
        aux_output_size=9,
    )
    return policy_kwargs


def get_variant(args):
    title = args.title + "-run-" + str(args.run)
    save_folder = os.path.join(os.path.abspath("./"), "trainings", title)

    variant = get_general_kwargs(args, save_folder, title)
    variant['randomization_kwargs'] = get_randomization_kwargs(args)
    variant['value_function_kwargs'] = get_value_function_kwargs(args)
    variant['policy_kwargs'] = get_policy_kwargs(args)
    variant['env_kwargs'] = get_env_kwargs(args, save_folder)
    variant['eval_kwargs'] = get_eval_kwargs(args, save_folder)
    variant['algorithm_kwargs'] = get_algorithm_kwargs(args, save_folder)
    variant['path_collector_kwargs'] = get_path_collector_kwargs(args)
    variant['replay_buffer_kwargs'] = get_replay_buffer_kwargs(args)
    variant['trainer_kwargs'] = get_trainer_kwargs(args)

    return variant


def setup_save_folder(variant):
    os.makedirs(variant["save_folder"], exist_ok=True)

    print(f'Created progress directory to: {variant["save_folder"]}')
    profiling_path = os.path.join(variant["save_folder"], "profiling")
    os.makedirs(profiling_path, exist_ok=True)

    with open(f"{variant['save_folder']}/params.json", "w") as outfile:
        json.dump(variant, outfile)

    dump_commit_hashes(variant['save_folder'])


def setup_training_device():
    if torch.cuda.is_available():
        logger.debug("Training with GPU")
        pytorch_util.set_gpu_mode(True)
    else:
        logger.debug("Training with CPU")
