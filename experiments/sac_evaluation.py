import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.data_collector import GoalConditionedPathCollector, KeyPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhCNNGaussianPolicy, MonsterTanhCNNGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.her.her import ClothSacHERTrainer
from rlkit.torch.networks import ConcatMlp, MergedCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
import gym
import mujoco_py
import argparse
import torch.nn as nn
import torch
import cProfile

def argsparser():
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('--run',  default=1, type=int)
    parser.add_argument('--title', default="notitle", type=str)
    parser.add_argument('--train_steps', default=100, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_cycles', default=100, type=int)
    parser.add_argument('--her_percent', default=0.0, type=float)
    parser.add_argument('--buffer_size', default=1E6, type=int)
    parser.add_argument('--max_path_length', default=50, type=int)
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--strict', default=1, type=int)
    parser.add_argument('--image_training', default=1, type=int)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--distance_threshold', type=float, default=0.05)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--min_expl_steps', type=int, default=1000)
    parser.add_argument('--randomize_params', type=int, default=1)
    parser.add_argument('--randomize_geoms', type=int, default=0)
    parser.add_argument('--uniform_jnt_tend', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=84)
    parser.add_argument('--rgb', type=int, default=1)
    parser.add_argument('--max_advance', type=float, default=0.05)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--profile', type=int, default=0)
    return parser.parse_args()


def experiment(variant):
    eval_env = NormalizedBoxEnv(gym.make(variant['env_name'], **variant['env_kwargs']))
    expl_env = NormalizedBoxEnv(gym.make(variant['env_name'], **variant['env_kwargs']))

    image_training = variant['image_training']
    if image_training:
        path_collector_observation_key = 'image'
    else:
        path_collector_observation_key = 'observation'

    obs_dim = expl_env.observation_space.spaces['observation'].low.size
    robot_obs_dim = expl_env.observation_space.spaces['robot_observation'].low.size
    model_params_dim = expl_env.observation_space.spaces['model_params'].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    desired_goal_key = 'desired_goal'


    M = variant['layer_size']

    if image_training:
        policy = TanhCNNGaussianPolicy(
            output_size=action_dim,
            added_fc_input_size=robot_obs_dim + goal_dim,
            **variant['policy_kwargs'],
        )
    else:
        policy = TanhGaussianPolicy(
            obs_dim=obs_dim + model_params_dim + goal_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M],
            **variant['policy_kwargs']
        )

    eval_policy = MakeDeterministic(policy)
    
    
    path = "good_policy.mdl"
    if not torch.cuda.is_available():
        eval_policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    else:
        eval_policy.load_state_dict(torch.load(path))

    eval_path_collector = KeyPathCollector(
        eval_env,
        eval_policy,
        render=True,
        render_kwargs={'mode': 'rgb_array'},
        observation_key=path_collector_observation_key,
        desired_goal_key=desired_goal_key,
        **variant['path_collector_kwargs']
    )

    eval_path_collector.collect_new_paths(30, 3000, False)




if __name__ == "__main__":
    # noinspection PyTypeChecker
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
        path_collector_kwargs = dict(),
        policy_kwargs = dict(),
        replay_buffer_kwargs=dict(),
        algorithm_kwargs=dict(),
    )
    args = argsparser()
    variant['env_name'] = args.env_name
    variant['version'] = args.title
    variant['image_training'] = bool(args.image_training)

    variant['algorithm_kwargs'] = dict(
        num_epochs=args.num_epochs,
        num_trains_per_train_loop = args.train_steps,
        num_expl_steps_per_train_loop = args.train_steps,
        num_train_loops_per_epoch = int(args.num_cycles),
        max_path_length = int(args.max_path_length),
        num_eval_steps_per_epoch=args.eval_steps,
        min_num_steps_before_training=args.min_expl_steps,
        batch_size=256,
    )

    variant['replay_buffer_kwargs'] =dict(
            max_size=int(args.buffer_size),
            fraction_goals_env_goals = 0,
            fraction_goals_rollout_goals = 1 - args.her_percent
        )

    variant['env_kwargs'] = dict(
        task=args.task,
        pixels=bool(args.image_training),
        strict=bool(args.strict),
        distance_threshold=args.distance_threshold,
        randomize_params=bool(args.randomize_params),
        randomize_geoms=bool(args.randomize_geoms),
        uniform_jnt_tend=bool(args.uniform_jnt_tend),
        image_size=args.image_size,
        rgb=bool(args.rgb),
        max_advance=args.max_advance,
        random_seed=args.seed
    )

    if args.image_training:
        if args.rgb:
            channels = 3
        else:
            channels = 1
        variant['policy_kwargs'] = dict(
            input_width=args.image_size,
            input_height=args.image_size,
            input_channels=channels,
            kernel_sizes=[3,3,3,3],
            n_channels=[32,32,32,32],
            strides=[2,2,2,2],
            paddings=[0,0,0,0],
            hidden_sizes=[256,256,256,256],
            init_w=1e-4
        )
        variant['path_collector_kwargs']['additional_keys'] = ['robot_observation']
        variant['replay_buffer_kwargs']['internal_keys'] = ['image','model_params','robot_observation']

    else:
        variant['path_collector_kwargs']['additional_keys'] = ['model_params']
        variant['replay_buffer_kwargs']['internal_keys'] = ['model_params']

    print("Args", args)
    if torch.cuda.is_available():
        print("Training with GPU")
        ptu.set_gpu_mode(True) 

    file_path = args.title + "-run-" + str(args.run)
    setup_logger(file_path, variant=variant)

    if bool(args.profile):
        cProfile.run('experiment(variant)', file_path +'-stats')
    else:
        trained_policy = experiment(variant)
        torch.save(trained_policy.state_dict(), file_path +'.mdl')
