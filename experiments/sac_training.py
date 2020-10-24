from gym.envs.mujoco import HalfCheetahEnv

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

def argsparser():
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('--run',  default=1, type=int)
    parser.add_argument('--title', default="notitle", type=str)
    parser.add_argument('--train_steps', default=1000, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--her_percent', default=0.0, type=float)
    parser.add_argument('--buffer_size', default=1E6, type=int)
    parser.add_argument('--max_path_length', default=50, type=int)
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--strict', default=True, type=bool)
    parser.add_argument('--gpu', default=False, type=bool)
    parser.add_argument('--image_training', default=False, type=bool)
    parser.add_argument('--asymmetric', default=False, type=bool)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--n_actions', type=int, default=3)
    parser.add_argument('--learn_grasp', type=str, default=False)
    parser.add_argument('--distance_threshold', type=float, default=0.05)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--min_expl_steps', type=int, default=1000)
    parser.add_argument('--randomize_params', type=bool, default=False)
    parser.add_argument('--uniform_jnt_tend', type=bool, default=True)
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

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'

    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    M = variant['layer_size']

    qf1 = ConcatMlp(
    input_size=obs_dim + action_dim + model_params_dim + goal_dim,
    output_size=1,
    hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim + model_params_dim + goal_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim + model_params_dim + goal_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim + model_params_dim + goal_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
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
    eval_path_collector = KeyPathCollector(
        eval_env,
        eval_policy,
        render=True,
        render_kwargs={'mode': 'rgb_array'},
        observation_key=path_collector_observation_key,
        desired_goal_key=desired_goal_key,
        **variant['path_collector_kwargs']
    )
    expl_path_collector = KeyPathCollector(
        expl_env,
        policy,
        observation_key=path_collector_observation_key,
        desired_goal_key=desired_goal_key,
        **variant['path_collector_kwargs']
    )
    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    trainer = ClothSacHERTrainer(trainer)
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    with mujoco_py.ignore_mujoco_warnings():
        algorithm.train()

    return eval_policy




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
    variant['image_training'] = args.image_training
    variant['asymmetric'] = args.asymmetric

    variant['algorithm_kwargs'] = dict(
        num_epochs=args.num_epochs,
        num_trains_per_train_loop = args.train_steps,
        num_expl_steps_per_train_loop = args.train_steps,
        num_train_loops_per_epoch = int(10000 / args.train_steps),
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
        learn_grasp = args.learn_grasp,
        n_actions=args.n_actions,
        task=args.task,
        pixels=args.image_training,
        strict=args.strict,
        distance_threshold=args.distance_threshold,
        randomize_params=args.randomize_params,
        uniform_jnt_tend=args.uniform_jnt_tend
    )

    if args.image_training:
        variant['policy_kwargs'] = dict(
            input_width=84,
            input_height=84,
            input_channels=3,
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

    #if args.gpu:
    print("Args", args)
    print("Cuda avail",torch.cuda.is_available)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    print("GPU training", ptu)

    file_path = args.title + "-run-" + str(args.run)
    setup_logger(file_path, variant=variant)

    trained_policy = experiment(variant)
    torch.save(trained_policy.state_dict(), file_path +'.mdl')
