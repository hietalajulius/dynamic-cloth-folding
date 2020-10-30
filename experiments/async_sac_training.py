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
from rlkit.torch.torch_rl_algorithm import TorchAsyncBatchRLAlgorithm, TorchBatchRLAlgorithm
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
import gym
import mujoco_py
import argparse
import torch.nn as nn
import torch
import cProfile


def experiment():
    env_kwargs = dict(
            task="sideways",
            pixels=False,
            strict=True,
            distance_threshold=0.05,
            randomize_params=False,
            randomize_geoms=False,
            uniform_jnt_tend=True,
            max_advance=0.05,
            random_seed=1
        )
    path_collector_kwargs = dict(additional_keys = ['model_params'])
    replay_buffer_kwargs = dict(
        internal_keys = ['model_params'],
        max_size=int(1E6),
        fraction_goals_env_goals = 0,
        fraction_goals_rollout_goals = 0.2
        )
    algorithm_kwargs = dict(
        num_epochs=1,
        num_trains_per_train_loop = 100,
        num_expl_steps_per_train_loop = 100,
        num_train_loops_per_epoch = 100,
        max_path_length = 50,
        num_eval_steps_per_epoch=0,
        min_num_steps_before_training=0,
        batch_size=256,
    )
    eval_env = NormalizedBoxEnv(gym.make("Cloth-v1", **env_kwargs))
    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    model_params_dim = eval_env.observation_space.spaces['model_params'].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size

    desired_goal_key = 'desired_goal'


    M = 256

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

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + model_params_dim + goal_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M]
    )

    trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **trainer_kwargs
    )
    trainer = ClothSacHERTrainer(trainer)

    comparison = True

    if comparison:
        
        eval_env = NormalizedBoxEnv(gym.make("Cloth-v1", **env_kwargs))
        expl_env = NormalizedBoxEnv(gym.make("Cloth-v1", **env_kwargs))
        path_collector_observation_key = 'observation'
        eval_policy = MakeDeterministic(policy)
        observation_key = 'observation'
        desired_goal_key = 'desired_goal'
        achieved_goal_key = desired_goal_key.replace("desired", "achieved")
        
        replay_buffer = ObsDictRelabelingBuffer(
            env=eval_env,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            achieved_goal_key=achieved_goal_key,
            **replay_buffer_kwargs
        )
        eval_path_collector = KeyPathCollector(
            eval_env,
            eval_policy,
            render=True,
            render_kwargs={'mode': 'rgb_array'},
            observation_key=path_collector_observation_key,
            desired_goal_key=desired_goal_key,
            **path_collector_kwargs
        )
        expl_path_collector = KeyPathCollector(
            expl_env,
            policy,
            observation_key=path_collector_observation_key,
            desired_goal_key=desired_goal_key,
            **path_collector_kwargs
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **algorithm_kwargs
        )

    else:
        algorithm = TorchAsyncBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=None,
            evaluation_env=None,
            exploration_data_collector=None,
            evaluation_data_collector=None,
            replay_buffer=None,
            batch_size=256,
            max_path_length=50,
            num_epochs=1,
            num_eval_steps_per_epoch=0,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=100
        )

    algorithm.to(ptu.device)
    with mujoco_py.ignore_mujoco_warnings():
        algorithm.train()





if __name__ == "__main__":
    variant = dict()
    setup_logger("test", variant=variant)
    cProfile.run('experiment()', 'async-stats')

