from utils import display_top, argsparser, get_variant
import linecache
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.data_collector import GoalConditionedPathCollector, KeyPathCollector, VectorizedKeyPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhCNNGaussianPolicy, MonsterTanhCNNGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.her.her import ClothSacHERTrainer
from rlkit.torch.networks import ConcatMlp, MergedCNN
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchAsyncBatchRLAlgorithm
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
import gym
import mujoco_py
import argparse
import torch.nn as nn
import torch
import cProfile
from rlkit.envs.wrappers import SubprocVecEnv
from gym.logger import set_level
from multiprocessing import set_start_method, Queue
import torch.multiprocessing as multiprocessing
import time
from rlkit.torch.core import np_to_pytorch_batch_explicit_device
import os
import psutil

from threadpoolctl import threadpool_info, threadpool_limits
from pprint import pprint
import logging
import sys
import copy
import logging
import sys
import tracemalloc
import numpy as np


start_method = "forkserver"
set_level(50)


def buffer(variant, batch_queue, path_queue, batch_processed_event, paths_available_event, keys, dims):
    process = psutil.Process(os.getpid())
    print("Buffer process PID", process)
    replay_buf_env = NormalizedBoxEnv(
        gym.make(variant['env_name'], **variant['env_kwargs']))

    replay_buffer = ObsDictRelabelingBuffer(
        env=replay_buf_env,  # Extract reward function
        observation_key=keys['observation_key'],
        desired_goal_key=keys['desired_goal_key'],
        achieved_goal_key=keys['achieved_goal_key'],
        **variant['replay_buffer_kwargs']
    )

    paths_initted = False
    buffer_initted = False

    while True:

        if paths_available_event.is_set() and not paths_initted:
            paths = path_queue.get()
            copied_paths = copy.deepcopy(paths)
            del paths
            replay_buffer.add_paths(copied_paths)
            paths_available_event.clear()
            paths_initted = True

        if paths_initted and not buffer_initted:
            batch = replay_buffer.random_batch(
                variant['algorithm_kwargs']['batch_size'])
            batch = np_to_pytorch_batch_explicit_device(batch, "cuda:0")
            batch_queue.put(batch)
            print("Initted buffer queue")
            buffer_initted = True

        if batch_processed_event.is_set():
            batch = replay_buffer.random_batch(
                variant['algorithm_kwargs']['batch_size'])
            batch = np_to_pytorch_batch_explicit_device(batch, "cuda:0")
            batch_queue.put(batch)
            batch_processed_event.clear()

            if paths_available_event.is_set():
                paths = path_queue.get()
                copied_paths = copy.deepcopy(paths)
                del paths
                replay_buffer.add_paths(copied_paths)
                paths_available_event.clear()
                paths_initted = True
                print("Added new paths to buffer, memory usage", process.memory_info().rss/10E9, "GB")
            

        

 
            


def collector(variant, path_queue, policy_weights_queue, paths_available_event, new_policy_event, keys, dims):
    process = psutil.Process(os.getpid())
    print("Collector process PID", process)
    def make_env():
        return NormalizedBoxEnv(gym.make(variant['env_name'], **variant['env_kwargs']))
    env_fns = [make_env for _ in range(variant['num_processes'])]
    vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    vec_env.seed(variant['env_kwargs']['random_seed'])
    image_training = variant['image_training']

    if image_training:
        path_collector_observation_key = 'image'
    else:
        path_collector_observation_key = keys['observation']

    new_policy_event.wait()

    M = variant['layer_size']

    if image_training:
        policy = TanhCNNGaussianPolicy(
            output_size=dims['action_dim'],
            added_fc_input_size=dims['robot_obs_dim'] + dims['goal_dim'],
            **variant['policy_kwargs'],
        )
    else:
        policy = TanhGaussianPolicy(
            obs_dim=dims['obs_dim'] +
            dims['model_params_dim'] + dims['goal_dim'],
            action_dim=dims['action_dim'],
            hidden_sizes=[M, M],
            **variant['policy_kwargs']
        )

    expl_path_collector = VectorizedKeyPathCollector(
        vec_env,
        policy,
        processes=variant['num_processes'],
        observation_key=path_collector_observation_key,
        desired_goal_key=keys['desired_goal_key'],
        **variant['path_collector_kwargs']
    )
    print("Created path collector")

    collected_items = 0
    while True:
        if new_policy_event.wait():
            state_dict = policy_weights_queue.get()
            local_state_dict = copy.deepcopy(state_dict)
            del state_dict

            policy.load_state_dict(local_state_dict)

            paths = expl_path_collector.collect_new_paths(
                variant['algorithm_kwargs']['max_path_length'],
                variant['algorithm_kwargs']['num_trains_per_train_loop'],
                discard_incomplete_paths=False,
            )
            path_queue.put(paths)
            paths_available_event.set()

            new_policy_event.clear()
            collected_items += variant['algorithm_kwargs']['num_trains_per_train_loop']
            print("Added new paths, collected steps so far:", collected_items)
            print("Memory usage in collector", process.memory_info().rss/10E9, "GB")


def experiment(variant):
    tracemalloc.start()
    print("PID of main process", os.getpid())

    dim_env = NormalizedBoxEnv(
        gym.make(variant['env_name'], **variant['env_kwargs']))

    image_training = variant['image_training']

    obs_dim = dim_env.observation_space.spaces['observation'].low.size
    robot_obs_dim = dim_env.observation_space.spaces['robot_observation'].low.size
    model_params_dim = dim_env.observation_space.spaces['model_params'].low.size
    action_dim = dim_env.action_space.low.size
    goal_dim = dim_env.observation_space.spaces['desired_goal'].low.size

    policy_target_entropy = -np.prod(
        dim_env.action_space.shape).item()

    dims = dict(obs_dim=obs_dim, robot_obs_dim=robot_obs_dim,
                model_params_dim=model_params_dim, action_dim=action_dim, goal_dim=goal_dim)
    del dim_env

    keys = dict(desired_goal_key='desired_goal',
                observation_key='observation', achieved_goal_key="achieved_goal")

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

    set_start_method(start_method)
    batch_queue = Queue()
    path_queue = Queue()
    policy_weights_queue = Queue()

    batch_processed_event = multiprocessing.Event()
    paths_available_event = multiprocessing.Event()
    new_policy_event = multiprocessing.Event()

    collector_process = multiprocessing.Process(target=collector, args=(
        variant, path_queue, policy_weights_queue, paths_available_event, new_policy_event, keys, dims))
    collector_process.start()

    replay_buffer_process = multiprocessing.Process(target=buffer, args=(
        variant, batch_queue, path_queue, batch_processed_event, paths_available_event, keys, dims))
    replay_buffer_process.start()

    trainer = SACTrainer(
        policy_target_entropy=policy_target_entropy,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    trainer = ClothSacHERTrainer(trainer)
    algorithm = TorchAsyncBatchRLAlgorithm(
        trainer=trainer,
        batch_queue=batch_queue,
        policy_weights_queue=policy_weights_queue,
        new_policy_event=new_policy_event,
        batch_processed_event=batch_processed_event,
        **variant['algorithm_kwargs']
    )

    algorithm.to(ptu.device)

    with mujoco_py.ignore_mujoco_warnings():
        algorithm.train()

    collector_process.join()
    replay_buffer_process.join()

    return eval_policy


if __name__ == "__main__":
    args = argsparser()
    variant = get_variant(args)

    if torch.cuda.is_available():
        print("Training with GPU")
        ptu.set_gpu_mode(True)
    else:
        print("Training with CPU")

    file_path = args.title + "-run-" + str(args.run)
    setup_logger(file_path, variant=variant)

    trained_policy = experiment(variant)
    torch.save(trained_policy.state_dict(), file_path + '.mdl')
