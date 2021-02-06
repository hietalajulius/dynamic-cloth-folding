from utils import argsparser, get_variant
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import VectorizedKeyPathCollector, KeyPathCollector, EvalKeyPathCollector, PresetEvalKeyPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhCNNGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.her.cloth.her import ClothSacHERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchAsyncBatchRLAlgorithm
from rlkit.data_management.future_obs_dict_replay_buffer import FutureObsDictRelabelingBuffer
import gym
import mujoco_py
import torch
from rlkit.envs.wrappers import SubprocVecEnv
from gym.logger import set_level

from torch.multiprocessing import set_start_method, Queue, Value
import torch.multiprocessing as multiprocessing
from rlkit.torch.core import np_to_pytorch_batch_explicit_device
import os
import psutil
import copy
import tracemalloc
import numpy as np
import pickle


START_METHOD = "forkserver"
set_level(50)


def buffer(variant, batch_queue, path_queue, batch_processed_event, paths_available_event, keys, device, buffer_memory_usage):
    process = psutil.Process(os.getpid())
    print("Buffer process PID", process)
    replay_buf_env = NormalizedBoxEnv(
        gym.make(variant['env_name'], **variant['env_kwargs']))
    reward_function = copy.deepcopy(replay_buf_env.reward_function)
    ob_spaces = copy.deepcopy(replay_buf_env.observation_space.spaces)
    action_space = copy.deepcopy(replay_buf_env.action_space)
    del replay_buf_env  # TODO: Pass spaces and reward function from main process
    # TODO: Parametrize this
    load_existing_buffer = False
    if load_existing_buffer:
        with open('async_policy/buffer_data.pkl', 'rb') as inp:
            replay_buffer = pickle.load(inp)
            print("LOADED BUFFER", replay_buffer._size)
    else:
        replay_buffer = FutureObsDictRelabelingBuffer(
            ob_spaces=ob_spaces,
            action_space=action_space,
            observation_key=keys['observation_key'],
            desired_goal_key=keys['desired_goal_key'],
            achieved_goal_key=keys['achieved_goal_key'],
            **variant['replay_buffer_kwargs']
        )
    replay_buffer.set_reward_function(reward_function)

    paths_available_event.wait()
    paths = path_queue.get()
    print("GOT INITIAL PATHS")
    copied_paths = copy.deepcopy(paths)
    del paths
    replay_buffer.add_paths(copied_paths)
    paths_available_event.clear()
    batch = replay_buffer.random_batch(
        variant['algorithm_kwargs']['batch_size'])
    batch = np_to_pytorch_batch_explicit_device(batch, device)
    batch_queue.put(batch)
    print("INSERTED INITIAL BATCH")

    # TODO: Parametrize this and sync with policy saves
    save_every_addition = 100
    additions = 0
    while True:
        if batch_processed_event.is_set():
            batch = replay_buffer.random_batch(
                variant['algorithm_kwargs']['batch_size'])
            batch = np_to_pytorch_batch_explicit_device(batch, device)
            batch_queue.put(batch)
            batch_processed_event.clear()

            if paths_available_event.is_set():
                paths = path_queue.get()
                copied_paths = copy.deepcopy(paths)
                del paths
                replay_buffer.add_paths(copied_paths)
                paths_available_event.clear()
                if additions % save_every_addition == 0:
                    replay_buffer.set_reward_function(None)
                    with open('async_policy/buffer_data.pkl', 'wb') as outp:
                        pickle.dump(replay_buffer, outp,
                                    pickle.HIGHEST_PROTOCOL)
                        print("DUMPED buffer")
                    replay_buffer.set_reward_function(reward_function)
                additions += 1

                buffer_memory_usage.value = process.memory_info().rss/10E9


def collector(variant, path_queue, policy_weights_queue, paths_available_event, new_policy_event, keys, dims, num_collected_steps, collector_memory_usage, env_memory_usages):
    process = psutil.Process(os.getpid())
    print("Collector process PID", process)

    def make_env():
        return NormalizedBoxEnv(gym.make(variant['env_name'], **variant['env_kwargs']))
    env_fns = [make_env for _ in range(variant['num_processes'])]
    vec_env = SubprocVecEnv(env_fns, env_memory_usages,
                            start_method=START_METHOD)
    vec_env.seed(variant['env_kwargs']['random_seed'])
    image_training = variant['image_training']

    path_collector_observation_key = keys['observation_key']

    new_policy_event.wait()

    M = variant['layer_size']

    if image_training:
        policy = TanhCNNGaussianPolicy(
            output_size=dims['action_dim'],
            added_fc_input_size=dims['robot_obs_dim'] + dims['goal_dim'],
            aux_output_size=12,
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

    steps_per_rollout = variant['algorithm_kwargs']['max_path_length'] * \
        variant['num_processes']
    while True:
        if new_policy_event.wait():
            state_dict = policy_weights_queue.get()
            local_state_dict = copy.deepcopy(state_dict)
            del state_dict

            policy.load_state_dict(local_state_dict)

            collector_memory_usage.value = process.memory_info().rss/10E9

        # Keep collecting paths even without new policy
        paths = expl_path_collector.collect_new_paths(
            variant['algorithm_kwargs']['max_path_length'],
            steps_per_rollout,
            discard_incomplete_paths=False,
        )
        path_queue.put(paths)
        paths_available_event.set()

        new_policy_event.clear()

        num_collected_steps.value = expl_path_collector._num_steps_total

        # TODO: Use for memory debug
        # print("Added new paths, collected steps so far:", collected_items)
        # print("Memory usage in collector",process.memory_info().rss/10E9, "GB")


def experiment(variant):
    tracemalloc.start()
    print("PID of main process", os.getpid())

    eval_env = NormalizedBoxEnv(
        gym.make(variant['env_name'], **variant['env_kwargs']))

    image_training = variant['image_training']

    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    robot_obs_dim = eval_env.observation_space.spaces['robot_observation'].low.size
    model_params_dim = eval_env.observation_space.spaces['model_params'].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size

    policy_target_entropy = -np.prod(
        eval_env.action_space.shape).item()

    # Image training does not require obs dims
    dims = dict(obs_dim=obs_dim, robot_obs_dim=robot_obs_dim,
                model_params_dim=model_params_dim, action_dim=action_dim, goal_dim=goal_dim)

    if image_training:
        path_collector_observation_key = 'image'
    else:
        path_collector_observation_key = 'observation'

    collector_keys = dict(desired_goal_key='desired_goal',
                          observation_key=path_collector_observation_key, achieved_goal_key="achieved_goal")

    buffer_keys = dict(desired_goal_key='desired_goal',
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
            aux_output_size=12,
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

    set_start_method(START_METHOD)
    batch_queue = Queue()
    path_queue = Queue()
    policy_weights_queue = Queue()

    batch_processed_event = multiprocessing.Event()
    paths_available_event = multiprocessing.Event()
    new_policy_event = multiprocessing.Event()

    num_collected_steps = Value('d', 0.0)

    collector_memory_usage = Value('d', 0.0)
    buffer_memory_usage = Value('d', 0.0)

    env_memory_usages = [Value('d', 0.0)
                         for _ in range(variant['num_processes'])]

    collector_process = multiprocessing.Process(target=collector, args=(
        variant, path_queue, policy_weights_queue, paths_available_event, new_policy_event, collector_keys, dims, num_collected_steps, collector_memory_usage, env_memory_usages))
    collector_process.start()

    replay_buffer_process = multiprocessing.Process(target=buffer, args=(
        variant, batch_queue, path_queue, batch_processed_event, paths_available_event, buffer_keys, ptu.device, buffer_memory_usage))
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

    load_existing = False  # TODO: parametrize
    if load_existing:
        policy.load_state_dict(torch.load('async_policy/current_policy.mdl'))
        trainer._base_trainer.alpha_optimizer.load_state_dict(
            torch.load('async_policy/current_alpha_optimizer.mdl'))
        trainer._base_trainer.policy_optimizer.load_state_dict(
            torch.load('async_policy/current_policy_optimizer.mdl'))
        trainer._base_trainer.qf1_optimizer.load_state_dict(
            torch.load('async_policy/current_qf1_optimizer.mdl'))
        trainer._base_trainer.qf2_optimizer.load_state_dict(
            torch.load('async_policy/current_qf2_optimizer.mdl'))
        print("LOADED EXISTING POLICY")

    eval_policy = MakeDeterministic(policy)

    # TODO: Most kwargs below a bit redundant
    eval_path_collector = EvalKeyPathCollector(
        eval_env,
        eval_policy,
        render=True,
        render_kwargs=dict(
            mode='rgb_array', image_capture=True, width=500, height=500),
        observation_key=collector_keys['observation_key'],
        desired_goal_key=collector_keys['desired_goal_key'],
        **variant['path_collector_kwargs']
    )

    preset_eval_path_collector = PresetEvalKeyPathCollector(
        eval_env,
        eval_policy,
        observation_key=collector_keys['observation_key'],
        desired_goal_key=collector_keys['desired_goal_key'],
        **variant['path_collector_kwargs']
    )

    algorithm = TorchAsyncBatchRLAlgorithm(
        trainer=trainer,
        batch_queue=batch_queue,
        policy_weights_queue=policy_weights_queue,
        new_policy_event=new_policy_event,
        batch_processed_event=batch_processed_event,
        evaluation_data_collector=eval_path_collector,
        preset_evaluation_data_collector=preset_eval_path_collector,
        num_collected_steps=num_collected_steps,
        buffer_memory_usage=buffer_memory_usage,
        collector_memory_usage=collector_memory_usage,
        env_memory_usages=env_memory_usages,
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
