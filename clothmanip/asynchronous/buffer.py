from rlkit.torch.core import np_to_pytorch_batch_explicit_device
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.data_management.future_obs_dict_replay_buffer import FutureObsDictRelabelingBuffer
from clothmanip.utils import reward_calculation
import copy
from rlkit.envs.wrappers import NormalizedBoxEnv
import psutil
import os
import time




def buffer(variant, batch_queue, path_queue, batch_processed_event, paths_available_event, keys, device, buffer_memory_usage):
    process = psutil.Process(os.getpid())
    env = ClothEnv(**variant['env_kwargs'], has_viewer=True, save_folder=variant['save_folder'])
    env = NormalizedBoxEnv(env)
    task_reward_function = reward_calculation.get_task_reward_function(
        variant['env_kwargs']['constraints'], 3, variant['env_kwargs']['sparse_dense'], variant['env_kwargs']['success_reward'], variant['env_kwargs']['fail_reward'], variant['env_kwargs']['extra_reward'])
    ob_spaces = copy.deepcopy(env.observation_space.spaces)
    action_space = copy.deepcopy(env.action_space)
    del env
    replay_buffer = FutureObsDictRelabelingBuffer(
        ob_spaces=ob_spaces,
        action_space=action_space,
        observation_key=keys['observation_key'],
        desired_goal_key=keys['desired_goal_key'],
        achieved_goal_key=keys['achieved_goal_key'],
        **variant['replay_buffer_kwargs']
    )
    replay_buffer.set_task_reward_function(task_reward_function)

    print("Buffer process waiting for paths")
    paths_available_event.wait()
    paths = path_queue.get()
    copied_paths = copy.deepcopy(paths)
    del paths
    replay_buffer.add_paths(copied_paths)
    paths_available_event.clear()
    batch = replay_buffer.random_batch(
        variant['algorithm_kwargs']['batch_size'])

    print("I have a batch", batch.keys(), batch["images"].shape)
    batch = np_to_pytorch_batch_explicit_device(batch, device)
    batch_queue.put(batch)

    while True:
        if batch_processed_event.is_set():
            print("Buffer: batch processed received")
            batch = replay_buffer.random_batch(
                variant['algorithm_kwargs']['batch_size'])
            batch = np_to_pytorch_batch_explicit_device(batch, device)
            batch_queue.put(batch)
            print("Buffer: put new batch to queue")
            batch_processed_event.clear()

            if paths_available_event.is_set():
                paths = path_queue.get()
                copied_paths = copy.deepcopy(paths)
                del paths
                replay_buffer.add_paths(copied_paths)
                paths_available_event.clear()
                
                buffer_memory_usage.value = process.memory_info().rss/10E9