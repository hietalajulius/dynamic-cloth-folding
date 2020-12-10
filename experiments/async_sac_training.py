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
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm,TorchAsyncBatchRLAlgorithm
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
import gym
import mujoco_py
import argparse
import torch.nn as nn
import torch
import cProfile
from rlkit.envs.wrappers import SubprocVecEnv
from gym.logger import set_level
#import multiprocessing
from multiprocessing import set_start_method, Queue
import torch.multiprocessing as multiprocessing
#from torch.multiprocessing import set_start_method, Queue
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



start_method = "forkserver"
set_level(50)

import linecache
import os
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def argsparser():
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('--run',  default=1, type=int)
    parser.add_argument('--title', default="notitle", type=str)
    parser.add_argument('--train_steps', default=1000, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_cycles', default=100, type=int)
    parser.add_argument('--her_percent', default=0.0, type=float)
    parser.add_argument('--buffer_size', default=1E5, type=int)
    parser.add_argument('--max_path_length', default=50, type=int)
    parser.add_argument('--env_name', type=str, default="Cloth-v1")
    parser.add_argument('--strict', default=1, type=int)
    parser.add_argument('--image_training', default=1, type=int)
    parser.add_argument('--task', type=str, default="sideways")
    parser.add_argument('--distance_threshold', type=float, default=0.05)
    parser.add_argument('--eval_steps', type=int, default=0)
    parser.add_argument('--min_expl_steps', type=int, default=0)
    parser.add_argument('--randomize_params', type=int, default=0)
    parser.add_argument('--randomize_geoms', type=int, default=0)
    parser.add_argument('--uniform_jnt_tend', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=84)
    parser.add_argument('--rgb', type=int, default=0)
    parser.add_argument('--max_advance', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cprofile', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_processes', type=int, default=3)
    return parser.parse_args()

def collector(variant, batch_queue, policy_weights_queue, batch_processed_event, new_policy_event):
    tracemalloc.start()
    #with threadpool_limits(limits=1):
    print("PID of collection process", os.getpid())
    #print("Threadpool in collector:")
    #pprint(threadpool_info())
    def make_env():
            return NormalizedBoxEnv(gym.make(variant['env_name'], **variant['env_kwargs']))
    env_fns = [make_env for _ in range(variant['num_processes'])]
    vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    vec_env.seed(variant['env_kwargs']['random_seed'])
    image_training = variant['image_training']
    desired_goal_key = 'desired_goal'
    observation_key = 'observation'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    if image_training:
        path_collector_observation_key = 'image'
    else:
        path_collector_observation_key = 'observation'

    new_policy_event.wait()
    print("Creating sub collector")

    replay_buf_env = NormalizedBoxEnv(gym.make(variant['env_name'], **variant['env_kwargs']))

    replay_buffer = ObsDictRelabelingBuffer(
        env=replay_buf_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    print("Created buffer")
    obs_dim = replay_buf_env.observation_space.spaces['observation'].low.size
    robot_obs_dim = replay_buf_env.observation_space.spaces['robot_observation'].low.size
    model_params_dim = replay_buf_env.observation_space.spaces['model_params'].low.size
    action_dim = replay_buf_env.action_space.low.size
    goal_dim = replay_buf_env.observation_space.spaces['desired_goal'].low.size

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'

    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

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
    

    
    expl_path_collector = VectorizedKeyPathCollector(
        vec_env,
        policy,
        processes=variant['num_processes'],
        observation_key=path_collector_observation_key,
        desired_goal_key=desired_goal_key,
        **variant['path_collector_kwargs']
    )
    print("Created path collector")

    version = 0
    
    buffer_initted = False
    last_batch_put = time.time()
    procc = psutil.Process(os.getpid())
    cur_mem = procc.memory_info().rss / 1E6
    prev_mem_used = procc.memory_info().rss / 1E6
    memories = []
    prev_snapshot = tracemalloc.take_snapshot()
    while True:
        if time.time() - last_batch_put > 60 and int(time.time() - last_batch_put) % 60 == 0:
            print("Loop is hanging, events:", batch_queue.qsize(), batch_processed_event.is_set(), new_policy_event.is_set(), time.time() - last_batch_put)
            time.sleep(2)
        if new_policy_event.is_set():
            print("New policy availale, version:", version)
            state_dict = policy_weights_queue.get()
            local_state_dict = copy.deepcopy(state_dict)
            del state_dict
            new_policy_event.clear()
            policy.load_state_dict(local_state_dict)
            
            print("Collect with new policy")
            paths = expl_path_collector.collect_new_paths(
                    variant['algorithm_kwargs']['max_path_length'],
                    variant['algorithm_kwargs']['num_trains_per_train_loop'],
                    discard_incomplete_paths=False,
                )
            replay_buffer.add_paths(paths)
            print("Added paths paths, size now:", len(paths), replay_buffer._size)
            version += 1

            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.compare_to(prev_snapshot, 'traceback')

            print("[ Top 5 differences in collect ]")
            for stat in top_stats[:5]:
                print(stat)

            top_stats = snapshot.statistics('lineno')
            print("[ Top 5 absolute in collect]")
            for stat in top_stats[:5]:
                print(stat)

            prev_snapshot = snapshot

            procc = psutil.Process(os.getpid())
            cur_mem = procc.memory_info().rss / 1E6
            if cur_mem > prev_mem_used:
                print("Using more memory than previously in COLLECT", cur_mem-prev_mem_used, "MB \n")
            else:
                print("Using less memory than previosuly in COLLECT", prev_mem_used-cur_mem, "MB \n")
            memories.append(cur_mem)
            print("Memories", memories)
            prev_mem_used = cur_mem

        if not buffer_initted:
            batch = replay_buffer.random_batch(variant['algorithm_kwargs']['batch_size'])
            batch = np_to_pytorch_batch_explicit_device(batch, "cuda:0")
            batch_queue.put(batch)
            buffer_initted = True

        if batch_processed_event.is_set():
            batch = replay_buffer.random_batch(variant['algorithm_kwargs']['batch_size'])
            batch = np_to_pytorch_batch_explicit_device(batch, "cuda:0")
            batch_queue.put(batch)
            batch_processed_event.clear()
            last_batch_put = time.time()




            

def experiment(variant):
    tracemalloc.start()
    print("PID of main process", os.getpid())
    
    dim_env = NormalizedBoxEnv(gym.make(variant['env_name'], **variant['env_kwargs']))

    image_training = variant['image_training']

    obs_dim = dim_env.observation_space.spaces['observation'].low.size
    robot_obs_dim = dim_env.observation_space.spaces['robot_observation'].low.size
    model_params_dim = dim_env.observation_space.spaces['model_params'].low.size
    action_dim = dim_env.action_space.low.size
    goal_dim = dim_env.observation_space.spaces['desired_goal'].low.size

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
    policy_weights_queue = Queue()
    new_policy_event = multiprocessing.Event()
    batch_processed_event = multiprocessing.Event()
    process = multiprocessing.Process(target=collector, args=(variant,batch_queue,policy_weights_queue,batch_processed_event,new_policy_event))
    process.start()
    
    trainer = SACTrainer(
        env=dim_env, #Refactor 
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
    #print("Threads before train:")
    #pprint(threadpool_info())
    del dim_env

    with mujoco_py.ignore_mujoco_warnings():
        algorithm.train()

    process.join()
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
    variant['image_training'] = bool(args.image_training)
    variant['num_processes'] = int(args.num_processes)

    variant['algorithm_kwargs'] = dict(
        num_epochs=args.num_epochs,
        num_trains_per_train_loop = args.train_steps,
        num_expl_steps_per_train_loop = args.train_steps,
        num_train_loops_per_epoch = int(args.num_cycles),
        max_path_length = int(args.max_path_length),
        num_eval_steps_per_epoch=args.eval_steps,
        min_num_steps_before_training=args.min_expl_steps,
        batch_size=args.batch_size,
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

    if torch.cuda.is_available():
        print("Training with GPU")
        ptu.set_gpu_mode(True) 

    file_path = args.title + "-run-" + str(args.run)
    setup_logger(file_path, variant=variant)

    if bool(args.cprofile):
        #with threadpool_limits(limits=1):
        cProfile.run('experiment(variant)', file_path +'-stats')
    else:
        trained_policy = experiment(variant)
        torch.save(trained_policy.state_dict(), file_path +'.mdl')
