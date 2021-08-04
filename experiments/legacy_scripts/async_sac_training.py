import rlkit.torch.pytorch_util as ptu
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.her.cloth.her import ClothSacHERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchAsyncBatchRLAlgorithm

from clothmanip.asynchronous.collector import collector
from clothmanip.asynchronous.buffer import buffer

import gym
import mujoco_py
import torch
from rlkit.envs.wrappers import SubprocVecEnv
from gym.logger import set_level

from torch.multiprocessing import set_start_method, Queue, Value
import torch.multiprocessing as multiprocessing
import tracemalloc
import numpy as np
from clothmanip.utils.utils import get_variant, argsparser, get_randomized_env, dump_commit_hashes, get_keys_and_dims
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhScriptPolicy, CustomScriptPolicy, CustomTanhScriptPolicy, ScriptPolicy

from rlkit.samplers.eval_suite.success_rate_test import SuccessRateTest
from rlkit.samplers.eval_suite.blank_images_test import BlankImagesTest
from rlkit.samplers.eval_suite.real_corner_prediction_test import RealCornerPredictionTest

from rlkit.samplers.eval_suite.base import EvalTestSuite
import os
import cProfile
import json
from rlkit.core import logger


START_METHOD = "forkserver"
set_level(50)

def experiment(variant):
    tracemalloc.start()


    env = ClothEnv(**variant['env_kwargs'], has_viewer=True, save_folder=variant['save_folder'])
    env = NormalizedBoxEnv(env)
    eval_env = get_randomized_env(env, variant)

    keys, dims = get_keys_and_dims(variant, eval_env)

    policy_target_entropy = -np.prod(
        eval_env.action_space.shape).item()

    M = variant['layer_size']

    qf1 = ConcatMlp(
        input_size=dims['value_input_size'],
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=dims['value_input_size'],
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=dims['value_input_size'],
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=dims['value_input_size'],
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhScriptPolicy(
        output_size=dims['action_dim'],
        added_fc_input_size=dims['added_fc_input_size'],
        aux_output_size=8,
        **variant['policy_kwargs'],
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
        variant, path_queue, policy_weights_queue, paths_available_event, new_policy_event, keys, dims, num_collected_steps, collector_memory_usage, env_memory_usages))
    collector_process.start()

    replay_buffer_process = multiprocessing.Process(target=buffer, args=(
        variant, batch_queue, path_queue, batch_processed_event, paths_available_event, keys, ptu.device, buffer_memory_usage))
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

    eval_policy = MakeDeterministic(policy)

    real_corner_prediction_test = RealCornerPredictionTest(eval_env, eval_policy, 'real_corner_error', ['corner_error'], 1, variant=variant)
    success_rate_test = SuccessRateTest(eval_env, eval_policy, 'regular', ['success_rate', 'corner_distance'], variant['num_eval_rollouts'] , variant=variant)
    blank_images_test = BlankImagesTest(eval_env, eval_policy, 'blank', ['success_rate', 'corner_distance'], variant['num_eval_rollouts'] , variant=variant)
                                        
    eval_test_suite = EvalTestSuite([real_corner_prediction_test, success_rate_test, blank_images_test], variant['save_folder'])

    script_policy = ScriptPolicy(
                        output_size=dims['action_dim'],
                        added_fc_input_size=dims['added_fc_input_size'],
                        aux_output_size=8,
                        **variant['policy_kwargs'],
                    )

    algorithm = TorchAsyncBatchRLAlgorithm(
        script_policy=script_policy,
        eval_suite=eval_test_suite,
        trainer=trainer,
        num_demos=variant['num_demos'],
        save_folder=variant['save_folder'],
        num_collected_steps=num_collected_steps,
        buffer_memory_usage=buffer_memory_usage,
        collector_memory_usage=collector_memory_usage,
        env_memory_usages=env_memory_usages,
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
    variant, arg_str = get_variant(args)

    if torch.cuda.is_available():
        print("Training with GPU")
        ptu.set_gpu_mode(True)
    else:
        print("Training with CPU")

    logger_path = args.title + "-run-" + str(args.run)
    setup_logger(logger_path, variant=variant,
                 log_tabular_only=variant['log_tabular_only'])

    variant['save_folder'] = f"./trainings/{logger._prefixes[0]}".strip()

    try:
        profiling_path = f"{variant['save_folder']}/profiling"
        os.makedirs(variant['save_folder'])
        os.makedirs(profiling_path)

        with open(f"{variant['save_folder']}/params.json", "w") as outfile:
            json.dump(variant, outfile)
        with open(f"{variant['save_folder']}/command.txt", "w") as outfile:
            json.dump(arg_str, outfile)

        dump_commit_hashes(variant['save_folder'])

    except OSError:
        print ("Creation of the directory %s failed" % variant['save_folder'])
    else:
        print ("Successfully created the directory %s" % variant['save_folder'])

    print("Profiling with cProfile")
    cProfile.run('experiment(variant)', f"{profiling_path}/profmain.prof")

