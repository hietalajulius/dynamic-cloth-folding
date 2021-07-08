import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import KeyPathCollector, VectorizedKeyPathCollector
from rlkit.torch.sac import policies
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhScriptPolicy, CustomScriptPolicy, CustomTanhScriptPolicy, ScriptPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.her.cloth.her import ClothSacHERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.data_management.future_obs_dict_replay_buffer import FutureObsDictRelabelingBuffer
import gym
import mujoco_py
import torch
import cProfile
from rlkit.envs.wrappers import SubprocVecEnv
from gym.logger import set_level
from clothmanip.utils.utils import get_variant, argsparser, get_randomized_env, dump_commit_hashes, get_keys_and_dims, dump_goal
import copy
from clothmanip.utils import reward_calculation
import numpy as np
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
import os
from rlkit.core import logger
import json
from rlkit.samplers.eval_suite.success_rate_test import SuccessRateTest
from rlkit.samplers.eval_suite.blank_images_test import BlankImagesTest
from rlkit.samplers.eval_suite.real_corner_prediction_test import RealCornerPredictionTest
from rlkit.samplers.eval_suite.dump_real_images_test import DumpRealCornerPredictions

from rlkit.samplers.eval_suite.base import EvalTestSuite

import tracemalloc


torch.cuda.empty_cache()


set_level(50)

def experiment(variant):
    env = ClothEnv(**variant['env_kwargs'], has_viewer=True, save_folder=variant['save_folder'])
    env = NormalizedBoxEnv(env)
    eval_env = get_randomized_env(env, variant)

    constant_goal_variant = copy.deepcopy(variant)
    constant_goal_variant['env_kwargs']['constant_goal'] = True
    constant_goal_env = ClothEnv(**constant_goal_variant['env_kwargs'], has_viewer=True, save_folder=variant['save_folder'])
    goal = constant_goal_env.goal.copy()
    dump_goal(variant['save_folder'], goal)
    del constant_goal_env
    del constant_goal_variant


    keys, dims = get_keys_and_dims(variant, eval_env)
    image_training = variant['image_training']
    

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

    #TODO: combine own script policy with pretrained structs
    if image_training:
        if variant['pretrained_cnn']:
            policy = CustomTanhScriptPolicy(
                output_size=dims['action_dim'],
                added_fc_input_size=dims['added_fc_input_size'],
                aux_output_size=8,
                **variant['policy_kwargs'],
            )
        else:
            policy = TanhScriptPolicy(
                output_size=dims['action_dim'],
                added_fc_input_size=dims['added_fc_input_size'],
                aux_output_size=8,
                **variant['policy_kwargs'],
            )
          
            if variant['pretrained_vision_model_path'] is not None:
                loaded_model = torch.jit.load(variant['pretrained_vision_model_path'] )
                state_dict = loaded_model.state_dict()
                policy.load_state_dict(state_dict)
                print("Loaded pretrained model")
            else:
                print("Fresh model")
        

    else:
        policy = TanhGaussianPolicy(
            obs_dim=dims['policy_obs_dim'],
            action_dim=dims['action_dim'],
            hidden_sizes=[M, M],
            **variant['policy_kwargs']
        )

    eval_policy = MakeDeterministic(policy)

    real_corner_prediction_test = RealCornerPredictionTest(eval_env, eval_policy, 'real_corner_error', ['corner_error'], 1, variant=variant)
    #real_corners_dump = DumpRealCornerPredictions(eval_env, eval_policy, 'real_corner_dump', [], 1, variant=variant)
    wipe_success_rate_test = SuccessRateTest(eval_env, eval_policy, 'wipe', ['success_rate', 'corner_distance'], variant['num_eval_rollouts'] , variant=variant)
    #kitchen_success_rate_test = SuccessRateTest(eval_env, eval_policy, 'kitchen', ['success_rate', 'corner_distance'], variant['num_eval_rollouts'] , variant=variant)
    blank_images_test = BlankImagesTest(eval_env, eval_policy, 'blank', ['success_rate', 'corner_distance'], variant['num_eval_rollouts'] , variant=variant)

    if variant['use_eval_suite']:                                  
        eval_test_suite = EvalTestSuite([real_corner_prediction_test, blank_images_test, wipe_success_rate_test], variant['save_folder'])
    else:
        eval_test_suite = EvalTestSuite([], variant['save_folder'])

    def make_env():
        env = ClothEnv(**variant['env_kwargs'], save_folder=variant['save_folder'], has_viewer=variant['image_training'])
        env = NormalizedBoxEnv(env)
        env = get_randomized_env(env, variant)
        return env

    env_fns = [make_env for _ in range(variant['num_processes'])]
    vec_env = SubprocVecEnv(env_fns)

    
    expl_path_collector = VectorizedKeyPathCollector(
        vec_env,
        policy,
        processes=variant['num_processes'],
        observation_key=keys['path_collector_observation_key'],
        desired_goal_key=keys['desired_goal_key'],
        demo_paths=variant['demo_paths'],
        demo_divider=variant['env_kwargs']['output_max'],
        **variant['path_collector_kwargs'],
    )

    '''
    #FOR DEBUG, REMEMBER TO REMOVE try/except from domain rand wrapper

    expl_path_collector = KeyPathCollector(
        eval_env,
        policy,
        observation_key=keys['path_collector_observation_key'],
        desired_goal_key=keys['desired_goal_key'],
        **variant['path_collector_kwargs'],
    )
    '''


    task_reward_function = reward_calculation.get_task_reward_function(
        variant['env_kwargs']['constraint_distances'], 3, variant['env_kwargs']['sparse_dense'], variant['env_kwargs']['success_reward'], variant['env_kwargs']['fail_reward'], variant['env_kwargs']['extra_reward'])
    ob_spaces = copy.deepcopy(eval_env.observation_space.spaces)
    action_space = copy.deepcopy(eval_env.action_space)
    replay_buffer = FutureObsDictRelabelingBuffer(
        ob_spaces=ob_spaces,
        action_space=action_space,
        observation_key=keys['observation_key'],
        desired_goal_key=keys['desired_goal_key'],
        achieved_goal_key=keys['achieved_goal_key'],
        max_path_length=variant['algorithm_kwargs']['max_path_length'],
        **variant['replay_buffer_kwargs']
    )
    replay_buffer.set_task_reward_function(task_reward_function)

    policy_target_entropy = -np.prod(
        eval_env.action_space.shape).item()

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

    script_policy = None
    if image_training:
        if variant['pretrained_cnn']:
            script_policy = CustomScriptPolicy(
                output_size=dims['action_dim'],
                added_fc_input_size=dims['added_fc_input_size'],
                aux_output_size=8,
                **variant['policy_kwargs'],
            )
        else:
            script_policy = ScriptPolicy(
                        output_size=dims['action_dim'],
                        added_fc_input_size=dims['added_fc_input_size'],
                        aux_output_size=8,
                        **variant['policy_kwargs'],
                    )
        

    algorithm = TorchBatchRLAlgorithm(
        script_policy=script_policy,
        eval_suite=eval_test_suite,
        trainer=trainer,
        exploration_env=eval_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        num_demoers=variant['num_demoers'],
        num_pre_demoers=variant['num_processes'],
        num_pre_demos=variant['num_pre_demos'],
        replay_buffer=replay_buffer,
        save_folder=variant['save_folder'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)

    with mujoco_py.ignore_mujoco_warnings():
        algorithm.train()

    vec_env.close()
    print("Closed subprocesses")
    return


if __name__ == "__main__":
    args = argsparser()
    variant, arg_str = get_variant(args)

    if torch.cuda.is_available():
        print("Training with GPU")
        ptu.set_gpu_mode(True)
    else:
        print("Training with CPU")

    logger_path = args.title + "-run-" + str(args.run)
    setup_logger(logger_path, variant=variant,log_tabular_only=True)

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
