import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import KeyPathCollector, EvalKeyPathCollector, VectorizedKeyPathCollector, PresetEvalKeyPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhCNNGaussianPolicy, GaussianPolicy, GaussianCNNPolicy, LegacyTanhCNNGaussianPolicy, TanhScriptPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.her.cloth.her import ClothSacHERTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.data_management.future_obs_dict_replay_buffer import FutureObsDictRelabelingBuffer
from rlkit.torch.sac.policies import ScriptPolicy
import gym
import mujoco_py
import torch
import cProfile
from rlkit.envs.wrappers import SubprocVecEnv
from gym.logger import set_level
from clothmanip.utils.utils import get_variant, argsparser, get_randomized_env
import copy
from clothmanip.utils import reward_calculation
import numpy as np
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
import os
from rlkit.core import logger
import json


set_level(50)


def experiment(variant):
    env = ClothEnv(**variant['env_kwargs'], has_viewer=True, save_folder=variant['save_folder'], initial_xml_dump=True)
    env = NormalizedBoxEnv(env)
    
    if variant['domain_randomization']:
        eval_env = get_randomized_env(env, variant)
    else:
        eval_env = env

    #TODO these into utils
    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    action_dim = eval_env.action_space.low.size
    policy_obs_dim = obs_dim + goal_dim
    value_input_size = obs_dim + action_dim + goal_dim
    added_fc_input_size = goal_dim

    if 'model_params' in eval_env.observation_space.spaces:
        model_params_dim = eval_env.observation_space.spaces['model_params'].low.size
        value_input_size += model_params_dim

    if 'robot_observation' in eval_env.observation_space.spaces:
        robot_obs_dim = eval_env.observation_space.spaces['robot_observation'].low.size
        policy_obs_dim += robot_obs_dim
        value_input_size += robot_obs_dim
        added_fc_input_size += robot_obs_dim

    image_training = variant['image_training']
    if image_training:
        path_collector_observation_key = 'image'
    else:
        path_collector_observation_key = 'observation'

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    M = variant['layer_size']

    qf1 = ConcatMlp(
        input_size=value_input_size,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=value_input_size,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=value_input_size,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=value_input_size,
        output_size=1,
        hidden_sizes=[M, M],
    )

    if image_training:
        if variant['legacy_cnn']:
            policy = LegacyTanhCNNGaussianPolicy(
                output_size=action_dim,
                added_fc_input_size=added_fc_input_size,
                aux_output_size=8,
                **variant['policy_kwargs'],
            )
        else:
            policy = TanhScriptPolicy(
                output_size=action_dim,
                added_fc_input_size=added_fc_input_size,
                aux_output_size=8,
                **variant['policy_kwargs'],
            )

    else:
        policy = TanhGaussianPolicy(
            obs_dim=policy_obs_dim,
            action_dim=action_dim,
            hidden_sizes=[M, M],
            **variant['policy_kwargs']
        )

    eval_policy = MakeDeterministic(policy)

    steps_per_second = 1 / eval_env.timestep
    new_action_every_ctrl_step = steps_per_second / variant['env_kwargs']['control_frequency']

    eval_path_collector = EvalKeyPathCollector(
        eval_env,
        eval_policy,
        save_images_every_epoch=variant['save_images_every_epoch'],
        observation_key=path_collector_observation_key,
        desired_goal_key=desired_goal_key,
        save_folder=variant['save_folder'],
        env_timestep=eval_env.timestep,
        new_action_every_ctrl_step=new_action_every_ctrl_step,
        **variant['path_collector_kwargs']
    )

    if 'randomize_params' in variant['env_kwargs'].keys() and variant['env_kwargs']['randomize_params']:
        preset_eval_path_collector = PresetEvalKeyPathCollector(
            eval_env,
            eval_policy,
            observation_key=path_collector_observation_key,
            desired_goal_key=desired_goal_key,
            **variant['path_collector_kwargs']
        )
    else:
        preset_eval_path_collector = None

    demo_path_collector = None
    if variant['use_demos']:
        demo_path_collector = KeyPathCollector(
            eval_env,
            policy,
            use_demos=True,
            demo_coef=variant['demo_coef'],
            demo_path=variant['demo_path'],
            save_folder=variant['save_folder'],
            observation_key=path_collector_observation_key,
            desired_goal_key=desired_goal_key,
            **variant['path_collector_kwargs']
        )

    def make_env():
        env = ClothEnv(**variant['env_kwargs'], save_folder=variant['save_folder'], has_viewer=variant['image_training'])
        env = NormalizedBoxEnv(env)
        
        if variant['domain_randomization']:
            env = get_randomized_env(env, variant)
            
        
        return env

    env_fns = [make_env for _ in range(variant['num_processes'])]
    vec_env = SubprocVecEnv(env_fns)

    expl_path_collector = VectorizedKeyPathCollector(
        vec_env,
        policy,
        output_max=variant["env_kwargs"]["output_max"],
        demo_coef=variant['demo_coef'],
        processes=variant['num_processes'],
        observation_key=path_collector_observation_key,
        desired_goal_key=desired_goal_key,
        use_demos=variant['use_demos'],
        demo_path=variant['demo_path'],
        num_demoers=variant['num_demoers'],
        **variant['path_collector_kwargs'],
    )

    task_reward_function = reward_calculation.get_task_reward_function(
        variant['env_kwargs']['constraints'], 3, variant['env_kwargs']['sparse_dense'], variant['env_kwargs']['success_reward'], variant['env_kwargs']['fail_reward'], variant['env_kwargs']['extra_reward'])
    ob_spaces = copy.deepcopy(eval_env.observation_space.spaces)
    action_space = copy.deepcopy(eval_env.action_space)
    replay_buffer = FutureObsDictRelabelingBuffer(
        ob_spaces=ob_spaces,
        action_space=action_space,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
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
        script_policy = ScriptPolicy(
                    output_size=action_dim,
                    added_fc_input_size=added_fc_input_size,
                    aux_output_size=8,
                    **variant['policy_kwargs'],
                )

    algorithm = TorchBatchRLAlgorithm(
        script_policy=script_policy,
        trainer=trainer,
        exploration_env=eval_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        preset_evaluation_data_collector=preset_eval_path_collector,
        demo_data_collector=demo_path_collector,
        num_demos=variant['num_demos'],
        replay_buffer=replay_buffer,
        save_folder=variant['save_folder'],
        title=variant['version'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)

    with mujoco_py.ignore_mujoco_warnings():
        algorithm.train()

    torch.save(eval_policy.state_dict(), f"{variant['save_folder']}/policies/trained_policy.mdl")

    if variant['num_processes'] > 1:
        vec_env.close()
        print("Closed subprocesses")


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
        images_path = f"{variant['save_folder']}/images"
        eval_trajs_path = f"{variant['save_folder']}/eval_trajs"
        policies_path = f"{variant['save_folder']}/policies"
        os.makedirs(variant['save_folder'])
        os.makedirs(profiling_path)
        os.makedirs(images_path)
        os.makedirs(eval_trajs_path)
        os.makedirs(policies_path)

        with open(f"{variant['save_folder']}/params.json", "w") as outfile:
            json.dump(variant, outfile)
        with open(f"{variant['save_folder']}/command.txt", "w") as outfile:
            json.dump(arg_str, outfile)

    except OSError:
        print ("Creation of the directory %s failed" % variant['save_folder'])
    else:
        print ("Successfully created the directory %s" % variant['save_folder'])

    print("Profiling with cProfile")
    cProfile.run('experiment(variant)', f"{profiling_path}/profmain.prof")
