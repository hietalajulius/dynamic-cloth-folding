import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import KeyPathCollector, EvalKeyPathCollector, VectorizedKeyPathCollector, PresetEvalKeyPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhCNNGaussianPolicy, GaussianPolicy, GaussianCNNPolicy
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
from utils import get_variant, argsparser
import numpy as np
import copy

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *


set_level(50)


def experiment(variant):
    if variant['env_type'] == 'robosuite':
        options = {}
        options["env_name"] = "Cloth"
        options["robots"] = "Panda"
        #controller_name = "OSC_POSE"
        controller_name = "IK_POSE"
        options["controller_configs"] = load_controller_config(
            default_controller=controller_name)
        options["controller_configs"]["interpolation"] = "linear"
        env = suite.make(
            **options,
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=10,
            constraints=variant['env_kwargs']['constraints']
        )

        eval_env = NormalizedBoxEnv(env)
        expl_env = NormalizedBoxEnv(env)
    else:
        eval_env = NormalizedBoxEnv(
            gym.make(variant['env_name'], **variant['env_kwargs']))
        expl_env = NormalizedBoxEnv(
            gym.make(variant['env_name'], **variant['env_kwargs']))

    obs_dim = expl_env.observation_space.spaces['observation'].low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    action_dim = eval_env.action_space.low.size
    policy_obs_dim = obs_dim + goal_dim
    value_input_size = obs_dim + action_dim + goal_dim
    added_fc_input_size = goal_dim

    if 'model_params' in expl_env.observation_space.spaces:
        model_params_dim = expl_env.observation_space.spaces['model_params'].low.size
        value_input_size += model_params_dim

    if 'robot_observation' in expl_env.observation_space.spaces:
        robot_obs_dim = expl_env.observation_space.spaces['robot_observation'].low.size
        policy_obs_dim += robot_obs_dim
        value_input_size += robot_obs_dim
        added_fc_input_size += robot_obs_dim

    image_training = variant['image_training']
    if image_training:
        path_collector_observation_key = 'image'
    else:
        path_collector_observation_key = 'observation'

    print("POL obs dim", policy_obs_dim)
    print("VAL INP SIZE", value_input_size)
    print("OBS DIMS", obs_dim, goal_dim, action_dim)

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

    use_tanh = True
    if image_training:
        if use_tanh:
            policy = TanhCNNGaussianPolicy(
                output_size=action_dim,
                added_fc_input_size=added_fc_input_size,
                aux_output_size=12,
                **variant['policy_kwargs'],
            )
        else:
            policy = GaussianCNNPolicy(
                output_size=action_dim,
                added_fc_input_size=added_fc_input_size,
                aux_output_size=12,
                max_log_std=0,
                min_log_std=-7,
                **variant['policy_kwargs'],
            )
    else:
        if use_tanh:
            policy = TanhGaussianPolicy(
                obs_dim=policy_obs_dim,
                action_dim=action_dim,
                hidden_sizes=[M, M],
                **variant['policy_kwargs']
            )
        else:
            policy = GaussianPolicy(
                obs_dim=policy_obs_dim,
                action_dim=action_dim,
                hidden_sizes=[M, M],
                max_log_std=0,
                min_log_std=-7,
                **variant['policy_kwargs']
            )

    eval_policy = MakeDeterministic(policy)

    # TODO: Most kwargs below a bit redundant
    eval_path_collector = EvalKeyPathCollector(
        eval_env,
        eval_policy,
        render=True,
        render_kwargs=dict(
            mode='rgb_array', image_capture=True, width=1000, height=1000),
        observation_key=path_collector_observation_key,
        desired_goal_key=desired_goal_key,
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

    if variant['num_processes'] > 1:
        print("Vectorized path collection")

        def make_env():
            return NormalizedBoxEnv(gym.make(variant['env_name'], **variant['env_kwargs']))

        def make_suite_env():
            options = {}
            options["env_name"] = "Cloth"
            options["robots"] = "Panda"
            #controller_name = "OSC_POSE"
            controller_name = "IK_POSE"
            options["controller_configs"] = load_controller_config(
                default_controller=controller_name)
            options["controller_configs"]["interpolation"] = "linear"
            env = suite.make(
                **options,
                has_renderer=False,
                has_offscreen_renderer=False,
                ignore_done=True,
                use_camera_obs=False,
                control_freq=10,
                constraints=variant['env_kwargs']['constraints']
            )
            return NormalizedBoxEnv(env)

        if variant['env_type'] == 'robosuite':
            env_fns = [make_suite_env for _ in range(variant['num_processes'])]
        else:
            env_fns = [make_env for _ in range(variant['num_processes'])]
        vec_env = SubprocVecEnv(env_fns)
        vec_env.seed(variant['random_seed'])

        expl_path_collector = VectorizedKeyPathCollector(
            vec_env,
            policy,
            processes=variant['num_processes'],
            observation_key=path_collector_observation_key,
            desired_goal_key=desired_goal_key,
            **variant['path_collector_kwargs']
        )
    else:
        print("Single env path collection")
        expl_path_collector = KeyPathCollector(
            expl_env,
            policy,
            observation_key=path_collector_observation_key,
            desired_goal_key=desired_goal_key,
            **variant['path_collector_kwargs']
        )

    reward_function = copy.deepcopy(eval_env.reward_function)
    ob_spaces = copy.deepcopy(eval_env.observation_space.spaces)
    action_space = copy.deepcopy(eval_env.action_space)
    replay_buffer = FutureObsDictRelabelingBuffer(
        ob_spaces=ob_spaces,
        action_space=action_space,
        observation_key=observation_key,  # Image key passed in additional keys
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    replay_buffer.set_reward_function(reward_function)

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

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        preset_evaluation_data_collector=preset_eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)

    with mujoco_py.ignore_mujoco_warnings():
        algorithm.train()

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

    if bool(args.cprofile):
        print("Profiling with cProfile")
        cProfile.run('experiment(variant)', file_path + '-stats')
    else:
        trained_policy = experiment(variant)
        torch.save(trained_policy.state_dict(), file_path + '.mdl')
