import rlkit.torch.pytorch_util as ptu
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
import copy
from gym.envs.robotics import reward_calculation
import numpy as np
from envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv

set_level(50)

DEFAULT_CAMERA_ARGS = {
    'camera_names': None,  # all cameras are randomized
    'randomize_position': True,
    'randomize_rotation': True,
    'randomize_fovy': True,
    'position_perturbation_size': 0.01,
    'rotation_perturbation_size': 0.087,
    'fovy_perturbation_size': 5.,
}


'''
def randomize_env(env):
    camera_randomization_args = DEFAULT_CAMERA_ARGS
    camera_randomization_args['camera_names'] = ['clothview2']
    return DomainRandomizationWrapper(
        env, randomize_on_reset=True,
        randomize_every_n_steps=0, custom_randomize_color=True, randomize_color=False,  camera_randomization_args=camera_randomization_args)
    if variant['domain_randomization']:
        macros.USING_INSTANCE_RANDOMIZATION = True
'''
def experiment(variant):
    env = ClothEnv(**variant['env_kwargs'], has_viewer=True)
    env = NormalizedBoxEnv(env)
    '''
    if variant['domain_randomization']:
        env = randomize_env(env)
    '''
    eval_env = env

    # TODO: Make sure inertials are in order everywhere

    with open("compiled_mujoco_model_no_inertias.xml", "w") as f:
        eval_env.sim.save(f, format='xml', keep_inertials=False)

    with open("compiled_mujoco_model_with_intertias.xml", "w") as f:
        eval_env.sim.save(f, format='xml', keep_inertials=True)

    print("Saved compiled xml mujoco models")

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
        policy = TanhCNNGaussianPolicy(
            output_size=action_dim,
            added_fc_input_size=added_fc_input_size,
            aux_output_size=12,
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

    # TODO: Most kwargs below a bit redundant
    eval_path_collector = EvalKeyPathCollector(
        eval_env,
        eval_policy,
        render=True,
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
            env = ClothEnv(**variant['env_kwargs'])
            env = NormalizedBoxEnv(env)
            '''
            if variant['domain_randomization']:
                env = randomize_env(env)
            '''
            return env

        env_fns = [make_env for _ in range(variant['num_processes'])]
        vec_env = SubprocVecEnv(env_fns)

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
            eval_env,
            policy,
            observation_key=path_collector_observation_key,
            desired_goal_key=desired_goal_key,
            **variant['path_collector_kwargs']
        )

    task_reward_function = reward_calculation.get_task_reward_function(
        variant['env_kwargs']['constraints'], 3, variant['env_kwargs']['sparse_dense'], variant['env_kwargs']['reward_offset'])
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

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=eval_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        preset_evaluation_data_collector=preset_eval_path_collector,
        replay_buffer=replay_buffer,
        title=variant['version'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)

    #with mujoco_py.ignore_mujoco_warnings():
    algorithm.train()

    torch.save(eval_policy.state_dict(), variant['version'] + '.mdl')

    if variant['num_processes'] > 1:
        vec_env.close()
        print("Closed subprocesses")


if __name__ == "__main__":
    args = argsparser()
    variant = get_variant(args)

    if torch.cuda.is_available():
        print("Training with GPU")
        ptu.set_gpu_mode(True)
    else:
        print("Training with CPU")

    file_path = args.title + "-run-" + str(args.run)

    setup_logger(file_path, variant=variant,
                 log_tabular_only=variant['log_tabular_only'])

    print("Profiling with cProfile")
    cProfile.run('experiment(variant)', "./profiling/profmain.prof")
