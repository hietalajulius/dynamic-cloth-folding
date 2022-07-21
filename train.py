import mujoco_py
import torch
import gym
from utils import general_utils, reward_calculation
import copy
import numpy as np
from env import cloth_env
from utils import task_definitions
import logging

from rlkit.torch import pytorch_util, networks, torch_rl_algorithm
from rlkit.torch.sac import policies as sac_policies, sac
from rlkit.torch.her.cloth import her
from rlkit.launchers import launcher_util
from rlkit.envs import wrappers

from rlkit.samplers.eval_suite import success_rate_test, eval_suite
from rlkit.samplers import data_collector
from rlkit.data_management import future_obs_dict_replay_buffer


torch.cuda.empty_cache()
gym.logger.set_level(50)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


def experiment(variant):

    eval_env = general_utils.get_randomized_env(wrappers.NormalizedBoxEnv(
        cloth_env.ClothEnv(**variant['env_kwargs'], has_viewer=True)), variant['env_kwargs']['randomization_kwargs'])

    env_keys, env_dims = general_utils.get_keys_and_dims(variant, eval_env)

    qf1 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[variant['fc_layer_size']
                      for _ in range(variant['fc_layer_depth'])],
    )
    qf2 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[variant['fc_layer_size']
                      for _ in range(variant['fc_layer_depth'])],
    )
    target_qf1 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[variant['fc_layer_size']
                      for _ in range(variant['fc_layer_depth'])],
    )
    target_qf2 = networks.ConcatMlp(
        input_size=env_dims['value_input_size'],
        output_size=1,
        hidden_sizes=[variant['fc_layer_size']
                      for _ in range(variant['fc_layer_depth'])],
    )

    policy = sac_policies.TanhScriptPolicy(
        output_size=env_dims['action_dim'],
        added_fc_input_size=env_dims['added_fc_input_size'],
        aux_output_size=9,
        **variant['policy_kwargs'],
    )

    eval_policy = sac_policies.MakeDeterministic(policy)

    evaluation_suite = eval_suite.EvalTestSuite(
        [success_rate_test.SuccessRateTest(eval_env, eval_policy, 'constant_cloth', ['success_rate', 'corner_distance',
                                                                                     'corner_0', 'corner_1', 'corner_2', 'corner_3', 'corner_sum_error'], variant['num_eval_rollouts'], keys=env_keys, variant=variant)], variant['save_folder'])

    def make_worker_env_function():
        return general_utils.get_randomized_env(wrappers.NormalizedBoxEnv(cloth_env.ClothEnv(**variant['env_kwargs'], has_viewer=True)), variant['env_kwargs']['randomization_kwargs'])

    env_functions = [make_worker_env_function for _ in range(
        variant['num_processes'])]
    vec_env = wrappers.SubprocVecEnv(env_functions)

    exploration_path_collector = data_collector.VectorizedKeyPathCollector(
        vec_env,
        policy,
        processes=variant['num_processes'],
        observation_key=env_keys['path_collector_observation_key'],
        desired_goal_key=env_keys['desired_goal_key'],
        demo_paths=variant['demo_paths'],
        demo_divider=variant['env_kwargs']['output_max'],
        **variant['path_collector_kwargs'],
    )

    # TODO: get rid of this stuff
    constraints = task_definitions.constraints["sideways"](
        0, 4, 8, args.success_distance)
    constraint_distances = [c['distance'] for c in constraints]

    reward_function = reward_calculation.get_task_reward_function(
        constraint_distances, 3, variant['env_kwargs']['sparse_dense'], variant['env_kwargs']['success_reward'], variant['env_kwargs']['fail_reward'], variant['env_kwargs']['extra_reward'])

    replay_buffer = future_obs_dict_replay_buffer.FutureObsDictRelabelingBuffer(
        ob_spaces=copy.deepcopy(eval_env.observation_space.spaces),
        action_space=copy.deepcopy(eval_env.action_space),
        observation_key=env_keys['observation_key'],
        desired_goal_key=env_keys['desired_goal_key'],
        achieved_goal_key=env_keys['achieved_goal_key'],
        **variant['replay_buffer_kwargs']
    )

    # TODO: pass as an arg
    replay_buffer.set_task_reward_function(reward_function)

    trainer = sac.SACTrainer(
        policy_target_entropy=-np.prod(
            eval_env.action_space.shape).item(),
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    trainer = her.ClothSacHERTrainer(trainer)

    # TODO: maybe create this in the algo when saving...
    script_policy = sac_policies.ScriptPolicy(
        output_size=env_dims['action_dim'],
        added_fc_input_size=env_dims['added_fc_input_size'],
        aux_output_size=9,
        **variant['policy_kwargs'],
    )

    # TODO: check that all passed params make sense
    algorithm = torch_rl_algorithm.TorchBatchRLAlgorithm(
        script_policy=script_policy,
        eval_suite=evaluation_suite,
        trainer=trainer,
        exploration_data_collector=exploration_path_collector,
        num_demoers=variant['num_demoers'],
        replay_buffer=replay_buffer,
        save_folder=variant['save_folder'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(pytorch_util.device)

    with mujoco_py.ignore_mujoco_warnings():
        algorithm.train()

    vec_env.close()
    logger.debug("Closed subprocesses")
    return


if __name__ == "__main__":
    args = general_utils.argsparser()
    variant = general_utils.get_variant(args)  # TODO: further clean up

    general_utils.setup_training_device()
    general_utils.setup_save_folder(variant)
    launcher_util.setup_logger(
        variant["title"], variant=variant, base_log_dir=variant["save_folder"])

    logger.debug('Training started')
    experiment(variant)
