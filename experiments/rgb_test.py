import numpy as np
from rlkit.envs.wrappers import NormalizedBoxEnv
from utils import argsparser, get_variant
import gym
from rlkit.samplers.data_collector import PresetEvalKeyPathCollector, EvalKeyPathCollector
from rlkit.torch.sac.policies import TanhCNNGaussianPolicy, MakeDeterministic
import torch


def experiment(variant):
    eval_env = NormalizedBoxEnv(
        gym.make(variant['env_name'], **variant['env_kwargs']))

    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    robot_obs_dim = eval_env.observation_space.spaces['robot_observation'].low.size
    model_params_dim = eval_env.observation_space.spaces['model_params'].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size

    image_training = variant['image_training']
    if image_training:
        path_collector_observation_key = 'image'
    else:
        path_collector_observation_key = 'observation'

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    policy = TanhCNNGaussianPolicy(
        output_size=action_dim,
        added_fc_input_size=robot_obs_dim + goal_dim,
        aux_output_size=12,
        **variant['policy_kwargs'],
    )

    #device = torch.device('cpu')

    policy.load_state_dict(torch.load(
        'current_policy.mdl'))

    eval_policy = MakeDeterministic(policy)

    eval_path_collector = EvalKeyPathCollector(
        eval_env,
        eval_policy,
        render=True,
        render_kwargs=dict(
            mode='rgb_array', image_capture=True, width=500, height=500),
        observation_key=path_collector_observation_key,
        desired_goal_key=desired_goal_key,
        **variant['path_collector_kwargs']
    )

    preset_eval_path_collector = PresetEvalKeyPathCollector(
        eval_env,
        eval_policy,
        render=True,
        render_kwargs=dict(
            mode='rgb_array', image_capture=True, width=500, height=500),
        observation_key=path_collector_observation_key,
        desired_goal_key=desired_goal_key,
        **variant['path_collector_kwargs']
    )

    eval_path_collector.collect_new_paths(
        100,
        1
    )


if __name__ == "__main__":
    args = argsparser()
    variant = get_variant(args)
    experiment(variant)
