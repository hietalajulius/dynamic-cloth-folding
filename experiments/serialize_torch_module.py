from rlkit.torch.sac.policies import ScriptPolicy, TanhScriptPolicy
import torch
from envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from utils import get_variant, argsparser

def main(variant):


    env = ClothEnv(**variant['env_kwargs'], has_viewer=True, eval_env=True, save_folder="/home/clothmanip/robotics/cloth-manipulation/model_serialization")
    env = NormalizedBoxEnv(env)
    eval_env = env

    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    action_dim = eval_env.action_space.low.size
    policy_obs_dim = obs_dim + goal_dim
    value_input_size = obs_dim + action_dim + goal_dim
    added_fc_input_size = goal_dim

    eval_env.reset()

    print("example goal", eval_env.goal)

    if 'model_params' in eval_env.observation_space.spaces:
        model_params_dim = eval_env.observation_space.spaces['model_params'].low.size
        value_input_size += model_params_dim

    if 'robot_observation' in eval_env.observation_space.spaces:
        robot_obs_dim = eval_env.observation_space.spaces['robot_observation'].low.size
        policy_obs_dim += robot_obs_dim
        value_input_size += robot_obs_dim
        added_fc_input_size += robot_obs_dim

    '''
    policy = TanhCNNGaussianPolicy(
                output_size=action_dim,
                added_fc_input_size=added_fc_input_size,
                aux_output_size=8,
                **variant['policy_kwargs'],
            )

    '''

    policy = ScriptPolicy(
                output_size=action_dim,
                added_fc_input_size=added_fc_input_size,
                aux_output_size=8,
                **variant['policy_kwargs'],
            )

    model_path = "/home/clothmanip/robotics/cloth-manipulation/current_policy"
    policy.load_state_dict(torch.load(model_path+".mdl"))
    policy.eval()

    sm = torch.jit.script(policy).cpu()
    torch.jit.save(sm, model_path+".pt")
    print("Maybe saved")

    tnhp = TanhScriptPolicy(output_size=action_dim,
                added_fc_input_size=added_fc_input_size,
                aux_output_size=8,
                **variant['policy_kwargs'],)


if __name__ == "__main__":
    args = argsparser()
    variant = get_variant(args)
    main(variant)
