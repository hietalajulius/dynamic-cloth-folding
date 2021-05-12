from rlkit.torch.sac.policies import ScriptPolicy, TanhScriptPolicy
import torch
from envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from utils import get_variant
import argparse
import json

def main(variant, input_folder, output_folder):


    env = ClothEnv(**variant['env_kwargs'], has_viewer=True, initial_xml_dump=True, save_folder=output_folder)
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

    policy = ScriptPolicy(
                output_size=action_dim,
                added_fc_input_size=added_fc_input_size,
                aux_output_size=8,
                **variant['policy_kwargs'],
            )

    model_path = input_folder + "/current_policy"
    policy.load_state_dict(torch.load(model_path+".mdl", map_location='cpu'))
    policy.eval()

    sm = torch.jit.script(policy).cpu()
    torch.jit.save(sm, model_path+".pt")

    tnhp = TanhScriptPolicy(output_size=action_dim,
                added_fc_input_size=added_fc_input_size,
                aux_output_size=8,
                **variant['policy_kwargs'],)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_folder', type=str)

    args = parser.parse_args()


    with open(f"{args.input_folder}/params.json")as json_file:
        variant = json.load(json_file)
    
    main(variant, args.input_folder, args.output_folder)
