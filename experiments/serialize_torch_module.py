from rlkit.torch.sac.policies import ScriptPolicy, TanhScriptPolicy
import torch
from envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from utils import get_variant, argsparser

def main(variant):


    env = ClothEnv(**variant['env_kwargs'], has_viewer=True, eval_env=True, save_folder="/home/clothmanip/robotics/cloth-manipulation/experiments/model_serialization")
    env = NormalizedBoxEnv(env)
    eval_env = env

    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    action_dim = eval_env.action_space.low.size
    added_fc_input_size = 0# goal_dim

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

    sm = torch.jit.script(policy).cpu()
    torch.jit.save(sm, "testjit.pt")
    print("Maybe saved")

    tnhp = TanhScriptPolicy(output_size=action_dim,
                added_fc_input_size=added_fc_input_size,
                aux_output_size=8,
                **variant['policy_kwargs'],)


if __name__ == "__main__":
    args = argsparser()
    variant = get_variant(args)
    main(variant)