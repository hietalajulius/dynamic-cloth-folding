from clothmanip.utils.utils import get_variant, argsparser, get_randomized_env, dump_commit_hashes, get_keys_and_dims, dump_goal
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhScriptPolicy, CustomScriptPolicy, CustomTanhScriptPolicy, ScriptPolicy
import copy
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
import torch
import numpy as np
import cv2
import os
import shutil
import pandas as pd


def obs_processor(o, obs_key, additional_keys):
    obs = o[obs_key]
    for additional_key in additional_keys:
        obs = np.hstack((obs, o[additional_key]))

    return np.hstack((obs, o['desired_goal']))


def evaluate(variant):
    foddl = "/home/julius/robotics/real_runs/thue-4cm-fullstate-run-0-646/policy_eval_runs_rand"
    eval_env_variant = copy.deepcopy(variant)
    eval_env_variant['env_kwargs']['randomization_kwargs']['dynamics_randomization'] = True
    eval_env_variant['env_kwargs']['constant_goal'] = True
    eval_env = ClothEnv(**eval_env_variant['env_kwargs'], has_viewer=True, save_folder=foddl)
    eval_env = NormalizedBoxEnv(eval_env)
    eval_env = get_randomized_env(eval_env, eval_env_variant)
    #goal = eval_env.goal.copy()
    #dump_goal(variant['save_folder'], goal)


    keys, dims = get_keys_and_dims(variant, eval_env)

    if variant["image_training"]:
        obs_key = 'image'
        additional_keys = ['robot_observation']
        policy = TanhScriptPolicy(
                output_size=dims['action_dim'],
                added_fc_input_size=dims['added_fc_input_size'],
                aux_output_size=9,
                **variant['policy_kwargs'],
            )

        loaded_model = torch.jit.load("/home/julius/robotics/real_runs/emo-4cm-fs1-single-ctrl-two-run-0-100/policy.pt")
        state_dict = loaded_model.state_dict()
        policy.load_state_dict(state_dict)
    else:
        obs_key = 'observation'
        additional_keys = []
        M = variant['fc_layer_size']
        policy = TanhGaussianPolicy(
            obs_dim=dims['policy_obs_dim'],
            action_dim=dims['action_dim'],
            hidden_sizes=[M for _ in range(variant['fc_layer_depth'])],
            **variant['policy_kwargs']
        )
        policy.load_state_dict(torch.load("/home/julius/robotics/real_runs/thue-4cm-fullstate-run-0-646/current_policy.mdl", map_location="cpu"))

    policy = MakeDeterministic(policy)

    num_suc = 0
    tries = 0
    while num_suc < 20:
        trajectory_log = pd.DataFrame()
        dir = os.path.join(foddl, str(tries))
        os.makedirs(dir, exist_ok=True)
        print("Roll", tries)
        o = eval_env.reset()
        success = False
        d = False
        steps = 0
        while steps < 25 and not d:
            #print("Step", steps, i)
            steps += 1
            o_for_agent = obs_processor(o, obs_key, additional_keys)
            a, agent_info, _ = policy.get_action(o_for_agent)
            next_o, r, d, env_info = eval_env.step(a.copy())
            #print("Succ", env_info["is_success"], d)
            if env_info["is_success"]:
                success = True
                print("suc", steps )

            trajectory_log_entry = eval_env.get_trajectory_log_entry()
            trajectory_log_entry["is_success"] = env_info['is_success']
            trajectory_log = trajectory_log.append(trajectory_log_entry, ignore_index=True)

            o = next_o
            if variant["image_training"]:
                image = o['image']
                image = image.reshape((-1, 100, 100))*255
                cv2.imwrite(os.path.join(dir, f"{steps}.png"), image[0])
            else:
                corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image = eval_env.capture_images(None)
                cv2.imwrite(os.path.join(dir, f"{steps}.png"), cnn_image)
        if not success:
            shutil.rmtree(dir)
            print("rem fold")
        else:
            num_suc += 1
            trajectory_log.to_csv(os.path.join(dir, "trajectory.csv"))
            print("keep fold")
        tries += 1
        print("Success rate", num_suc/tries)











if __name__ == "__main__":
    args = argsparser()
    variant, arg_str = get_variant(args)


    evaluate(variant)