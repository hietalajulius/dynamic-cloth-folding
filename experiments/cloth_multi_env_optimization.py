import argparse
import copy
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
import numpy as np
from rlkit.envs.wrappers import SubprocVecEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
import json
import pandas as pd
import mujoco_py
from gym.logger import set_level
import cv2
import os
import shutil
from clothmanip.utils.utils import get_variant, argsparser, get_randomized_env, dump_commit_hashes, get_keys_and_dims
set_level(50)

def get_env_fn(variant, output_folder):
    def env_fn():
        env = NormalizedBoxEnv(ClothEnv(**variant['env_kwargs'], save_folder=output_folder, has_viewer=True))
        return env

    return env_fn

default_model_kwargs = dict(
        joint_solimp_low = 0.986633333333333,
        joint_solimp_high = 0.9911,
        joint_solimp_width = 0.03,
        joint_solref_timeconst  = 0.03,
        joint_solref_dampratio = 1.01555555555556,

        tendon_shear_solimp_low = 0.98,
        tendon_shear_solimp_high = 0.99,
        tendon_shear_solimp_width = 0.03,
        tendon_shear_solref_timeconst  = 0.05,
        tendon_shear_solref_dampratio = 1.01555555555556,

        tendon_main_solimp_low = 0.993266666666667,
        tendon_main_solimp_high = 0.9966,
        tendon_main_solimp_width = 0.004222222222222,
        tendon_main_solref_timeconst  = 0.01,
        tendon_main_solref_dampratio = 0.98,

        geom_solimp_low = 0.984422222222222,
        geom_solimp_high = 0.9922,
        geom_solimp_width = 0.007444444444444,
        geom_solref_timeconst  = 0.005,
        geom_solref_dampratio = 1.01555555555556,

        grasp_solimp_low = 0.99,
        grasp_solimp_high = 0.99,
        grasp_solimp_width = 0.01,
        grasp_solref_timeconst  = 0.01,
        grasp_solref_dampratio = 1,

        geom_size = 0.011,
        friction = 0.05,
        impratio = 20
)

model_kwarg_ranges = dict(
    joint_solimp_low = (0.99,0.9999),
    joint_solimp_high = (0.99,0.9999),
    joint_solimp_width = (0.01, 0.03),
    joint_solref_timeconst  = (0.01, 0.05),
    joint_solref_dampratio = (0.98, 1.01555555555556),

    tendon_shear_solimp_low = (0.99,0.9999),
    tendon_shear_solimp_high = (0.99,0.9999),
    tendon_shear_solimp_width = (0.01, 0.03),
    tendon_shear_solref_timeconst  = (0.01, 0.05),
    tendon_shear_solref_dampratio = (0.98, 1.01555555555556),

    tendon_main_solimp_low = (0.99,0.9999),
    tendon_main_solimp_high = (0.99,0.9999),
    tendon_main_solimp_width = (0.01, 0.03),
    tendon_main_solref_timeconst  = (0.01, 0.05),
    tendon_main_solref_dampratio = (0.98, 1.01555555555556),

    geom_solimp_low = (0.99,0.9999),
    geom_solimp_high = (0.99,0.9999),
    geom_solimp_width = (0.01, 0.03),
    geom_solref_timeconst  = (0.01, 0.05),
    geom_solref_dampratio = (0.98, 1.01555555555556),

    grasp_solimp_low = (0.99,0.9999),
    grasp_solimp_high = (0.99,0.9999),
    grasp_solimp_width = (0.01, 0.02),
    grasp_solref_timeconst  = (0.01, 0.02),
    grasp_solref_dampratio = (0.98, 1.01555555555556),

    geom_size = (0.008, 0.011),
    friction = (0.01, 10),
    impratio = (1, 40)
)

def rollout(model_kwargs_list, variant, input_folder, output_folder_paths):
    num_trajs = len(model_kwargs_list)
    variants_list = []
    for output_folder in output_folder_paths:
        try:
            os.makedirs(output_folder)
            profile_path = f"{output_folder}/profiling"
            images_path = f"{output_folder}/images"
            os.makedirs(images_path)
            os.makedirs(profile_path)
        except:
            print("folders exist already")

    for model_kwargs in model_kwargs_list:
        fresh_variant = copy.deepcopy(variant)
        fresh_variant["env_kwargs"]["override_model_kwargs"] = model_kwargs
        fresh_variant["env_kwargs"]["cloth_type"] = "override"
        variants_list.append(fresh_variant)

    env_fns = [get_env_fn(variants_list[i], output_folder_paths[i]) for i in range(num_trajs)]
    vec_env = SubprocVecEnv(env_fns)

    deltas =  np.genfromtxt(f"{input_folder}/executable_raw_actions.csv", delimiter=',')
    ee_positions = [[] for _ in range(num_trajs)]
    all_images = [[] for _ in range(num_trajs)]
    vec_env.reset()
    smallest_reached_goals = np.ones(num_trajs)*np.inf
    got_dones = np.zeros(num_trajs)
    for delta in deltas:
        actions = np.array([delta for _ in range(num_trajs)])*1.2/variant["env_kwargs"]["output_max"]
        observations, rewards, dones, infos = vec_env.step(actions)
        reached_goals = np.array([np.linalg.norm(observations['achieved_goal'][i]-observations['desired_goal'][i]) for i in range(num_trajs)])
        got_dones[dones] = True
        smallest_reached_goals = np.min([smallest_reached_goals, reached_goals], axis=0)

        for t in range(num_trajs):
                data = observations['image'][t].copy().reshape((-1, 100, 100))
                for i, image in enumerate(data):
                    reshaped_image = image.reshape(100,100, 1)
                    all_images[t].append(reshaped_image)
                ee_positions[t].append(observations['robot_observation'][t][:3])

    


    for t_idx, images in enumerate(all_images):
        for image_idx, image in enumerate(images):
            cv2.imwrite(f'{output_folder_paths[t_idx]}/images/{image_idx}.png', image*255)

        np.savetxt(f"{output_folder_paths[t_idx]}/ee_positions.csv",
                    ee_positions[t_idx], delimiter=",", fmt='%f')
    if np.any(got_dones):
        print("Some got done", len(deltas))
    vec_env.close()
    
    return got_dones, smallest_reached_goals, reached_goals


def main(variant, folder):
    num_trajs = variant['num_processes']
    stats_df = pd.DataFrame()
    ranges = list(model_kwarg_ranges.values())
    tests = 0
    while True:
        print("Test", tests)
        model_kwarg_list = []
        
        for _ in range(num_trajs):
            kwargs = copy.deepcopy(default_model_kwargs)
            for idx, key in enumerate(default_model_kwargs.keys()):
                low = ranges[idx][0]
                high = ranges[idx][1]
                kwargs[key] = np.random.uniform(low, high)
            kwargs['cone_type'] = np.random.choice(["pyramidal", "elliptic"])
            kwargs['domain_randomization'] = True
            model_kwarg_list.append(kwargs)

        output_folder_paths = [folder + f"/param_optimization/{tests*num_trajs+i}" for i in range(num_trajs)]

        got_dones, smallest_reached_goals, reached_goals_at_end = rollout(model_kwarg_list, variant, folder, output_folder_paths)


        for model_kwargs, got_done, smallest_reached_goal, reached_goal_at_end in zip(model_kwarg_list, got_dones, smallest_reached_goals, reached_goals_at_end):
            settings = dict(got_done=got_done,
                            a_smallest_reached_goal=smallest_reached_goal,
                            a_reached_goal_at_end=reached_goal_at_end,
                            **model_kwargs
                            )
            stats_df = stats_df.append(
                    settings, ignore_index=True)

        for i, smallest_reached_goal in enumerate(smallest_reached_goals):
            if smallest_reached_goal > 0.5:
                shutil.rmtree(output_folder_paths[i])

        stats_df.to_csv(f"{folder}/cloth_optimization_stats.csv")
        tests += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('folder', type=str)

    args = parser.parse_args()


    with open(f"{args.folder}/params.json")as json_file:
        variant = json.load(json_file)

    with mujoco_py.ignore_mujoco_warnings():
        main(variant, args.folder)