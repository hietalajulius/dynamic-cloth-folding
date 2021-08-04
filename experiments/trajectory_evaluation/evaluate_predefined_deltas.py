import json, copy, time
from rlkit.envs.wrappers import NormalizedBoxEnv
import numpy as np
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
import cv2
import os
from clothmanip.utils.utils import get_variant, argsparser, get_randomized_env, dump_commit_hashes, get_keys_and_dims, dump_goal
import pandas as pd


def main(variant):
    constant_goal_variant = copy.deepcopy(variant)
    constant_goal_variant['env_kwargs']['constant_goal'] = True
    env = ClothEnv(**constant_goal_variant['env_kwargs'], has_viewer=True, save_folder=variant['save_folder'])
    goal = env.goal.copy()
    dump_goal(variant['save_folder'], goal)

    df = pd.read_csv("/home/clothmanip/robotics/clothmanip/clothmanip/envs/cloth_data/cloth_optimization_stats_examine.csv")
    predefined_actions = np.genfromtxt("/home/clothmanip/robotics/clothmanip/clothmanip/envs/cloth_data/scaled_executable_raw_actions.csv", delimiter=',')

    for idx, row in df.iterrows():
        env.setup_xml_model(randomize=False, rownum=None)
        o = env.reset()
        for delta in predefined_actions:
            corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image = env.capture_images(None)
            cv2.imshow("IMage", corner_image/255)
            cv2.imshow("IMage2", o["image"].copy().reshape((100, 100, 1)))
            cv2.waitKey(1)
            raw_action = np.clip(delta/env.output_max, -1, 1)
            o, r, d, env_info = env.step(raw_action)


        corner_dists = [env_info[f"corner_{key}"] for key in range(4)]
        corner_dist = np.array(corner_dists).sum()
        df.at[idx, "corner_0_distance"] = corner_dists[0]
        df.at[idx, "corner_sum_distance"] = corner_dist

    df.to_csv("/home/clothmanip/robotics/clothmanip/clothmanip/envs/cloth_data/cloth_optimization_stats_updated.csv")

if __name__ == "__main__":
    args = argsparser()
    variant, arg_str = get_variant(args)

    logger_path = args.title + "-evaluation-" + str(args.run)
    variant['save_folder'] = f"./trainings/{logger_path}"

    os.makedirs(variant['save_folder'], exist_ok=True)

    with open(f"{variant['save_folder']}/params.json", "w") as outfile:
        json.dump(variant, outfile)
    with open(f"{variant['save_folder']}/command.txt", "w") as outfile:
        json.dump(arg_str, outfile)

    dump_commit_hashes(variant['save_folder'])


    print("Profiling with cProfile")
    main(variant)