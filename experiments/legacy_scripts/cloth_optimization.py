import argparse
import json
from rlkit.envs.wrappers import NormalizedBoxEnv
import numpy as np
from envs.cloth import ClothEnvPickled as ClothEnv
import cv2
import os
import pandas as pd
import time
import shutil

def rollout(variant, output_folder, deltas):
    try:
        os.makedirs(output_folder)
        cnn_path = f"{output_folder}/sim_eval_images/cnn"
        corners_path = f"{output_folder}/sim_eval_images/corners"
        eval_path = f"{output_folder}/sim_eval_images/eval"
        os.makedirs(cnn_path)
        os.makedirs(corners_path)
        os.makedirs(eval_path)
    except:
        print("folders exist already")
    env = ClothEnv(**variant['env_kwargs'], has_viewer=True, save_folder=output_folder, initial_xml_dump=True)
    env = NormalizedBoxEnv(env)

    smallest_reached_goal = np.inf
    got_done = False

    env_timestep = env.timestep
    steps_per_second = 1 / env.timestep
    new_action_every_ctrl_step = steps_per_second / variant['env_kwargs']['control_frequency']


    o = env.reset()  

    trajectory_log = []
    trajectory_log.append(np.concatenate([env.desired_pos_step_W, env.desired_pos_ctrl_W, env.get_ee_position_I(), env.get_ee_position_W(), np.zeros(9)]))

    

    for path_length, delta in enumerate(deltas):
        train_image, eval_image = env.capture_image(None)
        cv2.imwrite(f'{output_folder}/sim_eval_images/corners/{str(path_length).zfill(3)}.png', train_image)
        cv2.imwrite(f'{output_folder}/sim_eval_images/eval/{str(path_length).zfill(3)}.png', eval_image)

        if "image" in o.keys():
            data = o['image'].copy().reshape((100, 100, 1))
            cv2.imwrite(f'{output_folder}/sim_eval_images/cnn/{str(path_length).zfill(3)}.png', data*255)
        
        o, r, d, env_info = env.step(delta[:3])

        if d:
            got_done = True

        reached_goal = np.linalg.norm(o['achieved_goal']-o['desired_goal'])
        print(path_length, "Reached goal", reached_goal, "Done", got_done)
        if reached_goal < smallest_reached_goal:
            smallest_reached_goal = reached_goal

        delta = env.get_ee_position_W() - trajectory_log[-1][9:12]
        velocity = delta / (env_timestep*new_action_every_ctrl_step)
        acceleration = (velocity - trajectory_log[-1][15:18]) / (env_timestep*new_action_every_ctrl_step)
        
        trajectory_log.append(np.concatenate([env.desired_pos_step_W, env.desired_pos_ctrl_W, env.get_ee_position_I(), env.get_ee_position_W(), delta, velocity, acceleration]))

    np.savetxt(f"{output_folder}/sim_trajectory.csv",
                    trajectory_log, delimiter=",", fmt='%f')


    return got_done, smallest_reached_goal, reached_goal




def main(variant, input_folder, base_output_folder):

    deltas =  np.genfromtxt(f"{input_folder}/executable_deltas.csv", delimiter=',')[20:]
    variant['env_kwargs']['output_max'] = 1
    del variant['env_kwargs']['timestep']

    joint_solimp_low_range = np.linspace(0.98, 0.9999, 10)
    joint_solimp_high_range = np.linspace(0.99, 0.9999, 10)
    joint_solimp_width_range = np.linspace(0.001, 0.03, 10)
    joint_solref_timeconst_range = np.linspace(0.005, 0.05, 10)
    joint_solref_dampratio_range = np.linspace(0.98, 1.02, 10)

    tendon_shear_solimp_low_range = np.linspace(0.98, 0.9999, 10)
    tendon_shear_solimp_high_range = np.linspace(0.99, 0.9999, 10)
    tendon_shear_solimp_width_range = np.linspace(0.001, 0.03, 10)
    tendon_shear_solref_timeconst_range = np.linspace(0.005, 0.05, 10)
    tendon_shear_solref_dampratio_range = np.linspace(0.98, 1.02, 10)

    tendon_main_solimp_low_range = np.linspace(0.98, 0.9999, 10)
    tendon_main_solimp_high_range = np.linspace(0.99, 0.9999, 10)
    tendon_main_solimp_width_range = np.linspace(0.001, 0.03, 10)
    tendon_main_solref_timeconst_range = np.linspace(0.005, 0.05, 10)
    tendon_main_solref_dampratio_range = np.linspace(0.98, 1.02, 10)

    geom_solimp_low_range = np.linspace(0.98, 0.9999, 10)
    geom_solimp_high_range = np.linspace(0.99, 0.9999, 10)
    geom_solimp_width_range = np.linspace(0.001, 0.03, 10)
    geom_solref_timeconst_range = np.linspace(0.005, 0.05, 10)
    geom_solref_dampratio_range = np.linspace(0.98, 1.02, 10)

    grasp_solimp_low_range = np.linspace(0.98, 0.9999, 10)
    grasp_solimp_high_range = np.linspace(0.99, 0.9999, 10)
    grasp_solimp_width_range = np.linspace(0.001, 0.03, 10)
    grasp_solref_timeconst_range = np.linspace(0.005, 0.05, 10)
    grasp_solref_dampratio_range = np.linspace(0.98, 1.02, 10)

    geom_size_range = np.linspace(0.008, 0.011, 10)
    friction_range = np.linspace(0.05, 1, 10)
    cone_type_range = ["pyramidal", "elliptic"]

    stats_df = pd.DataFrame()

    tests = 0
    while True:
        model_kwargs = dict(
            joint_solimp_low = np.random.choice(joint_solimp_low_range),
            joint_solimp_high = np.random.choice(joint_solimp_high_range),
            joint_solimp_width = np.random.choice(joint_solimp_width_range),
            joint_solref_timeconst  = np.random.choice(joint_solref_timeconst_range),
            joint_solref_dampratio = np.random.choice(joint_solref_dampratio_range),

            tendon_shear_solimp_low = np.random.choice(tendon_shear_solimp_low_range),
            tendon_shear_solimp_high = np.random.choice(tendon_shear_solimp_high_range),
            tendon_shear_solimp_width = np.random.choice(tendon_shear_solimp_width_range),
            tendon_shear_solref_timeconst  = np.random.choice(tendon_shear_solref_timeconst_range),
            tendon_shear_solref_dampratio = np.random.choice(tendon_shear_solref_dampratio_range),

            tendon_main_solimp_low = np.random.choice(tendon_main_solimp_low_range),
            tendon_main_solimp_high = np.random.choice(tendon_main_solimp_high_range),
            tendon_main_solimp_width = np.random.choice(tendon_main_solimp_width_range),
            tendon_main_solref_timeconst  = np.random.choice(tendon_main_solref_timeconst_range),
            tendon_main_solref_dampratio = np.random.choice(tendon_main_solref_dampratio_range),

            geom_solimp_low = np.random.choice(geom_solimp_low_range),
            geom_solimp_high = np.random.choice(geom_solimp_high_range),
            geom_solimp_width = np.random.choice(geom_solimp_width_range),
            geom_solref_timeconst  = np.random.choice(geom_solref_timeconst_range),
            geom_solref_dampratio = np.random.choice(geom_solref_dampratio_range),

            grasp_solimp_low = np.random.choice(grasp_solimp_low_range),
            grasp_solimp_high = np.random.choice(grasp_solimp_high_range),
            grasp_solimp_width = np.random.choice(grasp_solimp_width_range),
            grasp_solref_timeconst  = np.random.choice(grasp_solref_timeconst_range),
            grasp_solref_dampratio = np.random.choice(grasp_solref_dampratio_range),

            geom_size = np.random.choice(geom_size_range),
            friction = np.random.choice(friction_range),
            cone_type = np.random.choice(cone_type_range),
            timestep=0.01
        )

        if model_kwargs['joint_solimp_low'] > model_kwargs['joint_solimp_high'] or model_kwargs['tendon_shear_solimp_low'] > model_kwargs['tendon_shear_solimp_high'] or model_kwargs['tendon_main_solimp_low'] > model_kwargs['tendon_main_solimp_high'] or model_kwargs['geom_solimp_low'] > model_kwargs['geom_solimp_high'] or model_kwargs['grasp_solimp_low'] > model_kwargs['grasp_solimp_high']:
            print("Bad ranges")
        else:
            variant['env_kwargs']['mujoco_model_kwargs'] = model_kwargs
            output_folder_path = base_output_folder + f"/{tests}"

            start = time.time()
            got_done, smallest_reached_goal, reached_goal_at_end = rollout(variant, output_folder_path, deltas)
            end = time.time()

            settings = dict(folder=output_folder_path,
                            got_done=got_done,
                            a_smallest_reached_goal=smallest_reached_goal,
                            a_reached_goal_at_end=reached_goal_at_end,
                            elapsed_time=end-start,
                            **model_kwargs
                            )

            stats_df = stats_df.append(
                    settings, ignore_index=True)
            stats_df.to_csv(f"{base_output_folder}/cloth_optimization_stats.csv")

            if smallest_reached_goal > 0.05:
                shutil.rmtree(output_folder_path)

            tests += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_folder', type=str)

    args = parser.parse_args()


    with open(f"{args.input_folder}/params.json")as json_file:
        variant = json.load(json_file)

    main(variant, args.input_folder, args.output_folder)