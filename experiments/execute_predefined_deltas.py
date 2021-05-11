import argparse
import json
from rlkit.envs.wrappers import NormalizedBoxEnv
import numpy as np
from envs.cloth import ClothEnvPickled as ClothEnv
import cv2
import os


def main(variant, folder, output_folder):
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

    variant['env_kwargs']['output_max'] = 1
    env = ClothEnv(**variant['env_kwargs'], has_viewer=True, save_folder=output_folder, initial_xml_dump=True)
    env = NormalizedBoxEnv(env)

    env_timestep = env.timestep
    steps_per_second = 1 / env.timestep
    new_action_every_ctrl_step = steps_per_second / variant['env_kwargs']['control_frequency']


    o = env.reset()  

    trajectory_log = []
    trajectory_log.append(np.concatenate([env.desired_pos_step_W, env.desired_pos_ctrl_W, env.get_ee_position_I(), env.get_ee_position_W(), np.zeros(9)]))

    deltas =  np.genfromtxt(f"{folder}/executable_deltas.csv", delimiter=',')

    for path_length, delta in enumerate(deltas):
        print("del exec", path_length)
        train_image, eval_image = env.capture_image(None)
        cv2.imwrite(f'{output_folder}/sim_eval_images/corners/{str(path_length).zfill(3)}.png', train_image)
        cv2.imwrite(f'{output_folder}/sim_eval_images/eval/{str(path_length).zfill(3)}.png', eval_image)

        if "image" in o.keys():
            data = o['image'].copy().reshape((100, 100, 1))
            cv2.imwrite(f'{output_folder}/sim_eval_images/cnn/{str(path_length).zfill(3)}.png', data*255)
        
        o, r, d, env_info = env.step(delta[:3])

        delta = env.get_ee_position_W() - trajectory_log[-1][9:12]
        velocity = delta / (env_timestep*new_action_every_ctrl_step)
        acceleration = (velocity - trajectory_log[-1][15:18]) / (env_timestep*new_action_every_ctrl_step)
        
        trajectory_log.append(np.concatenate([env.desired_pos_step_W, env.desired_pos_ctrl_W, env.get_ee_position_I(), env.get_ee_position_W(), delta, velocity, acceleration]))

    np.savetxt(f"{output_folder}/sim_trajectory.csv",
                    trajectory_log, delimiter=",", fmt='%f')

    


    print("Done")

          

    













if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_folder', type=str)

    args = parser.parse_args()


    with open(f"{args.input_folder}/params.json")as json_file:
        variant = json.load(json_file)

    main(variant, args.input_folder, args.output_folder)