import cma
import numpy as np
from rlkit.envs.wrappers import SubprocVecEnv
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
import argparse
import json
import copy
import pandas as pd
import mujoco_py
from gym.logger import set_level
import cv2

set_level(50)


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
            joint_solimp_low = (0.97,0.9999),
            joint_solimp_high = (0.97,0.9999),
            joint_solimp_width = (0.01, 0.03),
            joint_solref_timeconst  = (0.01, 0.05),
            joint_solref_dampratio = (0.98, 1.01555555555556),

            tendon_shear_solimp_low = (0.97,0.9999),
            tendon_shear_solimp_high = (0.97,0.9999),
            tendon_shear_solimp_width = (0.01, 0.03),
            tendon_shear_solref_timeconst  = (0.01, 0.05),
            tendon_shear_solref_dampratio = (0.98, 1.01555555555556),

            tendon_main_solimp_low = (0.97,0.9999),
            tendon_main_solimp_high = (0.97,0.9999),
            tendon_main_solimp_width = (0.01, 0.03),
            tendon_main_solref_timeconst  = (0.01, 0.05),
            tendon_main_solref_dampratio = (0.98, 1.01555555555556),

            geom_solimp_low = (0.97,0.9999),
            geom_solimp_high = (0.97,0.9999),
            geom_solimp_width = (0.01, 0.03),
            geom_solref_timeconst  = (0.01, 0.05),
            geom_solref_dampratio = (0.98, 1.01555555555556),

            grasp_solimp_low = (0.9999,0.9999),
            grasp_solimp_high = (0.9999,0.9999),
            grasp_solimp_width = (0.01, 0.02),
            grasp_solref_timeconst  = (0.01, 0.02),
            grasp_solref_dampratio = (0.98, 1.01555555555556),

            geom_size = (0.008, 0.011),
            friction = (0.01, 10),
            impratio = (1, 40)
        )

def transform_cma_values_to_kwargs(values):
    ranges = list(model_kwarg_ranges.values())
    kwargs = copy.deepcopy(default_model_kwargs)
    for idx, key in enumerate(default_model_kwargs.keys()):
        a = ranges[idx][0]
        b = ranges[idx][1]
        x = values[idx]
        kwargs[key] = a + (b-a) * (1 - np.cos(np.pi * x / 10)) / 2

    return kwargs


def get_objective_function(variant, input_folder, output_folder):
    all_steps = []
    num_trajs = variant['num_processes']
    variant["cma_evals"] = 0
    for i in range(num_trajs):
        trajectory =  np.genfromtxt(f"{input_folder}/{i}/validation_trajectory.csv", delimiter=',')
        labels = pd.read_csv(f"{input_folder}/{i}/labels.csv", names=["corner", "u", "v", "file", "w", "h"])
        steps = []
        for j, row in enumerate(trajectory):
            if not j == 0:
                c0 = labels[(labels["corner"] == 0) & (labels["file"] == f'{j}.png')]
                c1 = labels[(labels["corner"] == 1) & (labels["file"] == f'{j}.png')]
                c2 = labels[(labels["corner"] == 2) & (labels["file"] == f'{j}.png')]
                c3 = labels[(labels["corner"] == 3) & (labels["file"] == f'{j}.png')]
                step = {'action': row[:3]/variant["env_kwargs"]["output_max"],
                        '0': np.array([c0['u']/100, c0['v']/100]).flatten(),
                        '1': np.array([c1['u']/100, c1['v']/100]).flatten(),
                        '2': np.array([c2['u']/100, c2['v']/100]).flatten(),
                        '3': np.array([c3['u']/100, c3['v']/100]).flatten(),
                }
                steps.append(step)
        all_steps.append(steps)

    max_traj_len = np.array([len(steps) for steps in all_steps]).max()



    def cma_objective(params):
        model_kwargs = dict(cone_type="pyramidal", timestep=variant["env_kwargs"]["mujoco_model_kwargs"]["timestep"], domain_randomization=False)
        for idx, key in enumerate(model_kwarg_ranges.keys()):
            a = model_kwarg_ranges[key][0]
            b = model_kwarg_ranges[key][1]
            x = params[idx]
            model_kwargs[key] = a + (b-a) * (1 - np.cos(np.pi * x / 10)) / 2

        variant["env_kwargs"]["mujoco_model_kwargs"] = model_kwargs
        dump_env = ClothEnv(**variant['env_kwargs'], has_viewer=True, save_folder=output_folder, initial_xml_dump=True)
        del dump_env
        env_fns = [lambda: NormalizedBoxEnv(ClothEnv(**variant['env_kwargs'], save_folder=output_folder, has_viewer=True)) for _ in range(num_trajs)]
        vec_env = SubprocVecEnv(env_fns)

        vec_env.reset()
        error = 0
        offsets = [dict() for _ in range(num_trajs)]
        all_images = [[] for _ in range(num_trajs)]
        for step_idx in range(max_traj_len):
            actions = np.zeros((num_trajs,3))
            real_corners = []
            for traj_idx in range(num_trajs):
                step = all_steps[traj_idx][-1]
                if step_idx < len(all_steps[traj_idx]):
                    step = all_steps[traj_idx][step_idx]
                    actions[traj_idx] = step['action']

                real_corners.append({"0": step["0"], "1": step["1"], "2": step["2"], "3": step["3"]})
            o, _, _, infos = vec_env.step(actions)

            for t in range(num_trajs):
                data = o['image'][t].copy().reshape((-1, 100, 100))
                for i, image in enumerate(data):
                    reshaped_image = image.reshape(100,100, 1)
                    all_images[t].append(reshaped_image)

                        


            corner_errors = 0
            #print("Step", step_idx)
            for info_idx, info in enumerate(infos):
                sim_corner_indices = np.array(info['corner_indices'])
                sim_corner_positions = np.array(info['corner_positions'])
                

                
                #offset_idx = np.where(sim_corner_indices == 1)[0]
                #offset_corner_real = real_corners[info_idx]['1']
                #offset_corner_sim = np.array([sim_corner_positions[offset_idx*2], sim_corner_positions[offset_idx*2+1]]).flatten()
                #offset = offset_corner_sim - offset_corner_real

                for sim_corner_idx, sim_corner_idx_in_real in enumerate(sim_corner_indices):
                    
                    corner_real = real_corners[info_idx][str(sim_corner_idx_in_real)]
                    corner_sim = np.array([sim_corner_positions[sim_corner_idx*2], sim_corner_positions[sim_corner_idx*2+1]]).flatten()
                    if step_idx == 0:
                        offsets[info_idx][sim_corner_idx_in_real] = corner_sim - corner_real

                    if corner_real.shape[0] == 2:
                        offset = offsets[info_idx][sim_corner_idx_in_real]
                        corner_error = np.linalg.norm(corner_real-corner_sim+offset)
                    else:
                        corner_error = 0
                    corner_errors += corner_error

            corner_errors /= num_trajs
            corner_errors /= max_traj_len
            corner_errors *= 100
            error += corner_errors

            #print("Corner errors", corner_errors, "Total error", error)
            
        #print("Total error", error)

        vec_env.close()

        if error < variant["lowest_error"]:
            variant["lowest_error"] = error
            for t_idx, images in enumerate(all_images):
                for image_idx, image in enumerate(images):
                    cv2.imwrite(f'{output_folder}/images/{t_idx}/{image_idx}.png', image*255)
            print(f"New lowest error {error}, wrote images")
        return error


    return cma_objective

def main(variant, input_folder, output_folder):
    variant["lowest_error"] = np.inf
    cma_options = cma.evolution_strategy.CMAOptions()
    cma_options['maxiter'] = 10
    cma_options['maxfevals'] = 100*len(list(default_model_kwargs.values()))
    es = cma.CMAEvolutionStrategy(np.ones(28)*5, 2)
    cma_objective = get_objective_function(variant, input_folder, output_folder)

    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [cma_objective(x) for x in solutions])
        es.disp()
        es.result_pretty()
        xbest = es.result[0]
        print("Best kwargs so far", transform_cma_values_to_kwargs(xbest))
        print("\n")

    es.result_pretty()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_folder', type=str)

    args = parser.parse_args()


    with open(f"{args.input_folder}/params.json")as json_file:
        variant = json.load(json_file)
    with mujoco_py.ignore_mujoco_warnings():
        main(variant, args.input_folder, args.output_folder)