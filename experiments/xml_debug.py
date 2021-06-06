import argparse
import json
import mujoco_py
from rlkit.envs.wrappers import NormalizedBoxEnv
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
import numpy as np


def main(variant, inp, outp):

    env = NormalizedBoxEnv(ClothEnv(
        **variant['env_kwargs'], save_folder=outp, has_viewer=True, initial_xml_dump=True))

    rollouts = 0
    while True:
        env.reset()
        for i in range(50):
            print(i, rollouts)
            env.step(np.random.normal(3))
        rollouts += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_folder', type=str)

    args = parser.parse_args()

    with open(f"{args.input_folder}/params.json")as json_file:
        variant = json.load(json_file)

    with mujoco_py.ignore_mujoco_warnings():
        main(variant, args.input_folder, args.output_folder)
