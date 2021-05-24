from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
from clothmanip.utils.utils import get_variant, argsparser
import json
import numpy as np
import cv2 

def main(variant):

    env = ClothEnv(**variant['env_kwargs'], has_viewer=True, save_folder="./", initial_xml_dump=True)
    example_trajectory =  np.genfromtxt("example_trajectory.csv", delimiter=',')
    
    env.reset()
    for i, row in enumerate(example_trajectory):
        action = row[:3]/variant["env_kwargs"]["output_max"]
        o, _, _, _ = env.step(action)

        corner_image, full_image = env.capture_image()
        cnn_image = o['image'].copy().reshape((100, 100, 1))*255
        cv2.imwrite(f'./images/corners/{str(i).zfill(3)}.png', corner_image)
        cv2.imwrite(f'./images/full/{str(i).zfill(3)}.png', full_image)
        cv2.imwrite(f'./images/cnn/{str(i).zfill(3)}.png', cnn_image)

    print("Saved images to ./images")


if __name__ == "__main__":
    
    with open("example_params.json")as json_file:
        variant = json.load(json_file)

    #Alternatively create own variant based on command line args by uncommentting below
    #args = argsparser()
    #variant, arg_str = get_variant(args)'

    main(variant)