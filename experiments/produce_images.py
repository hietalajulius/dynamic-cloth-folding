from clothmanip.utils.utils import get_variant, argsparser, get_randomized_env, dump_commit_hashes, get_keys_and_dims, dump_goal
from clothmanip.envs.cloth import ClothEnvPickled as ClothEnv
import numpy as np
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, TanhScriptPolicy, CustomScriptPolicy, CustomTanhScriptPolicy, ScriptPolicy
import cv2
import os
from rlkit.envs.wrappers import NormalizedBoxEnv


def main(variant):

    variant['save_folder'] = "/home/julius/robotics/clothmanip/experiments/paper_images"
    env = ClothEnv(**variant['env_kwargs'], has_viewer=True, save_folder=variant['save_folder'])
    env = NormalizedBoxEnv(env)
    env = get_randomized_env(env, variant)

    keys, dims = get_keys_and_dims(variant, env)

    demo_path = variant['demo_paths'][0]
    predefined_actions = np.genfromtxt(demo_path, delimiter=',')

    iter_folder = os.path.join(variant['save_folder'], "close_no_corners", "0")

    os.makedirs(os.path.join(iter_folder, "corners_images"), exist_ok=True)
    #os.makedirs(os.path.join(iter_folder, "env_images"), exist_ok=True)
    #os.makedirs(os.path.join(iter_folder, "cnn_images"), exist_ok=True)
    #os.makedirs(os.path.join(iter_folder, "cnn_color_images"), exist_ok=True)
    #os.makedirs(os.path.join(iter_folder, "cnn_color_full_images"), exist_ok=True)

    policy = TanhScriptPolicy(
            output_size=dims['action_dim'],
            added_fc_input_size=dims['added_fc_input_size'],
            aux_output_size=9,
            **variant['policy_kwargs'],
        )

    eval_policy = MakeDeterministic(policy)


    for step_number, delta in enumerate(predefined_actions):
        print(step_number)
        a = delta/env.output_max
        a = np.clip(a, -1, 1)

        corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image = env.capture_images(None, mask_type=None)
        cv2.imwrite(f'{iter_folder}/corners_images/{str(step_number).zfill(3)}.png', corner_image)
        #cv2.imwrite(f'{iter_folder}/env_images/{str(step_number).zfill(3)}.png', eval_image)
        #cv2.imwrite(f'{iter_folder}/cnn_images/{str(step_number).zfill(3)}.png', corner_image)
        #cv2.imwrite(f'{iter_folder}/cnn_color_images/{str(step_number).zfill(3)}.png', cnn_color_image)
        #cv2.imwrite(f'{iter_folder}/cnn_color_full_images/{str(step_number).zfill(3)}.png', cnn_color_image_full)

        o, r, d, env_info = env.step(a)









if __name__ == "__main__":
    args = argsparser()
    variant, arg_str = get_variant(args)
    main(variant)