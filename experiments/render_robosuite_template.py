from utils import get_variant, argsparser
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import fileinput


def get_robosuite_env(variant, evaluation=False):
    options = {}
    options["env_name"] = "Cloth"
    options["robots"] = "PandaReal"
    controller_name = "OSC_POS_VEL"
    options["controller_configs"] = load_controller_config(
        default_controller=controller_name)

    outp_override = 0.05
    options["controller_configs"]['output_max'][:3] = [
        outp_override for _ in range(3)]
    options["controller_configs"]['output_min'][:3] = [
        -outp_override for _ in range(3)]

    options["controller_configs"]['input_min'] = -1
    options["controller_configs"]['input_max'] = 1

    options["controller_configs"]["interpolation"] = 'filter'
    options["controller_configs"]["ramp_ratio"] = 0.03
    options["controller_configs"]["damping_ratio"] = 1
    options["controller_configs"]["kp"] = 1000


    options["controller_configs"]["position_limits"] = None

    if variant['image_training'] or evaluation:
        has_offscreen_renderer = True
    else:
        has_offscreen_renderer = False

    
    del variant['env_kwargs']['stay_in_place_coef']
    del variant['env_kwargs']['ctrl_filter']
    del variant['env_kwargs']['sphere_clipping']
    del variant['env_kwargs']['damping_ratio']
    del variant['env_kwargs']['kp']
    del variant['env_kwargs']['cosine_sim_coef']
    del variant['env_kwargs']['error_norm_coef']
    del variant['env_kwargs']['output_max']
    env = suite.make(
        **options,
        **variant['env_kwargs'],
        has_renderer=False,
        has_offscreen_renderer=has_offscreen_renderer,
        ignore_done=False,
        use_camera_obs=False,
    )
    return env





def main(variant):

    print("what")
    env = get_robosuite_env(variant)
    print("Env", env)

    
    with open(f"robosuite_mujoco_templates/compiled_mujoco_model_no_inertias.xml", "w") as f:
        env.sim.save(f, format='xml', keep_inertials=False)

    with open(f"robosuite_mujoco_templates/compiled_mujoco_model_with_inertias.xml", "w") as f:
        env.sim.save(f, format='xml', keep_inertials=True)


        

    with fileinput.FileInput(f"robosuite_mujoco_templates/compiled_mujoco_model_no_inertias.xml", inplace=True) as file:
        for line in file:
            print(line.replace("/home/clothmanip/robotics", "{{ robotics_dir }}"), end='')
        for line in file:
            print(line.replace("/home/julius/robotics", "{{ robotics_dir }}"), end='')

    with fileinput.FileInput(f"robosuite_mujoco_templates/compiled_mujoco_model_with_inertias.xml", inplace=True) as file:
        for line in file:
            print(line.replace("/home/clothmanip/robotics", "{{ robotics_dir }}"), end='')
        for line in file:
            print(line.replace("/home/julius/robotics", "{{ robotics_dir }}"), end='')


















if __name__ == "__main__":
    args = argsparser()
    variant = get_variant(args)
    main(variant)