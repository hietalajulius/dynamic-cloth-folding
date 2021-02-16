"""
Script to showcase domain randomization functionality.
"""

import robosuite.utils.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.wrappers import DomainRandomizationWrapper

from gym.envs.robotics import task_definitions
from utils import get_variant, argsparser


from mujoco_py.modder import TextureModder, MaterialModder


DEFAULT_LIGHTING_ARGS = {
    'light_names': None,  # all lights are randomized
    'randomize_position': True,
    'randomize_direction': True,
    'randomize_specular': True,
    'randomize_ambient': True,
    'randomize_diffuse': True,
    'randomize_active': True,
    'position_perturbation_size': 0.5,
    'direction_perturbation_size': 0.95,
    'specular_perturbation_size': 0.5,
    'ambient_perturbation_size': 0.5,
    'diffuse_perturbation_size': 0.5,
}

DEFAULT_CAMERA_ARGS = {
    'camera_names': None,  # all cameras are randomized
    'randomize_position': True,
    'randomize_rotation': True,
    'randomize_fovy': True,
    'position_perturbation_size': 0.01,
    'rotation_perturbation_size': 0.087,
    'fovy_perturbation_size': 5.,
}


def get_robosuite_env(variant):
    options = {}
    options["env_name"] = variant["env_name"]
    options["robots"] = "Panda"
    controller_name = variant["ctrl_name"]
    options["controller_configs"] = load_controller_config(
        default_controller=controller_name)
    options["controller_configs"]["interpolation"] = "linear"
    env = suite.make(
        **options,
        **variant['env_kwargs'],
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
    )
    return env


# We'll use instance randomization so that entire geom groups are randomized together
macros.USING_INSTANCE_RANDOMIZATION = True

if __name__ == "__main__":

    args = argsparser()
    variant = get_variant(args)

    env = get_robosuite_env(variant)

    camera_randomization_args = DEFAULT_CAMERA_ARGS
    camera_randomization_args['camera_names'] = ['clothview']
    env = DomainRandomizationWrapper(
        env, randomize_on_reset=True,
        randomize_every_n_steps=0, randomize_color=False, randomize_lighting=True, lighting_randomization_args=DEFAULT_LIGHTING_ARGS,  camera_randomization_args=camera_randomization_args)

    modder = TextureModder(env.sim)

    # Get action limits
    low, high = env.action_spec

    # do visualization
    while True:
        skin_mat = env.sim.model.skin_matid[0]
        for name in ['la_tabla_vis']:
            modder.whiten_materials()
            modder.set_checker(name, (255, 0, 0), (0, 0, 0))
            modder.rand_all(name)
        modder.set_checker('skin', (255, 0, 0), (0, 0, 0))
        modder.rand_all('skin')

        env.reset()
        for i in range(100):
            action = np.ones(3)*0.1
            obs, reward, done, _ = env.step(action)
            env.render()
