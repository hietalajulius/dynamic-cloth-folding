import mujoco_py
import osc_binding
import cv2
import numpy as np
import copy
from gym.utils import seeding, EzPickle
from utils import reward_calculation
import gym
import os
from scipy import spatial
from env.template_renderer import TemplateRenderer
import math
from collections import deque
import copy
from utils import mujoco_model_kwargs
from shutil import copyfile
import psutil
import gc
from xml.dom import minidom
from mujoco_py.utils import remove_empty_lines
import albumentations as A
import pandas as pd
from utils import task_definitions
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


def compute_cosine_distance(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0
    else:
        cosine_dist = spatial.distance.cosine(vec1, vec2)
        if np.isnan(cosine_dist):
            return 0
        else:
            return cosine_dist


class ClothEnv_(object):
    def __init__(
        self,
        timestep,
        sparse_dense,
        success_distance,
        goal_noise_range,
        frame_stack_size,
        output_max,
        success_reward,
        fail_reward,
        extra_reward,
        kp,
        damping_ratio,
        control_frequency,
        ctrl_filter,
        save_folder,
        randomization_kwargs,
        robot_observation,
        max_close_steps,
        model_kwargs_path,
        image_obs_noise_mean=1,
        image_obs_noise_std=0,
        has_viewer=True,
        image_size=100,
    ):
        self.albumentations_transform = A.Compose(
            [
                A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                           b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Blur(blur_limit=7, always_apply=False, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2,
                              saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0,
                               always_apply=False, p=0.5),
            ]
        )

        self.model_kwargs_path = model_kwargs_path
        self.success_distance = success_distance

        self.process = psutil.Process(os.getpid())
        self.randomization_kwargs = randomization_kwargs
        file_dir = os.path.dirname(os.path.abspath(__file__))
        source = os.path.join(file_dir, "mujoco_templates", "arena.xml")
        destination = os.path.join(save_folder, "arena.xml")
        copyfile(source, destination)
        self.template_renderer = TemplateRenderer(save_folder)
        self.save_folder = save_folder
        self.filter = ctrl_filter
        self.kp = kp
        self.damping_ratio = damping_ratio
        self.output_max = output_max
        self.success_reward = success_reward
        self.fail_reward = fail_reward
        self.extra_reward = extra_reward
        self.sparse_dense = sparse_dense
        self.goal_noise_range = goal_noise_range
        self.frame_stack_size = frame_stack_size
        self.image_size = (image_size, image_size)

        self.has_viewer = has_viewer
        self.image_obs_noise_mean = image_obs_noise_mean
        self.image_obs_noise_std = image_obs_noise_std
        self.robot_observation = robot_observation

        self.max_close_steps = max_close_steps
        self.frame_stack = deque([], maxlen=self.frame_stack_size)

        self.limits_min = [-0.35, -0.35, 0.0]
        self.limits_max = [0.05, 0.05, 0.4]

        self.single_goal_dim = 3
        self.timestep = timestep
        self.control_frequency = control_frequency

        steps_per_second = 1 / self.timestep
        self.substeps = 1 / (self.timestep*self.control_frequency)
        self.between_steps = 1000 / steps_per_second
        self.delta_tau_max = 1000 / steps_per_second
        self.eval_camera = "eval_camera"
        self.episode_ee_close_steps = 0

        self.seed()
        self.initial_qpos = np.array(
            [0.212422, 0.362907, -0.00733391, -1.9649, -0.0198034, 2.37451, -1.50499])

        self.ee_site_name = 'grip_site'
        self.joints = ["joint1", "joint2", "joint3",
                       "joint4", "joint5", "joint6", "joint7"]

        self.mjpy_model = None
        self.sim = None
        self.viewer = None

        model_kwargs, model_numerical_values = self.build_xml_kwargs_and_numerical_values(
            randomize=self.randomization_kwargs['dynamics_randomization'])
        self.mujoco_model_numerical_values = model_numerical_values
        self.setup_initial_state_and_sim(model_kwargs)
        self.dump_xml_models()

        self.action_space = gym.spaces.Box(-1,
                                           1, shape=(3,), dtype='float32')

        self.relative_origin = self.get_ee_position_W()
        self.goal, self.goal_noise = self.sample_goal_I()

        self.reset_camera()

        image_obs = self.get_image_obs()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(image_obs)
        obs = self.get_obs()

        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-np.inf, np.inf,
                                        shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf,
                                         shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=gym.spaces.Box(-np.inf, np.inf,
                                       shape=obs['observation'].shape, dtype='float32'),
            robot_observation=gym.spaces.Box(-np.inf, np.inf,
                                             shape=obs['robot_observation'].shape, dtype='float32'),
            image=gym.spaces.Box(-np.inf, np.inf,
                                 shape=obs['image'].shape, dtype='float32')
        ))

    def get_model_kwargs(self, randomize, rownum=None):
        df = pd.read_csv(self.model_kwargs_path)

        model_kwargs = copy.deepcopy(mujoco_model_kwargs.BASE_MODEL_KWARGS)

        if randomize:
            choice = np.random.randint(0, df.shape[0] - 1)
            model_kwargs_row = df.iloc[choice]

        if rownum is not None:
            model_kwargs_row = df.iloc[rownum]

        for col in model_kwargs.keys():
            model_kwargs[col] = model_kwargs_row[col]

        return model_kwargs

    def build_xml_kwargs_and_numerical_values(self, randomize, rownum=None):
        model_kwargs = self.get_model_kwargs(
            randomize=randomize, rownum=rownum)

        model_numerical_values = []
        for key in model_kwargs.keys():
            value = model_kwargs[key]
            if type(value) in [int, float]:
                model_numerical_values.append(value)

        model_kwargs['floor_material_name'] = "floor_real_material"
        model_kwargs['table_material_name'] = "table_real_material"
        model_kwargs['cloth_material_name'] = "wipe_real_material"
        if self.randomization_kwargs['materials_randomization']:
            model_kwargs['floor_material_name'] = np.random.choice(
                ["floor_real_material", "floor_material"])
            model_kwargs['table_material_name'] = np.random.choice(
                ["table_real_material", "table_material"])
            model_kwargs['cloth_material_name'] = np.random.choice(["bath_real_material", "bath_2_real_material", "kitchen_real_material", "kitchen_2_real_material",
                                                                   "wipe_real_material", "wipe_2_real_material", "cloth_material", "white_real_material", "blue_real_material", "orange_real_material"])

        # General
        model_kwargs['timestep'] = self.timestep
        model_kwargs['lights_randomization'] = self.randomization_kwargs['lights_randomization']
        model_kwargs['materials_randomization'] = self.randomization_kwargs['materials_randomization']
        model_kwargs['train_camera_fovy'] = (self.randomization_kwargs['camera_config']
                                             ['fovy_range'][0] + self.randomization_kwargs['camera_config']['fovy_range'][1])/2
        model_kwargs['num_lights'] = 1

        model_kwargs['geom_spacing'] = (
            self.randomization_kwargs['cloth_size'] - 2*model_kwargs['geom_size']) / 8
        model_kwargs['offset'] = 4 * model_kwargs['geom_spacing']

        # Appearance
        appearance_choices = mujoco_model_kwargs.appearance_kwarg_choices
        appearance_ranges = mujoco_model_kwargs.appearance_kwarg_ranges
        for key in appearance_choices.keys():
            model_kwargs[key] = np.random.choice(appearance_choices[key])
        for key in appearance_ranges.keys():
            values = appearance_ranges[key]
            model_kwargs[key] = np.random.uniform(values[0], values[1])

        # Camera fovy
        if self.randomization_kwargs['camera_position_randomization']:
            model_kwargs['train_camera_fovy'] = np.random.uniform(
                self.randomization_kwargs['camera_config']['fovy_range'][0], self.randomization_kwargs['camera_config']['fovy_range'][1])

        min_corner = 0
        max_corner = 8
        self.max_corner_name = f"B{max_corner}_{max_corner}"
        self.mid_corner_index = 4
        mid = int(max_corner / 2)
        self.corner_index_mapping = {"0": f"S{min_corner}_{max_corner}", "1": f"S{max_corner}_{max_corner}",
                                     "2": f"S{min_corner}_{min_corner}", "3": f"S{max_corner}_{min_corner}"}
        self.cloth_site_names = []

        for i in [min_corner, mid, max_corner]:
            for j in [min_corner, mid, max_corner]:
                self.cloth_site_names.append(f"S{i}_{j}")

        # TODO: remove side effects from methods,
        self.constraints = task_definitions.constraints["sideways"](
            0, 4, 8, self.success_distance)

        self.task_reward_function = reward_calculation.get_task_reward_function(
            self.constraints, self.single_goal_dim, self.sparse_dense, self.success_reward, self.fail_reward, self.extra_reward)

        return model_kwargs, model_numerical_values

    def setup_viewer(self):
        if self.has_viewer:
            if not self.viewer is None:
                del self.viewer
            self.viewer = mujoco_py.MjRenderContextOffscreen(
                self.sim, device_id=-1)
            self.viewer.vopt.geomgroup[0] = 0
            self.viewer.vopt.geomgroup[1] = 1

    def dump_xml_models(self):
        with open(f"{self.save_folder}/compiled_mujoco_model_no_inertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=False)

        with open(f"{self.save_folder}/compiled_mujoco_model_with_intertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=True)

    def reset_camera(self):
        lookat_offset = np.zeros(3)
        self.train_camera = self.randomization_kwargs['camera_type']
        if self.train_camera == "all":
            self.train_camera = np.random.choice(["up", "front", "side"])

        if self.randomization_kwargs['lookat_position_randomization']:
            radius = self.randomization_kwargs['lookat_position_randomization_radius']
            lookat_offset[0] += np.random.uniform(-radius, radius)
            lookat_offset[1] += np.random.uniform(-radius, radius)

        des_cam_look_pos = self.sim.data.get_body_xpos(
            f"B{self.mid_corner_index}_{self.mid_corner_index}").copy() + lookat_offset
        self.sim.data.set_mocap_pos("lookatbody", des_cam_look_pos)

    def add_mocap_to_xml(self, xml):
        dom = minidom.parseString(xml)
        for subelement in dom.getElementsByTagName("body"):
            if subelement.getAttribute("name") == self.max_corner_name:
                subelement.setAttribute("mocap", "true")
                for child_node in subelement.childNodes:
                    if child_node.nodeType == 1:
                        if child_node.tagName == "joint":
                            subelement.removeChild(child_node)

        return remove_empty_lines(dom.toprettyxml(indent=" " * 4))

    def setup_initial_state_and_sim(self, model_kwargs):
        if not self.mjpy_model is None:
            del self.mjpy_model
        if not self.sim is None:
            del self.sim
        temp_xml_1 = self.template_renderer.render_template(
            "arena.xml", **model_kwargs)
        temp_model = mujoco_py.load_model_from_xml(temp_xml_1)
        temp_xml_2 = copy.deepcopy(temp_model.get_xml())
        del temp_model
        del temp_xml_1

        temp_xml_2 = self.add_mocap_to_xml(temp_xml_2)

        self.mjpy_model = mujoco_py.load_model_from_xml(temp_xml_2)
        del temp_xml_2

        gc.collect()
        self.sim = mujoco_py.MjSim(self.mjpy_model)
        self.setup_viewer()

        body_id = self.sim.model.body_name2id(self.max_corner_name)
        self.ee_mocap_id = self.sim.model.body_mocapid[body_id]

        self.joint_indexes = [self.sim.model.joint_name2id(
            joint) for joint in self.joints]
        self.joint_pos_addr = [self.sim.model.get_joint_qpos_addr(
            joint) for joint in self.joints]
        self.joint_vel_addr = [self.sim.model.get_joint_qvel_addr(
            joint) for joint in self.joints]
        self.ee_site_adr = mujoco_py.functions.mj_name2id(
            self.sim.model, 6, "grip_site")

        # Sets robot to initial qpos and resets osc values
        self.set_robot_initial_joints()
        self.reset_osc_values()
        self.update_osc_values()

        for _ in range(30):
            mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
            self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr]
            mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)

        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.initial_qfrc_applied = self.sim.data.qfrc_applied[self.joint_vel_addr].copy(
        )
        self.initial_qfrc_bias = self.sim.data.qfrc_bias[self.joint_vel_addr].copy(
        )

    def set_robot_initial_joints(self):
        for j, joint in enumerate(self.joints):
            self.sim.data.set_joint_qpos(joint, self.initial_qpos[j])

        for _ in range(10):
            self.sim.forward()

    def run_controller(self):
        tau = osc_binding.step_controller(self.initial_O_T_EE,
                                          self.O_T_EE,
                                          self.initial_joint_osc,
                                          self.joint_pos_osc,
                                          self.joint_vel_osc,
                                          self.mass_matrix_osc,
                                          self.jac_osc,
                                          np.zeros(7),
                                          self.tau_J_d_osc,
                                          self.desired_pos_ctrl_W,
                                          np.zeros(3),
                                          self.delta_tau_max,
                                          self.kp,
                                          self.kp,
                                          self.damping_ratio
                                          )
        torques = tau.flatten()
        return torques

    def step_env(self):
        self.sim.data.mocap_pos[self.ee_mocap_id][:] = self.sim.data.get_geom_xpos(
            "grip_geom")

        mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
        self.update_osc_values()
        tau = self.run_controller()
        self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr] + tau
        mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)

    def step(self, action):
        raw_action = action.copy()
        action = raw_action*self.output_max

        image_obs_substep_idx_mean = self.image_obs_noise_mean * \
            (self.substeps-1)
        image_obs_substep_idx = int(np.random.normal(
            image_obs_substep_idx_mean, self.image_obs_noise_std))
        image_obs_substep_idx = np.clip(
            image_obs_substep_idx, 0, self.substeps-1)

        cosine_distance = compute_cosine_distance(
            self.previous_raw_action, raw_action)
        self.previous_raw_action = raw_action

        previous_desired_pos_step_W = self.desired_pos_step_W.copy()
        desired_pos_step_W = previous_desired_pos_step_W + action
        self.desired_pos_step_W = np.clip(
            desired_pos_step_W, self.min_absolute_W, self.max_absolute_W)

        for i in range(int(self.substeps)):
            for j in range(int(self.between_steps)):
                self.desired_pos_ctrl_W = self.filter*self.desired_pos_step_W + \
                    (1-self.filter)*self.desired_pos_ctrl_W
            self.step_env()
            if i == image_obs_substep_idx:
                image_obs = self.get_image_obs()
                self.frame_stack.append(image_obs)
                flattened_corners = self.post_action_image_capture()

        obs = self.get_obs()
        reward, done, info = self.post_action(obs, raw_action, cosine_distance)
        info['corner_positions'] = flattened_corners
        return obs, reward, done, info

    def compute_task_reward(self, achieved_goal, desired_goal, info):
        return self.task_reward_function(achieved_goal, desired_goal, info)

    def post_action_image_capture(self):
        camera_matrix, camera_transformation = self.get_camera_matrices(
            self.train_camera, self.image_size[0], self.image_size[1])
        corners_in_image = self.get_corner_image_positions(
            self.image_size[0], self.image_size[0], camera_matrix, camera_transformation)
        flattened_corners = []
        for corner in corners_in_image:
            flattened_corners.append(corner[0]/self.image_size[0])
            flattened_corners.append(corner[1]/self.image_size[1])
        flattened_corners = np.array(flattened_corners)

        return flattened_corners

    def get_corner_constraint_distances(self):
        inv_corner_index_mapping = {v: k for k,
                                    v in self.corner_index_mapping.items()}
        distances = {"0": 0, "1": 0, "2": 0, "3": 0}
        for i, contraint in enumerate(self.constraints):
            if contraint['origin'] in inv_corner_index_mapping.keys():
                origin_pos = self.sim.data.get_site_xpos(
                    contraint['origin']).copy() - self.relative_origin
                target_pos = self.goal[i *
                                       self.single_goal_dim:(i+1)*self.single_goal_dim]
                distances[inv_corner_index_mapping[contraint['origin']]
                          ] = np.linalg.norm(origin_pos-target_pos)
        return distances

    def post_action(self, obs, raw_action, cosine_distance):
        reward = self.compute_task_reward(np.reshape(
            obs['achieved_goal'], (1, -1)), np.reshape(self.goal, (1, -1)), dict())[0]
        is_success = reward > self.fail_reward

        delta_size = np.linalg.norm(raw_action)
        ctrl_error = np.linalg.norm(
            self.desired_pos_ctrl_W - self.get_ee_position_W())

        if is_success and self.episode_ee_close_steps == 0:
            logger.debug(
                f"Successful fold, reward: {np.round(reward, decimals=3)}")

        env_memory_usage = self.process.memory_info().rss
        info = {
            "reward": reward,
            'is_success': is_success,
            "delta_size": delta_size,
            "ctrl_error": ctrl_error,
            "env_memory_usage": env_memory_usage,
            "corner_sum_error": 0
        }

        constraint_distances = self.get_corner_constraint_distances()

        for key in constraint_distances.keys():
            info[f"corner_{key}"] = constraint_distances[key]
            info["corner_sum_error"] += constraint_distances[key]

        done = False

        if constraint_distances["1"] < self.success_distance:
            self.episode_ee_close_steps += 1
        else:
            self.episode_ee_close_steps = 0

        if self.episode_ee_close_steps >= self.max_close_steps:
            done = True

        return reward, done, info

    def update_osc_values(self):
        self.joint_pos_osc = np.ndarray(shape=(7,), dtype=np.float64)
        self.joint_vel_osc = np.ndarray(shape=(7,), dtype=np.float64)
        self.O_T_EE = np.ndarray(shape=(16,), dtype=np.float64)
        self.jac_osc = np.ndarray(shape=(42,), dtype=np.float64)
        self.mass_matrix_osc = np.ndarray(shape=(49,), dtype=np.float64)
        self.tau_J_d_osc = self.sim.data.qfrc_applied[self.joint_vel_addr] - \
            self.sim.data.qfrc_bias[self.joint_vel_addr]

        L = len(self.sim.data.qvel)
        p = self.sim.data.site_xpos[self.ee_site_adr]
        R = self.sim.data.site_xmat[self.ee_site_adr].reshape(
            [3, 3]).T  # SAATANA

        self.O_T_EE[0] = R[0, 0]
        self.O_T_EE[1] = R[0, 1]
        self.O_T_EE[2] = R[0, 2]
        self.O_T_EE[3] = 0.0

        self.O_T_EE[4] = R[1, 0]
        self.O_T_EE[5] = R[1, 1]
        self.O_T_EE[6] = R[1, 2]
        self.O_T_EE[7] = 0.0

        self.O_T_EE[8] = R[2, 0]
        self.O_T_EE[9] = R[2, 1]
        self.O_T_EE[10] = R[2, 2]
        self.O_T_EE[11] = 0.0

        self.O_T_EE[12] = p[0]
        self.O_T_EE[13] = p[1]
        self.O_T_EE[14] = p[2]
        self.O_T_EE[15] = 1.0

        for j in range(7):
            self.joint_pos_osc[j] = self.sim.data.qpos[self.joint_pos_addr[j]].copy(
            )
            self.joint_vel_osc[j] = self.sim.data.qvel[self.joint_vel_addr[j]].copy(
            )

        jac_pos_osc = np.ndarray(shape=(L*3,), dtype=np.float64)
        jac_rot_osc = np.ndarray(shape=(L*3,), dtype=np.float64)
        mujoco_py.functions.mj_jacSite(
            self.sim.model, self.sim.data, jac_pos_osc, jac_rot_osc, self.ee_site_adr)

        for j in range(7):
            for r in range(6):
                if (r < 3):
                    value = jac_pos_osc[L*r + self.joint_pos_addr[j]]
                else:
                    value = jac_rot_osc[L*(r-3) + self.joint_pos_addr[j]]
                self.jac_osc[j*6 + r] = value

        mass_array_osc = np.ndarray(shape=(L ** 2,), dtype=np.float64)
        mujoco_py.cymj._mj_fullM(
            self.sim.model, mass_array_osc, self.sim.data.qM)

        for c in range(7):
            for r in range(7):
                self.mass_matrix_osc[c*7 + r] = mass_array_osc[self.joint_pos_addr[r]
                                                               * L + self.joint_pos_addr[c]]

        if self.initial_O_T_EE is None:
            self.initial_O_T_EE = self.O_T_EE.copy()
        if self.initial_joint_osc is None:
            self.initial_joint_osc = self.joint_pos_osc.copy()

        if self.desired_pos_ctrl_W is None:
            self.desired_pos_ctrl_W = p.copy()
        if self.desired_pos_step_W is None:
            self.desired_pos_step_W = p.copy()
        if self.initial_ee_p_W is None:
            self.initial_ee_p_W = p.copy()
            self.min_absolute_W = self.initial_ee_p_W + self.limits_min
            self.max_absolute_W = self.initial_ee_p_W + self.limits_max

    def get_trajectory_log_entry(self):
        entry = {
            'origin': self.relative_origin,
            'output_max': self.output_max,
            'desired_pos_step_I': self.desired_pos_step_W - self.relative_origin,
            'desired_pos_ctrl_I': self.desired_pos_ctrl_W - self.relative_origin,
            'ee_position_I': self.get_ee_position_I(),
            'raw_action': self.previous_raw_action,
            'substeps': self.substeps,
            'timestep': self.timestep,
            'goal_noise': self.goal_noise
        }
        return entry

    def get_ee_position_W(self):
        return self.sim.data.get_site_xpos(self.ee_site_name).copy()

    def get_ee_position_I(self):
        return self.sim.data.get_site_xpos(self.ee_site_name).copy() - self.relative_origin

    def get_joint_positions(self):
        positions = [self.sim.data.get_joint_qpos(
            joint).copy() for joint in self.joints]
        return np.array(positions)

    def get_joint_velocities(self):
        velocities = [self.sim.data.get_joint_qvel(
            joint).copy() for joint in self.joints]
        return np.array(velocities)

    def get_ee_velocity(self):
        return self.sim.data.get_site_xvelp(self.ee_site_name).copy()

    def get_cloth_position_I(self):
        positions = dict()
        for site in self.cloth_site_names:
            positions[site] = self.sim.data.get_site_xpos(
                site).copy() - self.relative_origin
        return positions

    def get_cloth_position_W(self):
        positions = dict()
        for site in self.cloth_site_names:
            positions[site] = self.sim.data.get_site_xpos(site).copy()
        return positions

    def get_cloth_edge_positions_W(self):
        positions = dict()
        for i in range(9):
            for j in range(9):
                if (i in [0, 8]) or (j in [0, 8]):
                    site_name = f"S{i}_{j}"
                    positions[site_name] = self.sim.data.get_site_xpos(
                        site_name).copy()
        return positions

    def get_cloth_velocity(self):
        velocities = dict()
        for site in self.cloth_site_names:
            velocities[site] = self.sim.data.get_site_xvelp(site).copy()
        return velocities

    def get_image_obs(self):
        camera_id = self.sim.model.camera_name2id(
            self.train_camera)
        width = self.randomization_kwargs['camera_config']['width']
        height = self.randomization_kwargs['camera_config']['height']

        self.viewer.render(width, height, camera_id)
        image_obs = copy.deepcopy(
            self.viewer.read_pixels(width, height, depth=False))

        image_obs = image_obs[::-1, :, :]

        height_start = int(image_obs.shape[0]/2 - self.image_size[1]/2)
        height_end = height_start + self.image_size[1]

        width_start = int(image_obs.shape[1]/2 - self.image_size[0]/2)
        width_end = width_start + self.image_size[0]
        image_obs = image_obs[height_start:height_end,
                              width_start:width_end, :]

        if self.randomization_kwargs['albumentations_randomization']:
            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2RGB)
            image_obs = self.albumentations_transform(image=image_obs)["image"]
            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_RGB2GRAY)
        else:
            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)

        return (image_obs / 255).flatten().copy()

    def get_obs(self):
        achieved_goal_I = np.zeros(self.single_goal_dim*len(self.constraints))
        for i, constraint in enumerate(self.constraints):
            origin = constraint['origin']
            achieved_goal_I[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = self.sim.data.get_site_xpos(
                origin).copy() - self.relative_origin

        cloth_position = np.array(list(self.get_cloth_position_I().values()))
        cloth_velocity = np.array(list(self.get_cloth_velocity(
        ).values()))

        cloth_observation = np.concatenate(
            [cloth_position.flatten(), cloth_velocity.flatten()])

        desired_pos_ctrl_I = self.desired_pos_ctrl_W - self.relative_origin

        full_observation = {
            'achieved_goal': achieved_goal_I.copy(), 'desired_goal': self.goal.copy()}

        if self.robot_observation == "ee":
            robot_observation = np.concatenate(
                [self.get_ee_position_I(), self.get_ee_velocity(), desired_pos_ctrl_I])
        elif self.robot_observation == "ctrl":
            robot_observation = np.concatenate(
                [self.previous_raw_action, np.zeros(6)])
        elif self.robot_observation == "none":
            robot_observation = np.zeros(9)
        full_observation['image'] = np.array(
            [image for image in self.frame_stack]).flatten()
        if self.randomization_kwargs["dynamics_randomization"]:
            full_observation['observation'] = np.concatenate(
                [cloth_observation.copy(), np.array(self.mujoco_model_numerical_values)])
        else:
            full_observation['observation'] = cloth_observation.copy(
            ).flatten()
        full_observation['robot_observation'] = robot_observation.flatten(
        ).copy()

        return full_observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_goal_I(self):
        goal = np.zeros(self.single_goal_dim*len(self.constraints))
        noise = self.np_random.uniform(self.goal_noise_range[0],
                                       self.goal_noise_range[1])

        for i, constraint in enumerate(self.constraints):
            target = constraint['target']
            target_pos = self.sim.data.get_site_xpos(target).copy()
            offset = np.zeros(self.single_goal_dim)
            if 'noise_directions' in constraint.keys():
                for idx, offset_dir in enumerate(constraint['noise_directions']):
                    offset[idx] = offset_dir*noise

            goal[i*self.single_goal_dim: (i+1) *
                 self.single_goal_dim] = target_pos + offset - self.relative_origin

        return goal.copy(), noise

    def reset_osc_values(self):
        self.initial_O_T_EE = None
        self.initial_joint_osc = None
        self.initial_ee_p_W = None
        self.desired_pos_step_W = None
        self.desired_pos_ctrl_W = None
        self.previous_raw_action = np.zeros(3)
        self.raw_action = None

    def setup_xml_model(self, randomize, rownum=None):
        model_kwargs, model_numerical_values = self.build_xml_kwargs_and_numerical_values(
            randomize=randomize, rownum=rownum)
        self.mujoco_model_numerical_values = model_numerical_values
        self.setup_initial_state_and_sim(model_kwargs)

    def reset(self):
        self.sim.reset()
        self.sim.set_state(self.initial_state)
        self.sim.data.qfrc_applied[self.joint_vel_addr] = self.initial_qfrc_applied
        self.sim.data.qfrc_bias[self.joint_vel_addr] = self.initial_qfrc_bias
        self.sim.forward()  # BODY POSITIONS CORRECT
        self.reset_camera()
        self.sim.forward()  # CAMERA CHANGES CORRECT
        self.reset_osc_values()
        self.update_osc_values()

        self.relative_origin = self.get_ee_position_W()
        self.goal, self.goal_noise = self.sample_goal_I()

        if not self.viewer is None:
            del self.viewer._markers[:]

        self.episode_ee_close_steps = 0

        image_obs = self.get_image_obs()
        for _ in range(self.frame_stack_size):
            self.frame_stack.append(image_obs)

        q_ok = np.allclose(self.initial_qpos,
                           self.get_joint_positions(), rtol=0.01, atol=0.01)

        return self.get_obs()

    def get_corner_image_positions(self, w, h, camera_matrix, camera_transformation):
        corners = []
        cloth_positions = self.get_cloth_position_W()
        for site in self.corner_index_mapping.values():
            corner_in_image = np.ones(4)
            corner_in_image[:3] = cloth_positions[site]
            corner = (camera_matrix @ camera_transformation) @ corner_in_image
            u_c, v_c, _ = corner/corner[2]
            corner = [w-u_c, v_c]
            corners.append(corner)
        return corners

    def get_edge_image_positions(self, w, h, camera_matrix, camera_transformation):
        corners = []
        cloth_edge_positions = self.get_cloth_edge_positions_W()
        for site in cloth_edge_positions.keys():
            corner_in_image = np.ones(4)
            corner_in_image[:3] = cloth_edge_positions[site]
            corner = (camera_matrix @ camera_transformation) @ corner_in_image
            u_c, v_c, _ = corner/corner[2]
            corner = [w-u_c, v_c]
            corners.append(corner)
        return corners

    def get_camera_matrices(self, camera_name, w, h):
        camera_id = self.sim.model.camera_name2id(camera_name)
        fovy = self.sim.model.cam_fovy[camera_id]
        f = 0.5 * h / math.tan(fovy * math.pi / 360)
        camera_matrix = np.array(((f, 0, w / 2), (0, f, h / 2), (0, 0, 1)))
        xmat = self.sim.data.get_camera_xmat(camera_name)
        xpos = self.sim.data.get_camera_xpos(camera_name)

        camera_transformation = np.eye(4)
        camera_transformation[:3, :3] = xmat
        camera_transformation[:3, 3] = xpos
        camera_transformation = np.linalg.inv(camera_transformation)[:3, :]

        return camera_matrix, camera_transformation

    def get_masked_image(self, camera, width, height, ee_in_image, aux_output, point_size, greyscale=False, mask_type=None):
        camera_matrix, camera_transformation = self.get_camera_matrices(
            camera, width, height)
        camera_id = self.sim.model.camera_name2id(camera)
        self.viewer.render(width, height, camera_id)
        data = np.float32(self.viewer.read_pixels(
            width, height, depth=False)).copy()
        data = np.float32(data[::-1, :, :]).copy()
        data = np.float32(data)
        if greyscale:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        ee = camera_matrix @ camera_transformation @ ee_in_image
        u_ee, v_ee, _ = ee/ee[2]
        # cv2.circle(data, (width-int(u_ee), int(v_ee)), point_size, (0, 0, 0), -1)

        if mask_type == "corners":
            mask = self.get_corner_image_positions(
                width, height, camera_matrix, camera_transformation)
        elif mask_type == "edges":
            mask = self.get_edge_image_positions(
                width, height, camera_matrix, camera_transformation)
        else:
            mask = []

        for point in mask:
            u = int(point[0])
            v = int(point[1])
            cv2.circle(data, (u, v), point_size, (255, 0, 0), -1)

        if not aux_output is None:
            for aux_idx in range(4):
                aux_u = int(aux_output.flatten()[aux_idx*2]*width)
                aux_v = int(aux_output.flatten()[aux_idx*2+1]*height)
                cv2.circle(data, (aux_u, aux_v), point_size, (0, 255, 0), -1)

        return data

    def capture_images(self, aux_output=None, mask_type="corners"):
        w_eval, h_eval = 500, 500
        w_corners, h_corners = 500, 500
        w_cnn, h_cnn = self.image_size
        w_cnn_full, h_cnn_full = self.randomization_kwargs['camera_config'][
            'width'], self.randomization_kwargs['camera_config']['height']

        ee_in_image = np.ones(4)
        ee_pos = self.get_ee_position_W()
        ee_in_image[:3] = ee_pos

        corner_image = self.get_masked_image(
            self.train_camera, w_corners, h_corners, ee_in_image, aux_output, 8, greyscale=False, mask_type=mask_type)
        eval_image = self.get_masked_image(
            self.eval_camera, w_eval, h_eval, ee_in_image, None, 4, greyscale=False, mask_type=mask_type)
        cnn_color_image_full = self.get_masked_image(
            self.train_camera, w_cnn_full, h_cnn_full, ee_in_image, aux_output, 2, mask_type=mask_type)
        cnn_color_image = self.get_masked_image(
            self.train_camera, w_cnn, h_cnn, ee_in_image, aux_output, 2, mask_type=mask_type)
        cnn_image = self.get_masked_image(
            self.train_camera, w_cnn, h_cnn, ee_in_image, aux_output, 2, greyscale=True, mask_type=mask_type)

        return corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image


class ClothEnv(ClothEnv_, EzPickle):
    def __init__(self, **kwargs):
        ClothEnv_.__init__(
            self, **kwargs)
        EzPickle.__init__(self)
