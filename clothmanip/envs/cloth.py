import mujoco_py
import osc_binding
import cv2
import numpy as np
import copy
from gym.utils import seeding, EzPickle
from clothmanip.utils import reward_calculation
import gym
import time
import clothmanip.utils.utils as utils
import os
from scipy import spatial
from clothmanip.envs.template_renderer import TemplateRenderer
import math
from collections import deque
import copy
from clothmanip.utils import mujoco_model_kwargs
from shutil import copyfile
import psutil
import gc


def compute_cosine_distance(vec1, vec2):
    if np.all(vec1==0) or np.all(vec2==0):
        return 0
    else:
        cosine_dist = spatial.distance.cosine(vec1, vec2)
        if np.isnan(cosine_dist):
            return 0
        else:
            return cosine_dist



class ClothEnv(object):
    def __init__(
        self,
        timestep,
        cloth_type,
        sparse_dense,
        goal_noise_range,
        pixels,
        image_size,
        depth_frames,
        frame_stack_size,
        output_max,
        constant_goal,
        constraints,
        success_reward,
        fail_reward,
        extra_reward,
        kp,
        damping_ratio,
        control_frequency,
        ctrl_filter,
        max_episode_steps,
        ate_penalty_coef,
        action_norm_penalty_coef,
        cosine_penalty_coef,
        num_eval_rollouts,
        save_folder,
        randomization_kwargs,
        robot_observation,
        camera_type,
        camera_config,
        image_obs_noise_mean=1,
        image_obs_noise_std=0,
        has_viewer=False
    ):  
        self.process = psutil.Process(os.getpid())
        self.randomization_kwargs = randomization_kwargs       
        file_dir = os.path.dirname(os.path.abspath(__file__))
        source = os.path.join(file_dir, "mujoco_templates", "arena.xml")
        destination = os.path.join(save_folder, "arena.xml")
        copyfile(source, destination)
        self.template_renderer = TemplateRenderer(save_folder)
        self.save_folder = save_folder
        self.num_eval_rollouts = num_eval_rollouts
        self.ate_penalty_coef = ate_penalty_coef
        self.action_norm_penalty_coef = action_norm_penalty_coef
        self.cosine_penalty_coef = cosine_penalty_coef
        self.filter = ctrl_filter
        self.kp = kp
        self.damping_ratio = damping_ratio
        self.output_max = output_max
        self.success_reward = success_reward
        self.fail_reward = fail_reward
        self.extra_reward = extra_reward
        self.sparse_dense = sparse_dense
        self.constraints = constraints
        self.goal_noise_range = goal_noise_range
        self.pixels = pixels
        self.depth_frames = depth_frames
        self.frame_stack_size = frame_stack_size
        self.image_size = (image_size, image_size)
        self.constant_goal = constant_goal
        self.has_viewer = has_viewer
        self.max_episode_steps = max_episode_steps
        self.image_obs_noise_mean = image_obs_noise_mean
        self.image_obs_noise_std = image_obs_noise_std
        self.robot_observation = robot_observation
        self.camera_type = camera_type

        self.max_success_steps = 20
        self.frame_stack = deque([], maxlen = self.frame_stack_size)

        self.limits_min = [-0.35, -0.35, 0.0]
        self.limits_max = [0.05, 0.05, 0.4] 

        self.single_goal_dim = 3
        self.timestep = timestep
        self.control_frequency = control_frequency

        steps_per_second = 1 / self.timestep
        self.substeps = 1 / (self.timestep*self.control_frequency)
        self.between_steps = 1000 / steps_per_second
        self.delta_tau_max = 1000 / steps_per_second

        if camera_type == "up":
            self.train_camera = "train_camera_up"
        elif camera_type == "front":
            self.train_camera = "train_camera_front"
        else:
            self.train_camera = "train_camera_side"
        self.eval_camera = "eval_camera"
        self.camera_config = camera_config     
        self.episode_success_steps = 0

        self.seed()

        #sideways
        self.initial_qpos = np.array([0.212422, 0.362907, -0.00733391, -1.9649, -0.0198034, 2.37451, -1.50499])

        #diagonal
        #self.initial_qpos = np.array([0.114367, 0.575019, 0.0550664, -1.60919, -0.079246, 2.23369, -1.56064])
        #self.initial_des_pos = np.array([0.692898, 0.107481, 0.194496])

        self.cloth_site_names = []
        for i in [0, 4, 8]:
            for j in [0, 4, 8]:
                self.cloth_site_names.append(f"S{i}_{j}")
        self.corner_index_mapping = {"S0_8" : 0, "S8_8": 1, "S0_0": 2, "S8_0": 3}
        self.corner_indices = [] #TODO: fix ghetto
        for site in self.cloth_site_names:
            if not "4" in site:
                self.corner_indices.append(self.corner_index_mapping[site])

        self.ee_site_name = 'grip_site'
        self.joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

        self.mjpy_model = None
        self.sim = None
        self.viewer = None

        self.cloth_type = cloth_type
        model_kwargs, model_numerical_values = self.build_xml_kwargs_and_numerical_values()
        self.mujoco_model_numerical_values = model_numerical_values
        self.setup_initial_state_and_sim(model_kwargs)
        self.dump_xml_models()
        
        self.task_reward_function = reward_calculation.get_task_reward_function(
            self.constraints, self.single_goal_dim, self.sparse_dense, self.success_reward, self.fail_reward, self.extra_reward)

        self.action_space = gym.spaces.Box(-1,
                                           1, shape=(3,), dtype='float32')

        self.relative_origin = self.get_ee_position_W()
        self.goal = self.sample_goal_I()
        
        if self.pixels:
            image_obs = self.get_image_obs()
            for _ in range(self.frame_stack_size):
                self.frame_stack.append(image_obs)

        obs = self.get_obs()

        if self.pixels:
            self.observation_space = gym.spaces.Dict(dict(
                desired_goal=gym.spaces.Box(-np.inf, np.inf,
                                            shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=gym.spaces.Box(-np.inf, np.inf,
                                             shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=gym.spaces.Box(-np.inf, np.inf,
                                           shape=obs['observation'].shape, dtype='float32'),
                robot_observation=gym.spaces.Box(-np.inf, np.inf,
                                                 shape=obs['robot_observation'].shape, dtype='float32'),
                model_params=gym.spaces.Box(-np.inf, np.inf,
                                            shape=obs['model_params'].shape, dtype='float32'),
                image=gym.spaces.Box(-np.inf, np.inf,
                                     shape=obs['image'].shape, dtype='float32')
            ))
        else:
            self.observation_space = gym.spaces.Dict(dict(
                desired_goal=gym.spaces.Box(-np.inf, np.inf,
                                            shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=gym.spaces.Box(-np.inf, np.inf,
                                             shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=gym.spaces.Box(-np.inf, np.inf,
                                           shape=obs['observation'].shape, dtype='float32'),
                robot_observation=gym.spaces.Box(-np.inf, np.inf,
                                                 shape=obs['robot_observation'].shape, dtype='float32'),
                model_params=gym.spaces.Box(-np.inf, np.inf,
                                            shape=obs['model_params'].shape, dtype='float32')
            ))

    def build_xml_kwargs_and_numerical_values(self):
        model_kwargs = dict()
        wipe = mujoco_model_kwargs.WIPE_MODEL_KWARGS
        bath = mujoco_model_kwargs.BATH_MODEL_KWARGS
        kitchen = mujoco_model_kwargs.KITCHEN_MODEL_KWARGS

        #Dynamics
        model_numerical_values = [0]
        if self.randomization_kwargs['dynamics_randomization']:
            for key in mujoco_model_kwargs.model_kwarg_ranges.keys():
                val_min = np.min(np.array([wipe[key], bath[key], kitchen[key]]))
                val_max = np.max(np.array([wipe[key], bath[key], kitchen[key]]))
                val = np.random.uniform(val_min, val_max)
                model_kwargs[key] = val
                model_numerical_values.append(val)
            for key in mujoco_model_kwargs.model_kwarg_choices.keys():
                val = np.random.choice(mujoco_model_kwargs.model_kwarg_choices[key])
                model_kwargs[key] = val

        elif self.cloth_type == "bath":
            model_kwargs = bath
        elif self.cloth_type == "wipe": 
            model_kwargs = wipe
        elif self.cloth_type == "kitchen":
            model_kwargs = kitchen

        #General
        model_kwargs['timestep'] = self.timestep
        model_kwargs['cloth_texture_randomization'] = self.randomization_kwargs['cloth_texture_randomization']
        model_kwargs['background_texture_randomization'] = self.randomization_kwargs['background_texture_randomization']
        model_kwargs['lights_randomization'] = self.randomization_kwargs['lights_randomization']
        model_kwargs['robot_appearance_randomization'] = self.randomization_kwargs['robot_appearance_randomization']
        model_kwargs['finger_collisions'] = self.randomization_kwargs['finger_collisions']

        model_kwargs['train_camera_fovy'] = (self.camera_config['fovy_range'][0] + self.camera_config['fovy_range'][1])/2
        model_kwargs['num_lights'] = 2
        
        #Appearance
        appearance_choices = mujoco_model_kwargs.appearance_kwarg_choices
        appearance_ranges = mujoco_model_kwargs.appearance_kwarg_ranges
        for key in appearance_choices.keys():
            model_kwargs[key] = np.random.choice(appearance_choices[key])
        for key in appearance_ranges.keys():
            values = appearance_ranges[key]
            model_kwargs[key] = np.random.uniform(values[0], values[1])

        #Camera fovy
        if self.randomization_kwargs['camera_randomization']:
            model_kwargs['train_camera_fovy'] = np.random.uniform(self.camera_config['fovy_range'][0], self.camera_config['fovy_range'][1])
            #model_kwargs['num_lights'] = np.random.randint(0, 4)
            #TODO Fig out number of ligths
        
        return model_kwargs, model_numerical_values

    def setup_viewer(self):
        if self.has_viewer:
            if not self.viewer is None:
                #print_refs(self.viewer)
                del self.viewer
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self.viewer.vopt.geomgroup[0] = 0
            self.viewer.vopt.geomgroup[1] = 1
            

    def dump_xml_models(self):
        with open(f"{self.save_folder}/compiled_mujoco_model_no_inertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=False)

        with open(f"{self.save_folder}/compiled_mujoco_model_with_intertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=True)

    
    def reset_camera(self, randomize=False, radius=0):
        lookat_offset = np.zeros(3)
        if randomize:
            lookat_offset[0] += np.random.uniform(-radius, radius)
            lookat_offset[1] += np.random.uniform(-radius, radius)
        des_cam_look_pos = self.sim.data.get_body_xpos("B4_4").copy() + lookat_offset
        if self.train_camera == "train_camera_up":
            cam_scale = 1
            des_cam_pos = des_cam_look_pos + cam_scale * (np.array([0.52536418, -0.60,  1.03])-des_cam_look_pos)
        elif self.train_camera == "train_camera_front":
            des_cam_pos = np.array([0.5, -0.8, 0.69])
        else:
            if self.camera_config['type'] == "small":
                cam_scale = 1.6
            else:
                cam_scale = 0.4
            des_cam_pos = des_cam_look_pos + cam_scale * (np.array([-0.0, -0.312,  0.455])-des_cam_look_pos)
        cam_id = self.sim.model.camera_name2id(self.train_camera)
        #print("desired camera position", des_cam_pos)
        self.mjpy_model.cam_pos[cam_id] = des_cam_pos
        self.sim.data.set_mocap_pos("lookatbody", des_cam_look_pos)

    def setup_initial_state_and_sim(self, model_kwargs):
        xml = self.template_renderer.render_template("arena.xml", **model_kwargs)
        if not self.mjpy_model is None:
            del self.mjpy_model
        if not self.sim is None:
            del self.sim 
        self.mjpy_model = mujoco_py.load_model_from_xml(xml)
        del xml

        gc.collect()
        self.sim = mujoco_py.MjSim(self.mjpy_model)
        self.setup_viewer()
        utils.remove_distance_welds(self.sim)

        self.joint_indexes = [self.sim.model.joint_name2id(joint) for joint in self.joints]
        self.joint_pos_addr = [self.sim.model.get_joint_qpos_addr(joint) for joint in self.joints]
        self.joint_vel_addr = [self.sim.model.get_joint_qvel_addr(joint) for joint in self.joints]
        self.ee_site_adr = mujoco_py.functions.mj_name2id(self.sim.model, 6, "grip_site")
        
        #Sets robot to initial qpos and resets osc values
        self.set_robot_initial_joints()
        self.reset_osc_values()
        self.update_osc_values()

        for _ in range(30):
            mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
            self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr]
            mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)

        self.reset_camera()
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.initial_qfrc_applied = self.sim.data.qfrc_applied[self.joint_vel_addr].copy()
        self.initial_qfrc_bias = self.sim.data.qfrc_bias[self.joint_vel_addr].copy()


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
        mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
        self.update_osc_values()
        tau = self.run_controller()
        self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr] + tau
        mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)

    def step(self, action):
        raw_action = action.copy()
        action = raw_action*self.output_max
        if self.pixels:
            image_obs_substep_idx_mean = self.image_obs_noise_mean * (self.substeps-1)
            image_obs_substep_idx = int(np.random.normal(image_obs_substep_idx_mean, self.image_obs_noise_std))
            image_obs_substep_idx = np.clip(image_obs_substep_idx, 0, self.substeps-1)

        cosine_distance = compute_cosine_distance(self.previous_raw_action, raw_action)
        self.previous_raw_action = raw_action

        previous_desired_pos_step_W = self.desired_pos_step_W.copy()
        desired_pos_step_W = previous_desired_pos_step_W + action
        self.desired_pos_step_W = np.clip(desired_pos_step_W, self.min_absolute_W, self.max_absolute_W)

        for i in range(int(self.substeps)):
            for j in range(int(self.between_steps)):
                self.desired_pos_ctrl_W = self.filter*self.desired_pos_step_W + (1-self.filter)*self.desired_pos_ctrl_W
            self.step_env()
            if self.pixels and i == image_obs_substep_idx:
                image_obs = self.get_image_obs()
                self.frame_stack.append(image_obs)
                flattened_corners, corner_indices = self.post_action_image_capture()

        obs = self.get_obs()
        
        reward, done, info = self.post_action(obs, raw_action, cosine_distance)
        info['corner_positions'] = flattened_corners
        info['corner_indices'] = corner_indices

        return obs, reward, done, info


    def compute_task_reward(self, achieved_goal, desired_goal, info):
        return self.task_reward_function(achieved_goal, desired_goal, info)

    def post_action_image_capture(self):
        camera_matrix, camera_transformation = self.get_camera_matrices(self.train_camera, self.image_size[0], self.image_size[1])
        corners_in_image, corner_indices = self.get_corner_image_positions(self.image_size[0], self.image_size[0], camera_matrix, camera_transformation)
        flattened_corners = []
        for corner in corners_in_image:
            flattened_corners.append(corner[0]/self.image_size[0])
            flattened_corners.append(corner[1]/self.image_size[1])
        flattened_corners = np.array(flattened_corners)

        return flattened_corners, corner_indices


        
    def post_action(self, obs, raw_action, cosine_distance):
        task_reward = self.compute_task_reward(np.reshape(obs['achieved_goal'], (1, -1)), np.reshape(self.goal, (1, -1)), dict())[0]
        is_success = task_reward > self.fail_reward


        delta_size_penalty = np.linalg.norm(raw_action)
        scaled_delta_size_penalty = -delta_size_penalty*self.action_norm_penalty_coef

        ate_penalty = np.linalg.norm(self.desired_pos_ctrl_W - self.get_ee_position_W())
        scaled_ate_penalty = -ate_penalty*self.ate_penalty_coef

        cosine_penalty = cosine_distance
        scaled_cosine_penalty = -cosine_penalty*self.cosine_penalty_coef

        control_penalty =  scaled_delta_size_penalty + scaled_ate_penalty + scaled_cosine_penalty

        reward = task_reward + control_penalty

        if is_success and self.episode_success_steps == 0:
            print("Sim success",
                np.round(reward, decimals=3),
                np.round(task_reward, decimals=3),
                np.round(control_penalty, decimals=3))
            

         
        env_memory_usage = self.process.memory_info().rss
        info = {
                "task_reward": task_reward,
                "reward": reward,
                'is_success': is_success,

                "delta_size_penalty": delta_size_penalty,
                "scaled_delta_size_penalty": scaled_delta_size_penalty,

                "cosine_penalty" : cosine_penalty,
                "scaled_cosine_penalty": scaled_cosine_penalty,

                "ate_penalty": ate_penalty,
                "scaled_ate_penalty": scaled_ate_penalty,

                "control_penalty": control_penalty,
                "env_memory_usage": env_memory_usage
                }

        done = False
        self.episode_success_steps += int(is_success)
        if self.episode_success_steps >= self.max_success_steps:
            done = True

        return reward, done, info
        


    def update_osc_values(self):
        self.joint_pos_osc = np.ndarray(shape=(7,), dtype=np.float64)
        self.joint_vel_osc = np.ndarray(shape=(7,), dtype=np.float64)
        self.O_T_EE = np.ndarray(shape=(16,), dtype=np.float64)
        self.jac_osc = np.ndarray(shape=(42,), dtype=np.float64)
        self.mass_matrix_osc = np.ndarray(shape=(49,), dtype=np.float64)
        self.tau_J_d_osc = self.sim.data.qfrc_applied[self.joint_vel_addr] - self.sim.data.qfrc_bias[self.joint_vel_addr]
        
        L = len(self.sim.data.qvel)        
        p = self.sim.data.site_xpos[self.ee_site_adr]
        R = self.sim.data.site_xmat[self.ee_site_adr].reshape([3, 3]).T #SAATANA

        self.O_T_EE[0] = R[0,0]
        self.O_T_EE[1] = R[0,1]
        self.O_T_EE[2] = R[0,2]
        self.O_T_EE[3] = 0.0

        self.O_T_EE[4] = R[1,0]
        self.O_T_EE[5] = R[1,1]
        self.O_T_EE[6] = R[1,2]
        self.O_T_EE[7] = 0.0

        self.O_T_EE[8] = R[2,0]
        self.O_T_EE[9] = R[2,1]
        self.O_T_EE[10] = R[2,2]
        self.O_T_EE[11] = 0.0

        self.O_T_EE[12] = p[0]
        self.O_T_EE[13] = p[1]
        self.O_T_EE[14] = p[2]
        self.O_T_EE[15] = 1.0

        for j in range(7):
            self.joint_pos_osc[j] = self.sim.data.qpos[self.joint_pos_addr[j]].copy()
            self.joint_vel_osc[j] = self.sim.data.qvel[self.joint_vel_addr[j]].copy()

        jac_pos_osc = np.ndarray(shape=(L*3,), dtype=np.float64)
        jac_rot_osc = np.ndarray(shape=(L*3,), dtype=np.float64)
        mujoco_py.functions.mj_jacSite(self.sim.model, self.sim.data, jac_pos_osc, jac_rot_osc, self.ee_site_adr)

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
                self.mass_matrix_osc[c*7 + r] = mass_array_osc[self.joint_pos_addr[r]*L + self.joint_pos_addr[c]]

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
            'timestep': self.timestep
        }
        return entry


    def get_ee_position_W(self):
        return self.sim.data.get_site_xpos(self.ee_site_name).copy()
    
    def get_ee_position_I(self):
        return self.sim.data.get_site_xpos(self.ee_site_name).copy() - self.relative_origin

    def get_joint_positions(self):
        positions = [self.sim.data.get_joint_qpos(joint).copy() for joint in self.joints]
        return np.array(positions)
    
    def get_joint_velocities(self):
        velocities = [self.sim.data.get_joint_qvel(joint).copy() for joint in self.joints]
        return np.array(velocities)

    def get_ee_velocity(self):
        return self.sim.data.get_site_xvelp(self.ee_site_name).copy()

    def get_cloth_position_I(self):
        positions = dict()
        for site in self.cloth_site_names:
            positions[site] = self.sim.data.get_site_xpos(site).copy() - self.relative_origin
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
                if (i in [0,8]) or (j in [0,8]):
                    site_name = f"S{i}_{j}"
                    positions[site_name] = self.sim.data.get_site_xpos(site_name).copy()
        return positions

    def get_cloth_velocity(self):
        velocities = dict()
        for site in self.cloth_site_names:
            velocities[site] = self.sim.data.get_site_xvelp(site).copy()
        return velocities

    def get_image_obs(self):
        camera_id = self.sim.model.camera_name2id(
            self.train_camera) 
        width = self.camera_config['width']
        height = self.camera_config['height']
        self.viewer.render(width, height, camera_id)
        depth_obs = None
        if self.depth_frames:
            image_obs, depth_obs = copy.deepcopy(self.viewer.read_pixels(width, height, depth=True))
            depth_obs = depth_obs[::-1, :]
            depth_obs = cv2.normalize(depth_obs, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            image_obs = copy.deepcopy(self.viewer.read_pixels(width, height, depth=False))

        image_obs = image_obs[::-1, :, :]

        height_start = int(image_obs.shape[0]/2 - self.image_size[1]/2)
        height_end = height_start + self.image_size[1]

        width_start = int(image_obs.shape[1]/2 - self.image_size[0]/2)
        width_end = width_start + self.image_size[0]
        image_obs = image_obs[height_start:height_end, width_start:width_end, :]
        image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)

        blur_kernel_size = np.random.randint(1,4)
        image_obs = cv2.blur(image_obs,(blur_kernel_size, blur_kernel_size))

        gaussian_noise = np.zeros((self.image_size[0], self.image_size[1]),dtype=np.uint8)
        cv2.randn(gaussian_noise, 128, 20)
        gaussian_noise = (gaussian_noise*0.5).astype(np.uint8)
        image_obs = cv2.add(image_obs,gaussian_noise)

        if not depth_obs is None:
            return np.array([(image_obs / 255).flatten(), depth_obs.flatten()]).flatten()
        else:
            return (image_obs / 255).flatten()


    def get_obs(self):
        achieved_goal_I = np.zeros(self.single_goal_dim*len(self.constraints))
        for i, constraint in enumerate(self.constraints):
            origin = constraint['origin']
            achieved_goal_I[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = self.sim.data.get_site_xpos(
                origin).copy() - self.relative_origin

        cloth_position = np.array(list(self.get_cloth_position_I().values()))
        cloth_velocity = np.array(list(self.get_cloth_velocity(
            ).values()))

        cloth_observation = np.concatenate([cloth_position.flatten(), cloth_velocity.flatten()])

        desired_pos_ctrl_I = self.desired_pos_ctrl_W - self.relative_origin


        if self.robot_observation == "all":
            robot_observation = np.concatenate([self.get_ee_position_I(), self.get_joint_positions(), self.get_ee_velocity(), self.get_joint_velocities(), desired_pos_ctrl_I])
        elif self.robot_observation == "joint":
            robot_observation = np.concatenate([self.get_joint_positions(), self.get_joint_velocities(), desired_pos_ctrl_I])
        elif self.robot_observation == "ee":
            robot_observation = np.concatenate([self.get_ee_position_I(), self.get_ee_velocity(), desired_pos_ctrl_I])
        elif self.robot_observation == "ctrl":
            robot_observation = self.previous_raw_action
        elif self.robot_observation == "none":
            robot_observation = np.zeros(1)


        full_observation = {
            'observation': cloth_observation.copy(),
            'achieved_goal': achieved_goal_I.copy(),
            'desired_goal': self.goal.copy(),
            'model_params': np.array(self.mujoco_model_numerical_values),
            'robot_observation': robot_observation.flatten().copy()
        }

        if self.pixels:
            full_observation['image'] = np.array([image for image in self.frame_stack]).flatten()

        return full_observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def sample_goal_I(self):
        goal = np.zeros(self.single_goal_dim*len(self.constraints))
        if not self.constant_goal:
            noise = self.np_random.uniform(self.goal_noise_range[0],
                                           self.goal_noise_range[1])
        else:
            noise = 0
        for i, constraint in enumerate(self.constraints):
            target = constraint['target']
            target_pos = self.sim.data.get_site_xpos(target).copy()
            offset = np.zeros(self.single_goal_dim)
            if 'noise_directions' in constraint.keys():
                for idx, offset_dir in enumerate(constraint['noise_directions']):
                    offset[idx] = offset_dir*noise

            goal[i*self.single_goal_dim: (i+1) *
                 self.single_goal_dim] = target_pos + offset - self.relative_origin

        #print("Goal", goal)
        return goal.copy()


    def reset_osc_values(self):
        self.initial_O_T_EE = None
        self.initial_joint_osc = None
        self.initial_ee_p_W = None
        self.desired_pos_step_W = None
        self.desired_pos_ctrl_W = None
        self.previous_raw_action = np.zeros(3)
        self.raw_action = None

    def randomize_xml_model(self):
        model_kwargs, model_numerical_values = self.build_xml_kwargs_and_numerical_values()
        self.mujoco_model_numerical_values = model_numerical_values
        self.setup_initial_state_and_sim(model_kwargs)


    def reset(self):
        self.sim.reset()
        utils.remove_distance_welds(self.sim)
        self.sim.set_state(self.initial_state)
        self.sim.data.qfrc_applied[self.joint_vel_addr] = self.initial_qfrc_applied
        self.sim.data.qfrc_bias[self.joint_vel_addr] = self.initial_qfrc_bias
        self.reset_camera()
        self.sim.forward()
        self.reset_osc_values()
        self.update_osc_values()
        utils.enable_distance_welds(self.sim)

        
        self.relative_origin = self.get_ee_position_W()
        self.goal = self.sample_goal_I()


        if not self.viewer is None:
            del self.viewer._markers[:]

        self.episode_success_steps = 0

        if self.pixels:
            image_obs = self.get_image_obs()
            for _ in range(self.frame_stack_size):
                self.frame_stack.append(image_obs)

        q_off = self.initial_qpos - self.get_joint_positions()

        q_ok = np.allclose(self.initial_qpos, self.get_joint_positions(), rtol=0.01, atol=0.01)

        if not q_ok:
            print("Q values not ok", self.initial_qpos - self.get_joint_positions())
                
        return self.get_obs()

    def get_corner_image_positions(self, w, h, camera_matrix, camera_transformation):
        corners = []
        corner_indices = []
        cloth_positions = self.get_cloth_position_W()
        for site in self.cloth_site_names:
            if not "4" in site:
                corner_in_image = np.ones(4)
                corner_in_image[:3] = cloth_positions[site]
                corner = (camera_matrix @ camera_transformation) @ corner_in_image
                u_c, v_c, _ = corner/corner[2]
                corner = [w-u_c, v_c]
                corners.append(corner)
                corner_indices.append(self.corner_index_mapping[site])
        return corners, corner_indices

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
        camera_matrix = np.array(((f, 0, w/ 2), (0, f, h / 2), (0, 0, 1)))
        xmat = self.sim.data.get_camera_xmat(camera_name)
        xpos = self.sim.data.get_camera_xpos(camera_name)

        camera_transformation = np.eye(4)
        camera_transformation[:3,:3] = xmat
        camera_transformation[:3,3] = xpos
        camera_transformation = np.linalg.inv(camera_transformation)[:3,:]

        return camera_matrix, camera_transformation


    def get_masked_image(self, camera, width, height, ee_in_image, aux_output, point_size, greyscale=False, mask_type=None):
        camera_matrix, camera_transformation = self.get_camera_matrices(camera, width, height)
        camera_id = self.sim.model.camera_name2id(camera) 
        self.viewer.render(width, height, camera_id)
        data = np.float32(self.viewer.read_pixels(width, height, depth=False)).copy()
        data = np.float32(data[::-1, :, :]).copy()
        data = np.float32(data)
        if greyscale:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        ee = camera_matrix @ camera_transformation @ ee_in_image
        u_ee, v_ee, _ = ee/ee[2]
        cv2.circle(data, (width-int(u_ee), int(v_ee)), point_size, (0, 0, 0), -1)

        if mask_type == "corners":
            mask = self.get_corner_image_positions(width, height, camera_matrix, camera_transformation)[0]
        elif mask_type == "edges":
            mask = self.get_edge_image_positions(width, height, camera_matrix, camera_transformation)
        else:
            mask = []

        #print(mask_name)
        for point in mask:
            u = int(point[0])
            v = int(point[1])
            #print("{", u, ",",v,"},")
            cv2.circle(data, (u, v), point_size, (0, 0, 255), -1)
        #print("\n")
        if not aux_output is None:
            for aux_idx in range(int(aux_output.flatten().shape[0]/2)):
                aux_u = int(aux_output.flatten()[aux_idx*2]*width)
                aux_v = int(aux_output.flatten()[aux_idx*2+1]*height)
                cv2.circle(data, (aux_u, aux_v), point_size, (0, 255, 0), -1)

        return data



    def capture_images(self, aux_output=None):
        w_eval, h_eval = 1000, 1000
        w_corners, h_corners = 1000, 1000
        w_cnn, h_cnn = self.image_size
        w_cnn_full, h_cnn_full = self.camera_config['width'], self.camera_config['height']

        ee_in_image = np.ones(4)
        ee_pos = self.get_ee_position_W()
        ee_in_image[:3] = ee_pos

        corner_image = self.get_masked_image(self.train_camera, w_corners, h_corners, ee_in_image, aux_output, 8, greyscale=False, mask_type="corners")
        eval_image = self.get_masked_image(self.eval_camera, w_eval, h_eval, ee_in_image, None, 4, greyscale=False, mask_type="corners")
        cnn_color_image_full = self.get_masked_image(self.train_camera, w_cnn_full, h_cnn_full, ee_in_image, aux_output, 2, mask_type="corners")
        cnn_color_image = self.get_masked_image(self.train_camera, w_cnn, h_cnn, ee_in_image, aux_output, 2, mask_type="corners")
        cnn_image = self.get_masked_image(self.train_camera, w_cnn, h_cnn, ee_in_image, aux_output, 2, greyscale=True, mask_type="corners")


        return corner_image, eval_image, cnn_color_image_full, cnn_color_image, cnn_image



class ClothEnvPickled(ClothEnv, EzPickle):
    def __init__(self, **kwargs):
        ClothEnv.__init__(
            self, **kwargs)
        EzPickle.__init__(self)