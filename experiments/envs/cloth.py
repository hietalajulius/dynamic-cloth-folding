import mujoco_py
import osc_binding
import cv2
import numpy as np
import copy
from gym.utils import seeding, EzPickle
from gym.envs.robotics import reward_calculation
import gym
import time
import utils
import os
from skspatial.objects import Line, Sphere
from scipy import spatial
from template_renderer import TemplateRenderer
import math
from collections import deque
import copy


#np.set_printoptions(suppress=True)

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
        sparse_dense,
        goal_noise_range,
        pixels,
        depth_frames,
        frame_stack_size,
        output_max,
        randomize_params,
        randomize_geoms,
        uniform_jnt_tend,
        constant_goal,
        constraints,
        success_reward,
        fail_reward,
        extra_reward,
        kp,
        damping_ratio,
        clip_type,
        control_frequency,
        ctrl_filter,
        max_episode_steps,
        ate_penalty_coef,
        action_norm_penalty_coef,
        cosine_penalty_coef,
        num_eval_rollouts,
        save_folder,
        mujoco_model_kwargs,
        mujoco_model_kwarg_ranges,
        robot_observation,
        image_obs_noise_mean=1,
        image_obs_noise_std=0,
        initial_xml_dump=False,
        has_viewer=False
    ):  
        assert clip_type == "sphere" or clip_type == "spike" or clip_type == "none"

        self.mujoco_model_kwargs = mujoco_model_kwargs
        self.mujoco_model_kwarg_ranges = mujoco_model_kwarg_ranges

        self.save_folder = save_folder
        self.initial_xml_dump = initial_xml_dump
        self.num_eval_rollouts = num_eval_rollouts
        self.ate_penalty_coef = ate_penalty_coef
        self.action_norm_penalty_coef = action_norm_penalty_coef
        self.cosine_penalty_coef = cosine_penalty_coef
        self.filter = ctrl_filter
        self.clip_type = clip_type
        self.kp = kp
        self.damping_ratio = damping_ratio
        self.max_advance = output_max
        self.success_reward = success_reward
        self.fail_reward = fail_reward
        self.extra_reward = extra_reward
        self.sparse_dense = sparse_dense
        self.constraints = constraints
        self.goal_noise_range = goal_noise_range
        self.pixels = pixels
        self.depth_frames = depth_frames
        self.frame_stack_size = frame_stack_size
        self.image_size = (100, 100)
        self.randomize_geoms = randomize_geoms
        self.randomize_params = randomize_params
        self.uniform_jnt_tend = uniform_jnt_tend
        self.constant_goal = constant_goal
        self.has_viewer = has_viewer
        self.max_episode_steps = max_episode_steps
        self.image_obs_noise_mean = image_obs_noise_mean
        self.image_obs_noise_std = image_obs_noise_std
        self.robot_observation = robot_observation

        self.max_success_steps = 20
        self.frame_stack = deque([], maxlen = self.frame_stack_size)
        #TODO: figure out this case when no reward offset used

        
        self.single_goal_dim = 3
        self.timestep = mujoco_model_kwargs["timestep"]
        self.control_frequency = control_frequency

        steps_per_second = 1 / self.timestep
        self.substeps = 1 / (self.timestep*self.control_frequency)
        self.between_steps = 1000 / steps_per_second
        self.delta_tau_max = 1000 / steps_per_second

        self.train_camera = "train_camera"
        self.eval_camera = "eval_camera"
        self.min_damping = 0.0001  # TODO: pass ranges in from outside
        self.max_damping = 0.02
        self.min_stiffness = 0.0001  # TODO: pass ranges in from outside
        self.max_stiffness = 0.02
        self.min_geom_size = 0.005  # TODO: pass ranges in from outside
        self.max_geom_size = 0.011
        self.limits_min = [-0.35, -0.35, 0.0]
        self.limits_max = [0.05, 0.05, 0.4]        
        self.episode_success_steps = 0

        self.current_geom_size = self.max_geom_size/2
        self.current_joint_stiffness = self.max_stiffness/2
        self.current_joint_damping = self.max_damping/2
        self.current_tendon_stiffness = self.max_stiffness/2
        self.current_tendon_damping = self.max_damping/2

        self.seed()

        #sideways
        self.initial_qpos = np.array([0.212422, 0.362907, -0.00733391, -1.9649, -0.0198034, 2.37451, -1.50499])
        self.initial_des_pos = np.array([0.610123, 0.125163, 0.189346])

        #diagonal
        #self.initial_qpos = np.array([0.114367, 0.575019, 0.0550664, -1.60919, -0.079246, 2.23369, -1.56064])
        #self.initial_des_pos = np.array([0.692898, 0.107481, 0.194496])

        self.cloth_site_names = []
        #TODO: Figure out how many to use
        for i in [0, 4, 8]:
            for j in [0, 4, 8]:
                self.cloth_site_names.append(f"S{i}_{j}")
        self.ee_site_name = 'grip_site'
        self.joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.setup_initial_state_and_sim()
        self.setup_viewer()
        
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

    def setup_viewer(self):
        if self.has_viewer:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self.viewer.vopt.geomgroup[0] = 0
            self.viewer.vopt.geomgroup[1] = 1
        else:
            self.viewer = None

    def dump_xml_models(self):
        with open(f"{self.save_folder}/compiled_mujoco_model_no_inertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=False)

        with open(f"{self.save_folder}/compiled_mujoco_model_with_intertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=True)

        print("Saved compiled xml mujoco models")

    def setup_initial_state_and_sim(self):
        template_renderer = TemplateRenderer()
        template_renderer.render_to_file("arena.xml", f"{self.save_folder}/mujoco_template.xml", **self.mujoco_model_kwargs)
        self.mjpy_model = mujoco_py.load_model_from_path(f"{self.save_folder}/mujoco_template.xml")
        self.sim = mujoco_py.MjSim(self.mjpy_model)
        utils.remove_distance_welds(self.sim)

        self.joint_indexes = [self.sim.model.joint_name2id(joint) for joint in self.joints]
        self.joint_pos_addr = [self.sim.model.get_joint_qpos_addr(joint) for joint in self.joints]
        self.joint_vel_addr = [self.sim.model.get_joint_qvel_addr(joint) for joint in self.joints]
        self.ee_site_adr = mujoco_py.functions.mj_name2id(self.sim.model, 6, "grip_site")
        
        #Sets robot to initial qpos and resets osc values
        self.set_robot_initial_joints()
        self.reset_osc_values()
        self.update_osc_values()

        #TODO: Sets robot to initial ee pos and resets osc values
        #self.set_robot_to_ee_pos()
        #self.reset_osc()

        for _ in range(30):
            mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
            self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr]
            mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)

        #utils.enable_distance_welds(self.sim)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.initial_qfrc_applied = self.sim.data.qfrc_applied[self.joint_vel_addr].copy()
        self.initial_qfrc_bias = self.sim.data.qfrc_bias[self.joint_vel_addr].copy()
        if self.initial_xml_dump: #Slightly ghetto
            self.dump_xml_models()

        self.mjpy_model = mujoco_py.load_model_from_path(f"{self.save_folder}/compiled_mujoco_model_no_inertias.xml")
        self.sim = mujoco_py.MjSim(self.mjpy_model)

        

    def set_robot_to_ee_pos(self):
        #TODO: clean up
        desired = self.sim.data.get_site_xpos("S8_8").copy() + np.array([0,0,0.005])
        desired_above = desired[:2]

        dist_above = np.linalg.norm(self.sim.data.site_xpos[self.ee_site_adr].copy()[:2] - desired_above)
        while dist_above > 0.01:
            self.desired_pos_ctrl[:2] = desired_above
            self.step_env()
            dist_above = np.linalg.norm(self.sim.data.site_xpos[self.ee_site_adr].copy()[:2] - desired_above)
        dist = np.linalg.norm(self.sim.data.site_xpos[self.ee_site_adr].copy() - desired)
        while dist > 0.005:
            self.desired_pos_ctrl = desired
            self.step_env()
            dist = np.linalg.norm(self.sim.data.site_xpos[self.ee_site_adr].copy() - desired)
        print(self.sim.data.site_xpos[self.ee_site_adr].copy())

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
        action = raw_action*self.max_advance
        if self.pixels:
            image_obs_substep_idx_mean = self.image_obs_noise_mean * (self.substeps-1)
            image_obs_substep_idx = int(np.random.normal(image_obs_substep_idx_mean, self.image_obs_noise_std))
            image_obs_substep_idx = np.clip(image_obs_substep_idx, 0, self.substeps-1)

        #TODO: use for clipping
        cosine_distance = compute_cosine_distance(self.previous_delta_vector, action)
        self.previous_delta_vector = action
        
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

        obs = self.get_obs()
        
        #TODO: this is only for clear markers
        #self.prepare_eval_visualization(obs, previous_desired_pos_step_W)

        reward, done, info = self.post_action(action, obs, raw_action, cosine_distance)
        return obs, reward, done, info

    '''
    def spike_clip(self, action):
        if compute_cosine_similarity(action, self.previous_delta_vector) > 0:
            return action
            self.previous_delta_vector = action
        else:
            return np.zeros(3)



    def sphere_clip(self, action):
        if compute_cosine_similarity(action, self.previous_delta_vector) > 0:
            radius = self.max_advance/2
            sphere_center = radius*(self.previous_delta_vector/np.linalg.norm(self.previous_delta_vector))
            sphere = Sphere(sphere_center, radius)
            line = Line([0, 0, 0], action)
            points = sphere.intersect_line(line)
            norms = np.linalg.norm(points, axis=1)
            argmax = np.argmax(norms)
            action = points[argmax]
            self.previous_delta_vector = action
            return action
        else:
            return np.zeros(3)
    '''


    def compute_task_reward(self, achieved_goal, desired_goal, info):
        return self.task_reward_function(achieved_goal, desired_goal, info)

    def prepare_eval_visualization(self, obs, previous_desired_pos_step):


        del self.viewer._markers[:]

        '''
        
        for i in range(int(self.goal.shape[0]/3)):
            self.viewer.add_marker(size=np.array([.001, .001, .001]), pos=self.goal[i*self.single_goal_dim: (i+1) *
                self.single_goal_dim] + self.relative_origin, label="d"+str(i))
            self.viewer.add_marker(size=np.array([.001, .001, .001]), pos=obs['achieved_goal'][i*self.single_goal_dim: (i+1) *
                self.single_goal_dim] + self.relative_origin, label="a"+str(i))

        self.viewer.add_marker(size=np.array([.0005, .0005, .0005]), pos=self.desired_pos_ctrl_W, label="d")
        self.viewer.add_marker(size=np.array([.0005, .0005, .0005]), pos=previous_desired_pos_step, label="c")

        '''
    


        
    def post_action(self, action, obs, raw_action, cosine_distance):
        task_reward = self.compute_task_reward(np.reshape(obs['achieved_goal'], (1, -1)), np.reshape(self.goal, (1, -1)), dict())[0]
        is_success = task_reward > self.fail_reward


        delta_size_penalty = -np.linalg.norm(raw_action)
        ate_penalty = -np.linalg.norm(self.desired_pos_ctrl_W - self.get_ee_position_W())
        cosine_penalty = -cosine_distance

        control_penalty = delta_size_penalty*self.action_norm_penalty_coef + ate_penalty*self.ate_penalty_coef + cosine_penalty*self.cosine_penalty_coef

        reward = task_reward + control_penalty

        if is_success and self.episode_success_steps == 0:
            print("Real sim success",
                np.round(reward, decimals=3),
                np.round(task_reward, decimals=3),
                np.round(control_penalty, decimals=3))

        camera_matrix, camera_transformation = self.get_camera_matrices(self.train_camera, self.image_size[0], self.image_size[1])
        corners_in_image = self.get_corner_image_positions(self.image_size[0], self.image_size[0], camera_matrix, camera_transformation)

        flattened_corners = []
        for corner in corners_in_image:
            flattened_corners.append(corner[0]/self.image_size[0])
            flattened_corners.append(corner[1]/self.image_size[1])

        flattened_corners = np.array(flattened_corners) 

        info = {
                "task_reward": task_reward,
                "delta_size": np.linalg.norm(raw_action*self.max_advance),
                "cosine_distance" : cosine_distance,
                "delta_size_penalty": delta_size_penalty,
                "ate_penalty": ate_penalty,
                "control_penalty": control_penalty,
                "reward": reward,
                'is_success': is_success,
                'corner_positions': flattened_corners
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

    def get_cloth_velocity(self):
        velocities = dict()
        for site in self.cloth_site_names:
            velocities[site] = self.sim.data.get_site_xvelp(site).copy()
        return velocities

    def get_image_obs(self):
        camera_id = self.sim.model.camera_name2id(
            self.train_camera) 
        self.viewer.render(self.image_size[0], self.image_size[1], camera_id)
        depth_obs = None
        if self.depth_frames:
            image_obs, depth_obs = copy.deepcopy(self.viewer.read_pixels(self.image_size[0], self.image_size[1], depth=True))
            depth_obs = depth_obs[::-1, :]
            depth_obs = cv2.normalize(depth_obs, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            image_obs = copy.deepcopy(self.viewer.read_pixels(self.image_size[0], self.image_size[1], depth=False))

        image_obs = image_obs[::-1, :, :]
        image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)

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

        robot_observation = desired_pos_ctrl_I

        if self.robot_observation == "all":
            robot_observation = np.concatenate([self.get_ee_position_W(), self.get_joint_positions(), self.get_ee_velocity(), self.get_joint_velocities(), robot_observation])
        elif self.robot_observation == "joint":
            robot_observation = np.concatenate([self.get_joint_positions(), self.get_joint_velocities(), robot_observation])
        elif self.robot_observation == "ee":
            robot_observation = np.concatenate([self.get_ee_position_W(), self.get_ee_velocity(), robot_observation])

        

        if self.randomize_geoms and self.randomize_params:
            model_params = np.array([self.current_joint_stiffness, self.current_joint_damping,
                                     self.current_tendon_stiffness, self.current_tendon_damping, self.current_geom_size])
        elif self.randomize_params:
            model_params = np.array([self.current_joint_stiffness, self.current_joint_damping,
                                     self.current_tendon_stiffness, self.current_tendon_damping])
        else:
            model_params = np.array([0])

        full_observation = {
            'observation': cloth_observation.copy(),
            'achieved_goal': achieved_goal_I.copy(),
            'desired_goal': self.goal.copy(),
            'model_params': model_params.copy(),
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

        return goal.copy()

    def set_joint_tendon_params(self, joint_stiffness, joint_damping, tendon_stiffness, tendon_damping):
        for _, joint_name in enumerate(self.sim.model.joint_names):
            joint_id = self.sim.model.joint_name2id(joint_name)
            self.sim.model.jnt_stiffness[joint_id] = joint_stiffness
            self.sim.model.dof_damping[joint_id] = joint_damping

        for _, tendon_name in enumerate(self.sim.model.tendon_names):
            tendon_id = self.sim.model.tendon_name2id(tendon_name)
            self.sim.model.tendon_stiffness[tendon_id] = tendon_stiffness
            self.sim.model.tendon_damping[tendon_id] = tendon_damping

        self.sim.forward()

    def reset_osc_values(self):
        self.initial_O_T_EE = None
        self.initial_joint_osc = None
        self.initial_ee_p_W = None
        self.desired_pos_step_W = None
        self.desired_pos_ctrl_W = None
        self.previous_delta_vector = np.zeros(3)
        self.raw_action = None
    

    def reset(self):
        self.sim.reset()
        utils.remove_distance_welds(self.sim)
        self.sim.set_state(self.initial_state)
        self.sim.data.qfrc_applied[self.joint_vel_addr] = self.initial_qfrc_applied
        self.sim.data.qfrc_bias[self.joint_vel_addr] = self.initial_qfrc_bias
        self.sim.forward()
        self.reset_osc_values()
        self.update_osc_values()
        utils.enable_distance_welds(self.sim)
        
        self.relative_origin = self.get_ee_position_W()
        self.goal = self.sample_goal_I()
        self.min_absolute_W = self.initial_ee_p_W + self.limits_min
        self.max_absolute_W = self.initial_ee_p_W + self.limits_max

        if not self.viewer is None:
            del self.viewer._markers[:]

        self.episode_success_steps = 0


        if self.randomize_params:
            self.current_joint_stiffness = self.np_random.uniform(
                self.min_stiffness, self.max_stiffness)
            self.current_joint_damping = self.np_random.uniform(
                self.min_damping, self.max_damping)

            if self.uniform_jnt_tend:
                self.current_tendon_stiffness = self.current_joint_stiffness
                self.current_tendon_damping = self.current_joint_damping
            else:
                # Own damping/stiffness for tendons
                self.current_tendon_stiffness = self.np_random.uniform(
                    self.min_stiffness, self.max_stiffness)
                self.current_tendon_damping = self.np_random.uniform(
                    self.min_damping, self.max_damping)

            self.set_joint_tendon_params(self.current_joint_stiffness, self.current_joint_damping,
                                        self.current_tendon_stiffness, self.current_tendon_damping)

        if self.randomize_geoms:
            self.current_geom_size = self.np_random.uniform(
                self.min_geom_size, self.max_geom_size)
            self.set_geom_params()

        if self.pixels:
            image_obs = self.get_image_obs()
            for _ in range(self.frame_stack_size):
                self.frame_stack.append(image_obs)

        return self.get_obs()

    def get_corner_image_positions(self, w, h, camera_matrix, camera_transformation):
        corners = []
        cloth_positions = self.get_cloth_position_W()
        for site in self.cloth_site_names:
            if not "4" in site:
                corner_in_image = np.ones(4)
                corner_in_image[:3] = cloth_positions[site]
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



    def capture_image(self, aux_output):
        w_eval, h_eval = 1000, 1000
        w_train, h_train = self.image_size
        w_train *= 10
        h_train *= 10

        train_camera_matrix, train_camera_transformation = self.get_camera_matrices(self.train_camera, w_train, h_train)

        train_camera_id = self.sim.model.camera_name2id(self.train_camera) 

        ee_in_image = np.ones(4)
        ee_pos = self.get_ee_position_W()
        ee_in_image[:3] = ee_pos

        self.viewer.render(w_train, h_train, train_camera_id)
        train_data = np.float32(self.viewer.read_pixels(w_train, h_train, depth=False)).copy()
        train_data = np.float32(train_data[::-1, :, :]).copy()
        train_data = np.float32(train_data)
        ee = train_camera_matrix @ train_camera_transformation @ ee_in_image
        u_ee, v_ee, _ = ee/ee[2]
        cv2.circle(train_data, (w_train-int(u_ee), int(v_ee)), 10, (0, 0, 0), -1)

        corners = self.get_corner_image_positions(w_train, h_train, train_camera_matrix, train_camera_transformation)

        for corner in corners:
            u = int(corner[0])
            v = int(corner[1])
            cv2.circle(train_data, (u, v), 8, (0, 0, 255), -1)
        
        if not aux_output is None:
            for aux_idx in range(int(aux_output.flatten().shape[0]/2)):
                aux_u = int(aux_output.flatten()[aux_idx*2]*w_train)
                aux_v = int(aux_output.flatten()[aux_idx*2+1]*h_train)
                cv2.circle(train_data, (aux_u, aux_v), 8, (0, 255, 0), -1)



        eval_camera_id = self.sim.model.camera_name2id(self.eval_camera) 
        self.viewer.render(h_eval,w_eval, eval_camera_id)
        eval_data = np.float32(self.viewer.read_pixels(w_eval, h_eval, depth=False)).copy()
        eval_data = np.float32(eval_data[::-1, :, :]).copy()
        eval_data = np.float32(eval_data)

        return train_data, eval_data



class ClothEnvPickled(ClothEnv, EzPickle):
    def __init__(self, **kwargs):
        print("Creating pickled env")
        ClothEnv.__init__(
            self, **kwargs)
        EzPickle.__init__(self)