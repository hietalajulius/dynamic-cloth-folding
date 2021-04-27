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


#np.set_printoptions(suppress=True)

def compute_cosine_similarity(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)



class ClothEnv(object):
    def __init__(
        self,
        sparse_dense,
        goal_noise_range,
        pixels,
        output_max,
        randomize_params,
        randomize_geoms,
        uniform_jnt_tend,
        random_seed,
        velocity_in_obs,
        image_size,
        constant_goal,
        constraints,
        reward_offset,
        kp,
        damping_ratio,
        sphere_clipping,
        ctrl_filter,
        save_folder,
        max_episode_steps,
        control_penalty_coef,
        eval_env=False,
        has_viewer=False
    ):  
        self.control_penalty_coef = control_penalty_coef
        self.eval_env = eval_env
        self.filter = ctrl_filter
        self.save_folder = save_folder
        self.sphere_clipping = sphere_clipping
        self.kp = kp
        self.damping_ratio = damping_ratio
        self.max_advance = output_max
        self.reward_offset = reward_offset
        self.sparse_dense = sparse_dense
        self.constraints = constraints
        self.goal_noise_range = goal_noise_range
        self.velocity_in_obs = velocity_in_obs
        self.pixels = pixels
        self.image_size = image_size
        self.randomize_geoms = randomize_geoms
        self.randomize_params = randomize_params
        self.uniform_jnt_tend = uniform_jnt_tend
        self.constant_goal = constant_goal
        self.has_viewer = has_viewer
        self.max_episode_steps = max_episode_steps

        self.max_success_steps = 5
        #TODO: figure out this case when no reward offset used
        self.max_return = self.max_success_steps * self.reward_offset + (self.max_episode_steps - self.max_success_steps)*(self.reward_offset - 1)
    
        self.single_goal_dim = 3
        self.eval_saves = 0
        self.substeps = 10
        self.between_steps = 10
        self.delta_tau_max = 10
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

        self.initial_qpos = [0.227, 0.194, -0.0253, -1.84, -0.0124, 2.09, -1.76]
        
        self.ctrl_goals = []
        self.pos_goals = []
        self.ee_positions = []
        self.cloth_site_names = ["S0_0", "S4_0", "S8_0", "S0_4","S0_8", "S4_8", "S8_8", "S8_4"]
        self.ee_site_name = 'grip_site'
        self.joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.initial_reset()
        if self.has_viewer:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self.viewer.vopt.geomgroup[0] = 0
            self.viewer.vopt.geomgroup[1] = 1
        else:
            self.viewer = None
        
        self.goal = self.sample_goal()

        self.task_reward_function = reward_calculation.get_task_reward_function(
            self.constraints, self.single_goal_dim, self.sparse_dense, self.reward_offset)

        self.action_space = gym.spaces.Box(-1,
                                           1, shape=(3,), dtype='float32')
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
    def dump_xml_models(self):
        with open(f"{self.save_folder}/compiled_mujoco_model_no_inertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=False)

        with open(f"{self.save_folder}/compiled_mujoco_model_with_intertias.xml", "w") as f:
            self.sim.save(f, format='xml', keep_inertials=True)

        print("Saved compiled xml mujoco models")

    def initial_reset(self):
        template_renderer = TemplateRenderer()
        template_renderer.render_to_file("arena.xml", f"{self.save_folder}/mujoco_template.xml", timestep=0.01, geom_size=0.008)
        self.mjpy_model = mujoco_py.load_model_from_path(f"{self.save_folder}/mujoco_template.xml")
        self.sim = mujoco_py.MjSim(self.mjpy_model)
        utils.remove_distance_welds(self.sim)

        self.joint_indexes = [self.sim.model.joint_name2id(joint) for joint in self.joints]
        self.joint_pos_addr = [self.sim.model.get_joint_qpos_addr(joint) for joint in self.joints]
        self.joint_vel_addr = [self.sim.model.get_joint_qvel_addr(joint) for joint in self.joints]
        self.ee_site_adr = mujoco_py.functions.mj_name2id(self.sim.model, 6, "grip_site")
        
        #Sets robot to initial qpos and resets osc values
        self.reset_robot_initial()
        self.reset_osc()
        #Sets robot to initial ee pos and resets osc values
        self.reset_robot_to_pos()
        self.reset_osc()

        for _ in range(30):
            mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
            self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr]
            mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)

        utils.enable_distance_welds(self.sim)

        self.initial_state = copy.deepcopy(self.sim.get_state())
        if self.eval_env: #Slightly ghetto
            self.dump_xml_models()

        self.mjpy_model = mujoco_py.load_model_from_path(f"{self.save_folder}/compiled_mujoco_model_no_inertias.xml")
        self.sim = mujoco_py.MjSim(self.mjpy_model)
        self.sim.set_state(self.initial_state)
        self.relative_origin = self.sim.data.site_xpos[self.ee_site_adr].copy()
        

    def reset_robot_to_pos(self):
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
        print("Found valid initial config")

    def reset_robot_initial(self):
        for j, joint in enumerate(self.joints):
            self.sim.data.set_joint_qpos(joint, self.initial_qpos[j])

        for _ in range(10):
            self.sim.forward()

    def reset_osc(self):
        if len(self.ee_positions) > 0:
            traj = np.concatenate([np.array(self.pos_goals), np.array(self.ctrl_goals), np.array(self.ee_positions)], axis=1)
            np.savetxt(f"{self.save_folder}/eval_trajs/{self.eval_saves}.csv",
                   traj, delimiter=",", fmt='%f')
            self.eval_saves += 1
        self.initial_O_T_EE = None
        self.initial_joint_osc = None
        self.initial_ee_p = None
        self.desired_pos_step = None
        self.desired_pos_ctrl = None
        self.current_delta_vector = None
        self.previous_delta_vector = None
        self.raw_action = None
        self.ctrl_goals = []
        self.pos_goals = []
        self.ee_positions = []
        self.update_osc_vals()

    def reset(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        self.reset_osc()
        self.goal = self.sample_goal()
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

        return self.get_obs()

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
                                                self.desired_pos_ctrl,
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
        self.update_osc_vals()
        tau = self.run_controller()
        self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr] + tau
        mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)

    def step(self, action, evaluation=False):
        self.pre_action(action,evaluation)
        self.raw_action = action.copy()
        action = self.clip_action(action)

        #This is the first delta
        if self.previous_delta_vector is None and not np.allclose(action, np.zeros(3), atol=1e-05):
            self.previous_delta_vector = action
        if not np.allclose(self.current_delta_vector, np.zeros(3), atol=1e-05):
            self.previous_delta_vector = self.current_delta_vector.copy()


        previous_desired_pos_step = self.desired_pos_step.copy()
        desired_pos_step_absolute = previous_desired_pos_step + action
        min_absolute = self.initial_ee_p + self.limits_min
        max_absolute = self.initial_ee_p + self.limits_max
        self.desired_pos_step = np.clip(desired_pos_step_absolute, min_absolute, max_absolute)
        self.current_delta_vector = self.desired_pos_step - previous_desired_pos_step
        for i in range(self.substeps):
            for j in range(self.between_steps):
                self.desired_pos_ctrl = self.filter*self.desired_pos_step + (1-self.filter)*self.desired_pos_ctrl
            self.step_env()

        obs = self.get_obs()
        reward, done, info = self.post_action(obs, previous_desired_pos_step, evaluation=evaluation)
        return obs, reward, done, info

    def clip_action(self, action):
        action = action.copy()*self.max_advance
        if self.sphere_clipping:
            first_step = self.previous_delta_vector is None
            if first_step or compute_cosine_similarity(self.previous_delta_vector, action) > 0:
                radius = self.max_advance/2
                if not first_step:
                    sphere_center = radius*(self.previous_delta_vector/np.linalg.norm(self.previous_delta_vector))
                else:
                    sphere_center = radius*(action/np.linalg.norm(action))
                sphere = Sphere(sphere_center, radius)
                line = Line([0, 0, 0], action)
                points = sphere.intersect_line(line)
                norms = np.linalg.norm(points, axis=1)
                argmax = np.argmax(norms)
                argmin = np.argmin(norms)
                norm_action = np.linalg.norm(action)
                if norm_action > norms[argmax]:
                    action = points[argmax]
                if not np.allclose(points[argmin], np.zeros(3),atol=1e-05):
                    print("Not close", points[argmin])
            else:
                action = np.zeros(3)

        return action

    def compute_task_reward(self, achieved_goal, desired_goal, info):
        return self.task_reward_function(achieved_goal, desired_goal, info)

    def pre_action(self, action, evaluation):
        if evaluation:
            self.ee_positions.append(self.get_ee_position())
            self.ctrl_goals.append(self.desired_pos_ctrl)
            self.pos_goals.append(self.desired_pos_step)

            del self.viewer._markers[:]



        
    def post_action(self, obs, previous_desired_pos_step, evaluation=False):
        if evaluation:
            for i in range(int(self.goal.shape[0]/3)):
                self.viewer.add_marker(size=np.array([.001, .001, .001]), pos=self.goal[i*self.single_goal_dim: (i+1) *
                    self.single_goal_dim] + self.relative_origin, label="d"+str(i))
                self.viewer.add_marker(size=np.array([.001, .001, .001]), pos=obs['achieved_goal'][i*self.single_goal_dim: (i+1) *
                    self.single_goal_dim] + self.relative_origin, label="a"+str(i))

            self.viewer.add_marker(size=np.array([.0005, .0005, .0005]), pos=self.desired_pos_ctrl, label="d")
            self.viewer.add_marker(size=np.array([.0005, .0005, .0005]), pos=previous_desired_pos_step, label="c")

        ctrl_penalty_only = False
        is_success = (self.compute_task_reward(np.reshape(obs['achieved_goal'], (1, -1)), np.reshape(self.goal, (1, -1)), dict(ctrl_penalty_onlys=np.array([ctrl_penalty_only])))[0] - self.reward_offset) > -1
        self.episode_success_steps += int(is_success)
        if self.episode_success_steps >= self.max_success_steps:
            ctrl_penalty_only = True
        
        task_reward = self.compute_task_reward(np.reshape(obs['achieved_goal'], (1, -1)), np.reshape(self.goal, (1, -1)), dict(ctrl_penalty_onlys=np.array([ctrl_penalty_only])))[0]

        

        if not self.previous_delta_vector is None and not np.allclose(self.raw_action, np.zeros(3), atol=1e-05):
            cosine_similarity = compute_cosine_similarity(self.previous_delta_vector, self.raw_action)
        else:
            cosine_similarity = 0

        penalty_multiplier = -(self.max_return/(2*self.max_episode_steps))
        cosine_similarity_penalty = penalty_multiplier*(1-cosine_similarity)
        delta_size_penalty = penalty_multiplier*(np.linalg.norm(self.raw_action*self.max_advance)/np.linalg.norm(np.ones(3)*self.max_advance))

        control_penalty = cosine_similarity_penalty + delta_size_penalty
        scaled_control_penalty = control_penalty*self.control_penalty_coef

        reward = task_reward + scaled_control_penalty

        if is_success and self.episode_success_steps == 1:
            print("Real sim success",
                    np.round(reward, decimals=1),
                    np.round(task_reward, decimals=1),
                    np.round(control_penalty, decimals=1),
                    "Joint values:",
                    np.round(self.current_joint_damping, decimals=4),
                    np.round(self.current_joint_stiffness, decimals=4))

        camera_matrix, camera_transformation = self.get_camera_matrices(self.train_camera, self.image_size, self.image_size)
        corners_in_image = self.get_corner_image_positions(self.image_size, self.image_size, camera_matrix, camera_transformation)
        
        if evaluation and self.pixels:
            data = obs['image'].copy().reshape((100, 100, 1))
            for corner in corners_in_image:
                cv2.circle(data, (int(corner[0]), int(corner[1])), 3, (0, 0, 255), -1)

            cv2.imwrite(f'{self.save_folder}/cnn_images/{str(len(self.ee_positions)).zfill(3)}.png', data*255)
        

        flattened_corners = np.array(corners_in_image).flatten()/self.image_size
        error_norm = np.linalg.norm(self.desired_pos_step - self.get_ee_position())    

        info = {
                "task_reward": task_reward,
                "delta_size": np.linalg.norm(self.raw_action*self.max_advance),
                "cosine_similarity" : cosine_similarity,
                "cosine_similarity_penalty": cosine_similarity_penalty,
                "delta_size_penalty": delta_size_penalty,
                "control_penalty": control_penalty,
                "scaled_control_penalty": scaled_control_penalty,
                "reward": reward,
                'is_success': is_success,
                'error_norm': error_norm,
                'corner_positions': flattened_corners,
                'ctrl_penalty_only': ctrl_penalty_only
                }

        done = False  # Let run for max steps

        return reward, done, info
        


    def update_osc_vals(self):
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

        if self.desired_pos_ctrl is None:
            self.desired_pos_ctrl = p.copy()
        if self.desired_pos_step is None:
            self.desired_pos_step = p.copy()
        if self.initial_ee_p is None:
            self.initial_ee_p = p.copy()
        if self.current_delta_vector is None:
            self.current_delta_vector = np.zeros(3)


    def get_ee_position(self):
        return self.sim.data.get_site_xpos(self.ee_site_name).copy()

    def get_joint_positions(self):
        positions = [self.sim.data.get_joint_qpos(joint).copy() for joint in self.joints]
        return np.array(positions)
    
    def get_joint_velocities(self):
        velocities = [self.sim.data.get_joint_qvel(joint).copy() for joint in self.joints]
        return np.array(velocities)

    def get_ee_velocity(self):
        return self.sim.data.get_site_xvelp(self.ee_site_name).copy()

    def get_cloth_position(self):
        positions = dict()
        for site in self.cloth_site_names:
            positions[site] = self.sim.data.get_site_xpos(site).copy()
        return positions

    def get_cloth_velocity(self):
        velocities = dict()
        for site in self.cloth_site_names:
            velocities[site] = self.sim.data.get_site_xvelp(site).copy()
        return velocities

    def get_obs(self):
        achieved_goal = np.zeros(self.single_goal_dim*len(self.constraints))
        for i, constraint in enumerate(self.constraints):
            origin = constraint['origin']
            achieved_goal[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = self.sim.data.get_site_xpos(
                origin).copy() - self.relative_origin

        #Position observations relative to initial position
        cloth_position = np.array(list(self.get_cloth_position().values())) - self.relative_origin
        robot_position = self.get_joint_positions() 

        if self.velocity_in_obs:
            cloth_velocity = np.array(list(self.get_cloth_velocity(
            ).values()))
            robot_velocity = self.get_joint_velocities()
            observation = np.concatenate([cloth_position.flatten(), cloth_velocity.flatten()])

            robot_observation = np.concatenate(
                [robot_position, robot_velocity])
        else:
            observation = np.concatenate([cloth_position.flatten()])
            robot_observation = robot_position

        robot_observation = np.concatenate(
            [robot_observation, self.desired_pos_ctrl - self.relative_origin])

        if self.randomize_geoms and self.randomize_params:
            model_params = np.array([self.current_joint_stiffness, self.current_joint_damping,
                                     self.current_tendon_stiffness, self.current_tendon_damping, self.current_geom_size])
        elif self.randomize_params:
            model_params = np.array([self.current_joint_stiffness, self.current_joint_damping,
                                     self.current_tendon_stiffness, self.current_tendon_damping])
        else:
            model_params = np.array([0])

        full_observation = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'model_params': model_params.copy(),
            'robot_observation': robot_observation.flatten().copy()
        }

        if self.pixels:
            camera_id = self.sim.model.camera_name2id(
                self.train_camera) 
            self.viewer.render(self.image_size, self.image_size, camera_id)
            image_obs = self.viewer.read_pixels(self.image_size, self.image_size, depth=False)
            image_obs = image_obs[::-1, :, :]
            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)
            full_observation['image'] = (image_obs / 255).flatten()
        return full_observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def sample_goal(self):
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

    def get_corner_image_positions(self, w, h, camera_matrix, camera_transformation):
        corners = []
        cloth_positions = self.get_cloth_position()
        for site in self.cloth_site_names:
            if not "4" in site:
                corner_in_image = np.ones(4)
                corner_in_image[:3] = cloth_positions[site]
                corner = (camera_matrix @ camera_transformation) @ corner_in_image
                u_c, v_c, _ = corner/corner[2]

                #cv2.circle(data, (w-int(u_c), int(v_c)), 10, (0, 0, 255), -1)
                corner = [w-u_c, v_c]
                corners.append(corner)
        return corners

    def get_camera_matrices(self, camera_name, h, w):
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



    def capture_image(self, aux_output, path_length):
        w, h = 1000, 1000

        camera_matrix, camera_transformation = self.get_camera_matrices(self.train_camera, h, w)
        train_camera_id = self.sim.model.camera_name2id(self.train_camera) 

        ee_in_image = np.ones(4)
        ee_pos = self.get_ee_position()
        ee_in_image[:3] = ee_pos

        self.viewer.render(h,w, train_camera_id)
        data = np.float32(self.viewer.read_pixels(w, h, depth=False)).copy()
        data = np.float32(data[::-1, :, :]).copy()
        data = np.float32(data)
        ee = camera_matrix @ camera_transformation @ ee_in_image
        u_ee, v_ee, _ = ee/ee[2]
        cv2.circle(data, (w-int(u_ee), int(v_ee)), 10, (0, 0, 0), -1)

        corners = self.get_corner_image_positions(w, h, camera_matrix, camera_transformation)

        for corner in corners:
            u = int(corner[0])
            v = int(corner[1])
            cv2.circle(data, (u, v), 8, (0, 0, 255), -1)
        
        if not aux_output is None:
            for aux_idx in range(int(aux_output.flatten().shape[0]/2)):
                aux_u = int(aux_output.flatten()[aux_idx*2]*w)
                aux_v = int(aux_output.flatten()[aux_idx*2+1]*h)
                cv2.circle(data, (aux_u, aux_v), 8, (0, 255, 0), -1)


        cv2.imwrite(f'{self.save_folder}/eval_corner_images/{str(path_length).zfill(3)}.png', data)

        eval_camera_id = self.sim.model.camera_name2id(self.eval_camera) 
        self.viewer.render(h,w, eval_camera_id)
        data = np.float32(self.viewer.read_pixels(w, h, depth=False)).copy()
        data = np.float32(data[::-1, :, :]).copy()
        data = np.float32(data)
        cv2.imwrite(f'{self.save_folder}/eval_images/{str(path_length).zfill(3)}.png', data)



class ClothEnvPickled(ClothEnv, EzPickle):
    def __init__(self, **kwargs):
        print("Creating pickled env")
        ClothEnv.__init__(
            self, **kwargs)
        EzPickle.__init__(self)