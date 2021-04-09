import mujoco_py
import osc_binding
import cv2
import numpy as np
import copy
from gym.utils import seeding
from gym.envs.robotics import reward_calculation
import gym
import time
from gym import utils
import os

#np.set_printoptions(suppress=True)


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
        has_viewer=False
    ):  
        self.seed()
        self.mjpy_model = mujoco_py.load_model_from_path("/home/clothmanip/school/osc_ws/src/osc/mujoco_models/laptop/testmodel_cloth_10ms.xml")
        self.sim = mujoco_py.MjSim(self.mjpy_model)
        if has_viewer:
            self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self.viewer.cam.distance = 1
            self.viewer.cam.azimuth = 160
            self.viewer.cam.lookat[0] = 0
            self.viewer.cam.lookat[1] = 0
            self.viewer.cam.lookat[2] = 0.2
        else:
            self.viewer = None
        #TODO: fix in image train
        

        self.single_goal_dim = 3
        #TODO: parametrize ctrl frequency
        self.substeps = 10
        self.between_steps = 10
        self.filter = 0.03

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

        self.cloth_site_names = ["S0_0", "S4_0", "S8_0", "S0_4","S0_8", "S4_8", "S8_8", "S8_4"]
        self.ee_site_name = 'gripper0_grip_site'
        self.joints = ["robot0_joint1", "robot0_joint2", "robot0_joint3", "robot0_joint4", "robot0_joint5", "robot0_joint6", "robot0_joint7"]
        self.joint_indexes = [self.sim.model.joint_name2id(joint) for joint in self.joints]
        self.joint_pos_addr = [self.sim.model.get_joint_qpos_addr(joint) for joint in self.joints]
        self.joint_vel_addr = [self.sim.model.get_joint_qvel_addr(joint) for joint in self.joints]
        self.ee_site_adr = mujoco_py.functions.mj_name2id(self.sim.model, 6, "gripper0_grip_site")
        self.initial_qpos = [0.149, - 0.134, 0.021, - 2.596, - 0.043, 2.458, - 0.808]
        self.min_damping = 0.0001  # TODO: pass ranges in from outside
        self.max_damping = 0.02

        self.min_stiffness = 0.0001  # TODO: pass ranges in from outside
        self.max_stiffness = 0.02

        self.min_geom_size = 0.005  # TODO: pass ranges in from outside
        self.max_geom_size = 0.011

        self.limits_min = [-0.05, -0.3, 0.0]
        self.limits_max = [0.25, 0.1, 0.3]

        self.episode_success = False

        self.current_geom_size = self.max_geom_size/2

        self.current_joint_stiffness = self.max_stiffness/2
        self.current_joint_damping = self.max_damping/2

        self.current_tendon_stiffness = self.max_stiffness/2
        self.current_tendon_damping = self.max_damping/2

        self.set_joint_tendon_params(self.current_joint_stiffness, self.current_joint_damping,
                                        self.current_tendon_stiffness, self.current_tendon_damping)

        for j, joint in enumerate(self.joints):
            self.sim.data.set_joint_qpos(joint, self.initial_qpos[j])

        for _ in range(10):
            self.sim.forward()

        for _ in range(30):
            mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
            self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr]
            mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)

        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.ctrl_goals = []
        self.pos_goals = []
        self.ee_positions = []
        self.reset_osc()
        

        self.goal = self._sample_goal()

        self.task_reward_function = reward_calculation.get_task_reward_function(
            self.constraints, self.single_goal_dim, self.sparse_dense)

        self.action_space = gym.spaces.Box(-1,
                                           1, shape=(3,), dtype='float32')
        obs = self._get_obs()

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


    def reset_osc(self):
        if len(self.ee_positions) > 0:
            traj = np.array([[self.ctrl_goals, self.ctrl_goals, self.ee_positions]]).reshape((-1,9))
            np.savetxt(f"./eval_trajs/{str(time.time())}-traj.csv",
                   traj, delimiter=",", fmt='%f')
        self.initial_O_T_EE = None
        self.initial_joint_osc = None
        self.initial_ee_p = None
        self.desired_pos_step = None
        self.desired_pos_ctrl = None
        self.ctrl_goals = []
        self.pos_goals = []
        self.ee_positions = []
        self.update_osc_vals()

    def reset(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        self.reset_osc()
        self.goal = self._sample_goal()
        if not self.viewer is None:
            del self.viewer._markers[:]
        self.episode_success = False


        '''
        print("RESET")
        print(self.goal)
        print(self._get_ee_position())
        '''

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

        return self._get_obs()

    def run_controller(self):
        #print("OSC STEP START", os.getpid())
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
                                                10.0,
                                                1000,
                                                1000,
                                                1.0
                                                )
        #print("OSC STEP END", os.getpid())
        torques = tau.flatten()
        return torques

    def step_env(self, i):
        #print("MJSTEP1 START", i, os.getpid())
        mujoco_py.functions.mj_step1(self.sim.model, self.sim.data)
        #print("MJSTEP1 DONE", i, os.getpid())
        self.update_osc_vals()
        tau = self.run_controller()
        self.sim.data.qfrc_applied[self.joint_vel_addr] = self.sim.data.qfrc_bias[self.joint_vel_addr] + tau
        #print("MJSTEP2 START", i, os.getpid())
        mujoco_py.functions.mj_step2(self.sim.model, self.sim.data)
        #print("MJSTEP2 START", i, os.getpid())

    def step(self, action, evaluation=False):
        if evaluation:
            self.ee_positions.append(self._get_ee_position())
            self.ctrl_goals.append(self.desired_pos_ctrl)
            self.pos_goals.append(self.desired_pos_step)
        act = action.copy()*self.max_advance
        desired_pos_step_absolute = self.desired_pos_step + act
        min_absolute = self.initial_ee_p + self.limits_min
        max_absolute = self.initial_ee_p + self.limits_max
        self.desired_pos_step = np.clip(desired_pos_step_absolute, min_absolute, max_absolute)
        for i in range(self.substeps):
            for j in range(self.between_steps):
                self.desired_pos_ctrl = self.filter*self.desired_pos_step + (1-self.filter)*self.desired_pos_ctrl
            self.step_env(i)
        '''
        print("Step")
        print(action.copy())
        print(act)
        print(self.desired_pos_step)
        print(self.desired_pos_ctrl)
        print(self._get_ee_position())
        print("\n")
        '''

        #print("GET OBS", os.getpid())
        obs = self._get_obs()
        #print("POST ACT", os.getpid())
        reward, done, info = self._post_action(obs,evaluation=evaluation)
        #print("RETURN", os.getpid())
        return obs, reward, done, info

    def compute_task_reward(self, achieved_goal, desired_goal, info):
        return self.task_reward_function(achieved_goal, desired_goal, info)


        
    def _post_action(self, obs, evaluation=False):
        if evaluation:
            del self.viewer._markers[:]
            for i in range(int(self.goal.shape[0]/3)):
                self.viewer.add_marker(size=np.array([.001, .001, .001]), pos=self.goal[i*self.single_goal_dim: (i+1) *
                    self.single_goal_dim], label="d"+str(i))
                self.viewer.add_marker(size=np.array([.001, .001, .001]), pos=obs['achieved_goal'][i*self.single_goal_dim: (i+1) *
                    self.single_goal_dim], label="a"+str(i))



        base_task_reward = self.compute_task_reward(np.reshape(
            obs['achieved_goal'], (1, -1)), np.reshape(self.goal, (1, -1)), dict(real_sim=True))[0]

        # Use values in controller's frame
        error_norm = np.linalg.norm(self.desired_pos_step - self._get_ee_position())
        control_penalty = -error_norm

        task_reward = base_task_reward + self.reward_offset
        reward = task_reward # + control_penalty

        success = False
        if base_task_reward > -1:
            success = True
            if not self.episode_success:
                print("Real sim success",
                      np.round(reward, decimals=1),
                      np.round(task_reward, decimals=1),
                      np.round(base_task_reward, decimals=1),
                      np.round(task_reward, decimals=1),
                      np.round(control_penalty, decimals=1),
                      "Joint values:",
                      np.round(self.current_joint_damping, decimals=4),
                      np.round(self.current_joint_stiffness, decimals=4))
                self.episode_success = True

        corner_positions = np.array([])
        cloth_positions = self._get_cloth_position()
        for site in self.cloth_site_names:
            if not "4" in site:
                corner_positions = np.concatenate(
                    [corner_positions, cloth_positions[site]])

        info = {"base_task_reward": base_task_reward,
                "task_reward": task_reward,
                "control_penalty": control_penalty,
                "reward": reward,
                'is_success': success,
                'squared_error_norm': error_norm**2,
                'corner_positions': corner_positions}

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
            self.joint_pos_osc[j] = self.sim.data.qpos[self.joint_pos_addr[j]]
            self.joint_vel_osc[j] = self.sim.data.qvel[self.joint_vel_addr[j]]

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

    def _get_ee_position(self):
        return self.sim.data.get_site_xpos(self.ee_site_name).copy()

    def _get_ee_velocity(self):
        return self.sim.data.get_site_xvelp(self.ee_site_name).copy()

    def _get_cloth_position(self):
        positions = dict()
        for site in self.cloth_site_names:
            positions[site] = self.sim.data.get_site_xpos(site).copy()
        return positions

    def _get_cloth_velocity(self):
        velocities = dict()
        for site in self.cloth_site_names:
            velocities[site] = self.sim.data.get_site_xvelp(site).copy()
        return velocities

    def _get_obs(self):
        achieved_goal = np.zeros(self.single_goal_dim*len(self.constraints))
        for i, constraint in enumerate(self.constraints):
            origin = constraint['origin']
            achieved_goal[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = self.sim.data.get_site_xpos(
                origin).copy()

        cloth_position = np.array(list(self._get_cloth_position().values()))

        robot_position = self._get_ee_position()

        if self.velocity_in_obs:
            cloth_velocity = np.array(list(self._get_cloth_velocity(
            ).values()))
            robot_velocity = self._get_ee_velocity()
            observation = np.concatenate([cloth_position, cloth_velocity])
            robot_observation = np.concatenate(
                [robot_position, robot_velocity])
        else:
            observation = cloth_position
            robot_observation = robot_position

        controller_goal = self.desired_pos_step
        robot_observation = np.concatenate(
            [robot_observation, controller_goal])  # To get an idea of the current goal traj

        if self.randomize_geoms and self.randomize_params:
            model_params = np.array([self.current_joint_stiffness, self.current_joint_damping,
                                     self.current_tendon_stiffness, self.current_tendon_damping, self.current_geom_size])
        elif self.randomize_params:
            model_params = np.array([self.current_joint_stiffness, self.current_joint_damping,
                                     self.current_tendon_stiffness, self.current_tendon_damping])
        else:
            model_params = np.array([0])

        full_observation = {
            'observation': observation.flatten().copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'model_params': model_params.copy(),
            'robot_observation': robot_observation.flatten().copy()
        }

        if self.pixels:
            camera_id = self.sim.model.camera_name2id(
                'clothview2')  # TODO: parametrize camera
            self.sim._render_context_offscreen.render(
                self.image_size, self.image_size, camera_id)
            image_obs = self.sim._render_context_offscreen.read_pixels(
                self.image_size, self.image_size, depth=False)
            image_obs = image_obs[::-1, :, :]
            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)
            full_observation['image'] = (image_obs / 255).flatten()
        return full_observation

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def _sample_goal(self):
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
                 self.single_goal_dim] = target_pos + offset

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

    def set_aux_positions(self, corner1, corner2, corner3, corner4):
        self.viewer.add_marker.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner1, label="corner1")
        self.viewer.add_marker.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner2, label="corner2")
        self.viewer.add_marker.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner3, label="corner3")
        self.viewer.add_marker.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner4, label="corner4")

    def clear_aux_positions(self):
        del self.viewer._markers[:]



class ClothEnvPickled(ClothEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        print("Creating pickled env")
        ClothEnv.__init__(
            self, **kwargs)
        utils.EzPickle.__init__(self)