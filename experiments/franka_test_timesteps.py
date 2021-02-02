from rlkit.envs.wrappers import NormalizedBoxEnv
import gym
from utils import get_variant, argsparser
import numpy as np
import time
import copy


args = argsparser()

args1 = copy.deepcopy(args)
args1.ctrl_frequency = 10
args1.model_timestep = 0.001

args2 = copy.deepcopy(args)
args2.ctrl_frequency = 10
args2.model_timestep = 0.01


variant1 = get_variant(args1)
variant2 = get_variant(args2)


env1 = gym.make('Franka-v1', **variant1['env_kwargs'])
env2 = gym.make('Franka-v1', **variant2['env_kwargs'])

start = time.time()
epis = 50
while True:
    env1.reset()
    env2.reset()
    for i in range(epis):
        print(i)
        # action = np.random.uniform(-10, 10, (7,))
        action = -np.zeros(7)
        action = -np.ones(7)*0.05
        #action[3] = 0.075

        _ = env1.step(action)
        _ = env2.step(action)

        # env2.render()
        # env1.render()

        pos_diff_vec = env1.sim.data.get_site_xpos(
            'panda0_end_effector').copy()-env2.sim.data.get_site_xpos(
            'panda0_end_effector').copy()

        print("Pos diff cartesian", np.linalg.norm(pos_diff_vec), pos_diff_vec)

        joint_vals_1 = []
        joint_vals_2 = []

        commanded_1 = []
        commanded_2 = []
        for i in range(action.shape[0]):
            idx_1 = env1.sim.model.jnt_qposadr[env1.sim.model.actuator_trnid[i, 0]]
            idx_2 = env2.sim.model.jnt_qposadr[env2.sim.model.actuator_trnid[i, 0]]

            joint_vals_1.append(env1.sim.data.qpos[idx_1].copy())
            joint_vals_2.append(env2.sim.data.qpos[idx_2].copy())

            commanded_1 = env1.sim.data.ctrl[idx_1].copy()
            commanded_2 = env2.sim.data.ctrl[idx_2].copy()

        joint_vals_1 = np.array(joint_vals_1)
        joint_vals_2 = np.array(joint_vals_2)

        commanded_1 = np.array(commanded_1)
        commanded_2 = np.array(commanded_2)

        # print("acc 1", com1_diff_vec)
        # print("acc 2", com2_diff_vec)
        com1_diff = np.linalg.norm(joint_vals_1-commanded_1)
        com2_diff = np.linalg.norm(joint_vals_2-commanded_2)

        print("Commands equal", commanded_1 == commanded_2)
        diff = np.sum(abs(joint_vals_1-commanded_1)) - \
            np.sum(abs(joint_vals_2-commanded_2))
        diff_norm = com1_diff - com2_diff
        if com1_diff < com2_diff:
            print("env 1 closer", diff, diff_norm)
        else:
            print("env 2 closer", diff, diff_norm)
        print("diff 1 vec from commanded", np.round(
            joint_vals_1-commanded_1, decimals=2))
        print("diff 2 vec from commanded", np.round(
            joint_vals_2-commanded_2, decimals=2))
        print("Difference vector", abs(
            np.round(joint_vals_1-joint_vals_2, decimals=2)))
        # print("Position diff norm", np.linalg.norm(pos_diff_vec))
        # print("Actuator command norm", np.linalg.norm(actuator_diff_vec))
        # env.render(mode='rgb_array')
        # env.render()
    print(f"Took to collect {epis}", time.time()-start)
