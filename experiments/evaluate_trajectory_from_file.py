import numpy as np
import matplotlib.pyplot as plt
import argparse

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

prefix = "/home/julius/robotics/" #/home/clothmanip/school/" #

def main(args):
    sim = bool(args.sim)
    real = bool(args.real)
    save_as_deltas = bool(args.save_as_deltas)
    sim_comparison = bool(args.sim_comparison)

    fig = plt.figure()
    ax1 = Axes3D(fig)
    

    ee_deltas = np.genfromtxt(prefix+'cloth-manipulation/traj_opt/actual_deltas_and_inferred_velocities.csv', delimiter=',')
    ee_targets = np.array([np.sum(ee_deltas[:i,:3], axis=0) for i in range(ee_deltas.shape[0])])



    rang = 0.2
    x_min = np.min(ee_targets[:, 0])
    y_min = np.min(ee_targets[:, 1])
    z_min = np.min(ee_targets[:, 2])
    
    ax1.set_xlim3d(x_min, x_min+rang)
    ax1.set_ylim3d(y_min, y_min+rang)
    ax1.set_zlim3d(z_min, z_min+rang)
    ax1.set_box_aspect((1, 1, 1))

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.plot(ee_targets[:, 0], ee_targets[:,1], ee_targets[:, 2], linewidth=3, label="target", color="orange")
    ax1.text(ee_targets[0,0], ee_targets[0,1],
                ee_targets[0,2], "start", size=10, zorder=1, color='k')

    if sim:
        if save_as_deltas:
            deltas =  []
            for idx in range(ee_trajectory_sim.shape[0]-1):
                delta_pos = ee_trajectory_sim[idx+1,:3]-ee_trajectory_sim[idx,:3]
                velocity_at_end = ee_trajectory_sim[idx+1,3:6]
                deltas.append(np.concatenate([delta_pos, velocity_at_end]))
            deltas = np.array(deltas)
            np.savetxt(prefix+"osc_ws/src/sim_deltas_velocities.csv",
                        deltas, delimiter=",", fmt='%f')
            print("Saved sim traj as deltas")
        ax1.plot(ee_trajectory_sim[:, 0], ee_trajectory_sim[:, 1], ee_trajectory_sim[:, 2], linewidth=3, label="achieved sim", color="green")

    if real:
        ee_trajectory_real = np.genfromtxt(prefix+'osc_ws/src/osc/trajectories/ee_trajectory_real.csv', delimiter=',')
        if save_as_deltas:
            deltas =  []
            for idx in range(ee_trajectory_real.shape[0]-1):
                delta_pos = ee_trajectory_real[idx+1,:3]-ee_trajectory_real[idx,:3]
                velocity_at_end = ee_trajectory_real[idx+1,3:6]
                deltas.append(np.concatenate([delta_pos, velocity_at_end]))
            deltas = np.array(deltas)
            np.savetxt(prefix+"osc_ws/src/real_deltas_velocities.csv",
                        deltas, delimiter=",", fmt='%f')
            print("Saved real traj as deltas")

        ax1.plot(ee_trajectory_real[:, 0], ee_trajectory_real[:, 1], ee_trajectory_real[:, 2], linewidth=3, label="achieved real", color="blue")

    if sim_comparison:
        reg_cloth = np.genfromtxt(prefix+'osc_ws/src/osc/trajectories/ee_trajectory_sim_regular__cloth.csv', delimiter=',')
        slow_cloth = np.genfromtxt(prefix+'osc_ws/src/osc/trajectories/ee_trajectory_sim_slower__cloth.csv', delimiter=',')
        reg_no_cloth = np.genfromtxt(prefix+'osc_ws/src/osc/trajectories/ee_trajectory_sim_regular__no_cloth.csv', delimiter=',')
        slow_no_cloth = np.genfromtxt(prefix+'osc_ws/src/osc/trajectories/ee_trajectory_sim_slower__no_cloth.csv', delimiter=',')

        print("HAPES", reg_cloth.shape, slow_cloth.shape, reg_no_cloth.shape, slow_no_cloth.shape)

        ax1.plot(reg_cloth[:, 0], reg_cloth[:, 1]+0.005, reg_cloth[:, 2], linewidth=2, label="reg_cloth")
        ax1.plot(slow_cloth[:, 0], slow_cloth[:, 1]+0.01, slow_cloth[:, 2], linewidth=2, label="slow_cloth")
        ax1.plot(reg_no_cloth[:, 0], reg_no_cloth[:, 1]+0.015, reg_no_cloth[:, 2], linewidth=2, label="reg_no_clot")
        ax1.plot(slow_no_cloth[:, 0], slow_no_cloth[:, 1]+0.02, slow_no_cloth[:, 2], linewidth=2, label="slow_no_cloth")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('--save_as_deltas',  default=0, type=int)
    parser.add_argument('--sim',  default=0, type=int)
    parser.add_argument('--sim_comparison',  default=0, type=int)
    parser.add_argument('--real',  default=0, type=int)
    args = parser.parse_args()
    main(args)