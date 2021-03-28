import numpy as np
import matplotlib.pyplot as plt
import argparse

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def main(args):
    sim = bool(args.sim)
    real = bool(args.real)
    save_as_deltas = bool(args.save_as_deltas)

    fig = plt.figure()
    ax1 = Axes3D(fig)
    ax1.set_xlim3d(-0.1, 0.1)
    ax1.set_ylim3d(-0.18, 0.02)
    ax1.set_zlim3d(-0.02, 0.18)
    ax1.set_box_aspect((1, 1, 1))

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ee_deltas = np.genfromtxt('/home/julius/robotics/osc_ws/src/real_deltas_velocities.csv', delimiter=',')
    ee_targets = np.array([np.sum(ee_deltas[:i,:3], axis=0) for i in range(ee_deltas.shape[0])])
    ax1.plot(ee_targets[:, 0], ee_targets[:,1], ee_targets[:, 2], linewidth=3, label="target", color="orange")
    ax1.text(ee_targets[0,0], ee_targets[0,1],
                ee_targets[0,2], "start", size=10, zorder=1, color='k')

    if sim:
        ee_trajectory_sim = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_trajectory_sim.csv', delimiter=',')
        if save_as_deltas:
            deltas =  []
            for idx in range(ee_trajectory_sim.shape[0]-1):
                delta_pos = ee_trajectory_sim[idx+1,:3]-ee_trajectory_sim[idx,:3]
                velocity_at_end = ee_trajectory_sim[idx+1,3:6]
                deltas.append(np.concatenate([delta_pos, velocity_at_end]))
            deltas = np.array(deltas)
            np.savetxt("/home/julius/robotics/osc_ws/src/sim_deltas_velocities.csv",
                        deltas, delimiter=",", fmt='%f')
            print("Saved sim traj as deltas")
        ax1.plot(ee_trajectory_sim[:, 0], ee_trajectory_sim[:, 1], ee_trajectory_sim[:, 2], linewidth=3, label="achieved sim", color="green")

    if real:
        ee_trajectory_real = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_trajectory_real.csv', delimiter=',')
        if save_as_deltas:
            deltas =  []
            for idx in range(ee_trajectory_real.shape[0]-1):
                delta_pos = ee_trajectory_real[idx+1,:3]-ee_trajectory_real[idx,:3]
                velocity_at_end = ee_trajectory_real[idx+1,3:6]
                deltas.append(np.concatenate([delta_pos, velocity_at_end]))
            deltas = np.array(deltas)
            np.savetxt("/home/julius/robotics/osc_ws/src/real_deltas_velocities.csv",
                        deltas, delimiter=",", fmt='%f')
            print("Saved real traj as deltas")

        ax1.plot(ee_trajectory_real[:, 0], ee_trajectory_real[:, 1], ee_trajectory_real[:, 2], linewidth=3, label="achieved real", color="blue")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('--save_as_deltas',  default=0, type=int)
    parser.add_argument('--sim',  default=0, type=int)
    parser.add_argument('--real',  default=0, type=int)
    args = parser.parse_args()
    main(args)