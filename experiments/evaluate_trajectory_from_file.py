import numpy as np
import matplotlib.pyplot as plt
import argparse

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def main(args):
    start = args.start

    ee_reached = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_reached.csv', delimiter=',')
    ee_reached_sim = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_reached_sim.csv', delimiter=',')
    ee_deltas = np.genfromtxt('/home/julius/robotics/osc_ws/src/real_nice_deltas.csv', delimiter=',')
    ee_targets = np.array([np.sum(ee_deltas[:i], axis=0) for i in range(ee_deltas.shape[0])])

    if bool(args.save_as_deltas):
        deltas =  []
        for idx in range(start, ee_reached.shape[0]-1):
            delta = ee_reached[idx+1]-ee_reached[idx]
            deltas.append(delta)
        deltas = np.array(deltas)
        np.savetxt("/home/julius/robotics/osc_ws/src/real_deltas.csv",
                    deltas, delimiter=",", fmt='%f')
        print("Saved real traj as deltas")

    if bool(args.save_sim_as_deltas):
        deltas =  []
        for idx in range(start, ee_reached_sim.shape[0]-1):
            delta = ee_reached_sim[idx+1]-ee_reached_sim[idx]
            deltas.append(delta)
        deltas = np.array(deltas)
        np.savetxt("/home/julius/robotics/osc_ws/src/sim_deltas.csv",
                    deltas, delimiter=",", fmt='%f')
        print("Saved sim traj as deltas")

    fig = plt.figure()
    ax1 = Axes3D(fig)
    ax1.set_xlim3d(-0.1, 0.1)
    ax1.set_ylim3d(-0.18, 0.02)
    ax1.set_zlim3d(-0.02, 0.18)
    ax1.set_box_aspect((1, 1, 1))

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.plot(ee_reached[:, 0], ee_reached[:, 1], ee_reached[:, 2], linewidth=3, label="achieved real", color="blue")
    ax1.plot(ee_reached_sim[:, 0], ee_reached_sim[:, 1], ee_reached_sim[:, 2], linewidth=3, label="achieved sim", color="green")
    ax1.plot(ee_targets[:, 0], ee_targets[:,1], ee_targets[:, 2], linewidth=3, label="target", color="orange")
    ax1.text(ee_reached[0,0], ee_reached[0,1],
                ee_reached[0,2], "start", size=10, zorder=1, color='k')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('--save_as_deltas',  default=0, type=int)
    parser.add_argument('--save_sim_as_deltas',  default=0, type=int)
    parser.add_argument('--start',  default=0, type=int)
    args = parser.parse_args()
    main(args)