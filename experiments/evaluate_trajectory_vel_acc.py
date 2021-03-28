


import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot(ax, data, label):
    ax.plot(range(data.shape[0]), data, label=label)


def main(args):
    sim = bool(args.sim)
    real = bool(args.real)

    if sim:
        ee_trajectory_sim = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_trajectory_sim.csv', delimiter=',')
        ee_trajectory_desired_sim = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_trajectory_desired_sim.csv', delimiter=',')

    if real:
        ee_trajectory_real = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_trajectory_real.csv', delimiter=',')
        ee_trajectory_desired_real = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_trajectory_desired_real.csv', delimiter=',')
    

    fig, axs = plt.subplots(9)

    for i, ax in enumerate(axs):
        if sim:
            plot(ax, ee_trajectory_sim[:,i], "sim")
        if real:
            plot(ax, ee_trajectory_real[:,i], "real")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('--sim',  default=0, type=int)
    parser.add_argument('--real',  default=0, type=int)
    args = parser.parse_args()
    main(args)