


import numpy as np
import matplotlib.pyplot as plt
import argparse

prefix = "/home/julius/robotics/" # "/home/clothmanip/school/" #

def plot(ax, data, label):
    ax.plot(range(data.shape[0]), data, label=label)

#/home/julius/robotics/cloth-manipulation/traj_opt/actual_deltas_and_inferred_velocities.csv
def main(args):
    sim = bool(args.sim)
    real = bool(args.real)
    desired = bool(args.desired)
    predefined = bool(args.predefined)
    ee_predefined_deltas = np.genfromtxt(prefix+'cloth-manipulation/traj_opt/actual_deltas_and_inferred_velocities.csv', delimiter=',')
    ee_predefined_targets = np.array([np.sum(ee_predefined_deltas[:i,:3], axis=0) for i in range(ee_predefined_deltas.shape[0])])
    ee_predefined_velocities = ee_predefined_deltas[:,3:]
    ee_predefined_accelerations = [np.zeros(3)] #TODO: Save these in the ctrl logging
    for idx in range(ee_predefined_velocities.shape[0]-1):
        delta_vel = (ee_predefined_velocities[idx+1]-ee_predefined_velocities[idx])
        ee_predefined_accelerations.append(delta_vel/0.001)
    
    ee_predefined_accelerations = np.array(ee_predefined_accelerations)
    ee_predefined_trajectory = np.concatenate([ee_predefined_targets, ee_predefined_velocities, ee_predefined_accelerations], axis=1)

    if sim:
        ee_trajectory_sim = np.genfromtxt(prefix+'osc_ws/src/osc/trajectories/ee_trajectory_sim.csv', delimiter=',')
        ee_trajectory_desired_sim = np.genfromtxt(prefix+'osc_ws/src/osc/trajectories/ee_trajectory_desired_sim.csv', delimiter=',')

    if real:
        ee_trajectory_real = np.genfromtxt(prefix+'osc_ws/src/osc/trajectories/ee_trajectory_real.csv', delimiter=',')
        ee_trajectory_desired_real = np.genfromtxt(prefix+'osc_ws/src/osc/trajectories/ee_trajectory_desired_real.csv', delimiter=',')
    

    fig, axs = plt.subplots(9)

    pos_range = 0.2
    vel_range = 0.5
    acc_range = 30

    x_min = np.min(ee_predefined_targets[:, 0])
    y_min = np.min(ee_predefined_targets[:, 1])
    z_min = np.min(ee_predefined_targets[:, 2])

    starts = np.array([x_min, y_min, z_min, -0.2, -0.05, -0.2, -15, -15, -15])
    ends = np.zeros(9)
    ends[:3] = starts[:3] + pos_range
    ends[3:6] = starts[3:6] + vel_range
    ends[6:] = starts[6:] + acc_range

    for i, ax in enumerate(axs):
        ax.set_ylim(starts[i], ends[i])
        if predefined:
            plot(ax, ee_predefined_trajectory[:,i], "predefined")
        if sim:
            plot(ax, ee_trajectory_sim[:,i], "sim actual")
            if desired:
                plot(ax, ee_trajectory_desired_sim[:,i], "sim desired")
        if real:
            plot(ax, ee_trajectory_real[:,i], "real actual")
            if desired:
                plot(ax, ee_trajectory_desired_real[:,i], "real desired")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('--sim',  default=0, type=int)
    parser.add_argument('--real',  default=0, type=int)
    parser.add_argument('--desired',  default=0, type=int)
    parser.add_argument('--predefined',  default=0, type=int)
    args = parser.parse_args()
    main(args)