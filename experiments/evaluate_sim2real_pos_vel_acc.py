import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(real_folder, sim_folder, metric):

    assert metric == "pos" or metric == "vel" or metric == "acc"

    sim_trajectory = np.genfromtxt(f"{sim_folder}/sim_trajectory.csv", delimiter=',')
    sim_ee_pos = sim_trajectory[:, 9:12]
    sim_ee_desired_pos = sim_trajectory[:, 0:3]
    sim_ee_vel = sim_trajectory[:, 15:18]
    sim_ee_acc = sim_trajectory[:, 18:]

    real_trajectory = np.genfromtxt(f"{real_folder}/real_trajectory.csv", delimiter=',')
    real_ee_pos = real_trajectory[:, 9:12]
    real_ee_desired_pos = real_trajectory[:, 0:3]
    real_ee_vel = real_trajectory[:, 15:18]
    real_ee_acc = real_trajectory[:, 18:]


    fig, axs = plt.subplots(3)

    alignment =  np.argmax(real_ee_pos[:,2]) - np.argmax(sim_ee_pos[:,2])

    if metric == "pos":
        X_sim = sim_ee_pos[:,0]
        Y_sim = sim_ee_pos[:,1]
        Z_sim = sim_ee_pos[:,2]

        X_real = real_ee_pos[:,0]
        Y_real = real_ee_pos[:,1]
        Z_real = real_ee_pos[:,2]

    elif metric == "vel":
        X_sim = sim_ee_vel[:,0]
        Y_sim = sim_ee_vel[:,1]
        Z_sim = sim_ee_vel[:,2]

        X_real = real_ee_vel[:,0]
        Y_real = real_ee_vel[:,1]
        Z_real = real_ee_vel[:,2]

    else:
        X_sim = sim_ee_acc[:,0]
        Y_sim = sim_ee_acc[:,1]
        Z_sim = sim_ee_acc[:,2]

        X_real = real_ee_acc[:,0]
        Y_real = real_ee_acc[:,1]
        Z_real = real_ee_acc[:,2]

    max_range = np.array([X_sim.max()-X_sim.min(), Y_sim.max()-Y_sim.min(), Z_sim.max()-Z_sim.min()]).max() / 1.9

    labels = ["X", "Y", "Z"]

    for i, (label, sim, real) in enumerate(zip(labels, [X_sim, Y_sim, Z_sim], [X_real, Y_real, Z_real])):
        mid = (sim.max()+sim.min()) * 0.5
        axs[i].set_ylim([mid-max_range,mid+max_range])
        axs[i].plot(sim, label=f"sim")
        axs[i].plot(real[alignment:], label=f"real")
        axs[i].set_title(f"{metric} {label} ")
        plt.legend()


    


    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('real_folder', type=str)
    parser.add_argument('sim_folder', type=str)
    parser.add_argument('metric', type=str)
    args = parser.parse_args()

    main(args.real_folder, args.sim_folder, args.metric)