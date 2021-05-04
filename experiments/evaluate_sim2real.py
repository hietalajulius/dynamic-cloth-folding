from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(folder):

    sim_trajectory = np.genfromtxt(f"{folder}/sim_trajectory.csv", delimiter=',')
    sim_ee_pos = sim_trajectory[:, 9:12]
    sim_ee_desired = sim_trajectory[:, 0:3]

    real_trajectory = np.genfromtxt(f"{folder}/real_trajectory.csv", delimiter=',')
    real_ee_pos = real_trajectory[:, 9:12]
    real_ee_desired = real_trajectory[:, 0:3]

    print("shapes", sim_trajectory.shape, real_trajectory.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_box_aspect((1,1,1))

    X = sim_ee_pos[:,0]
    Y = sim_ee_pos[:,1]
    Z = sim_ee_pos[:,2]

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.scatter(sim_ee_pos[:, 0], sim_ee_pos[:, 1], sim_ee_pos[:,
                                                            2], linewidth=1, label="sim")
    ax.scatter(sim_ee_desired[:, 0], sim_ee_desired[:, 1], sim_ee_desired[:,
                                                            2], linewidth=1, label="sim des")

    ax.scatter(real_ee_pos[:, 0], real_ee_pos[:, 1], real_ee_pos[:,
                                                            2], linewidth=1, label="real")

    ax.scatter(real_ee_desired[:, 0], real_ee_desired[:, 1], real_ee_desired[:,
                                                            2], linewidth=1, label="real des")



    plt.legend()
    plt.show()









if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('folder', type=str)
    args = parser.parse_args()

    main(args.folder)