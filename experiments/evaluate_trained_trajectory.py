import numpy as np
import matplotlib.pyplot as plt
import argparse

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



def main(args):
    fig = plt.figure()
    ax1 = Axes3D(fig)

    traj = np.genfromtxt(args.filename, delimiter=',')

    rang = 0.2
    x_min = np.min(traj[:, 0])
    y_min = np.min(traj[:, 1])
    z_min = np.min(traj[:, 2])

    #ax1.set_xlim3d(x_min, x_min+rang)
    #ax1.set_ylim3d(y_min, y_min+rang)
    #ax1.set_zlim3d(z_min, z_min+rang)
    #ax1.set_box_aspect((1, 1, 1))

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=3, label="pos_des")
    ax1.plot(traj[:, 3], traj[:, 4], traj[:, 5], linewidth=3, label="ctrl_des")
    ax1.plot(traj[:, 6], traj[:, 7], traj[:, 8], linewidth=3, label="pos")
    ax1.text(traj[0, 0], traj[0, 1], traj[0, 2], "start", size=10, zorder=1, color='k')

    plt.legend()
    plt.show()








if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('filename')
    args = parser.parse_args()
    main(args)