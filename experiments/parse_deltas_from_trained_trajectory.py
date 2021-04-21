import numpy as np
import matplotlib.pyplot as plt
import argparse

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



def main(args):
    #traj = np.concatenate([np.array(self.pos_goals), np.array(self.ctrl_goals), np.array(self.ee_positions)], axis=1)
    trajectory = np.genfromtxt(args.input, delimiter=',')
    deltas = []
    for idx in range(trajectory.shape[0]-1):
        delta_goal_pos = trajectory[idx + 1, 6:9]-trajectory[idx, 6:9]
        deltas.append(delta_goal_pos)

    deltas = np.array(deltas)
    np.savetxt(args.output, deltas, delimiter=",", fmt='%f')
    print("Saved traj as deltas")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    main(args)