from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils import calculate_ate, calculate_cosine_distances, deltas_from_positions

def main(real_folder, sim_folder):

    sim_trajectory = np.genfromtxt(f"{sim_folder}/sim_trajectory.csv", delimiter=',')
    sim_ee_pos = sim_trajectory[:, 9:12]
    sim_ee_desired_pos = sim_trajectory[:, 0:3]
    sim_ee_desired_ctrl = sim_trajectory[:, 3:6]

    sim_ee_pos_deltas = sim_trajectory[:, 12:15]
    sim_ee_ctrl_deltas = deltas_from_positions(sim_ee_desired_pos)

    real_trajectory = np.genfromtxt(f"{real_folder}/real_trajectory.csv", delimiter=',')
    real_ee_pos = real_trajectory[:, 9:12]
    real_ee_desired_pos = real_trajectory[:, 0:3]
    real_ee_desired_ctrl = real_trajectory[:, 3:6]

    real_ee_pos_deltas = real_trajectory[:, 12:15]
    real_ee_ctrl_deltas = deltas_from_positions(real_ee_desired_pos)


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

    ax.plot(sim_ee_pos[:, 0], sim_ee_pos[:, 1], sim_ee_pos[:,
                                                            2], linewidth=1, label="sim")
                                                    
    ax.plot(sim_ee_desired_pos[:, 0], sim_ee_desired_pos[:, 1], sim_ee_desired_pos[:,
                                                            2], linewidth=1, label="sim desired")
    
    '''
    ax.plot(real_ee_pos[:, 0], real_ee_pos[:, 1], real_ee_pos[:,
                                                            2], linewidth=1, label="real")
    ax.plot(real_ee_desired_pos[:, 0], real_ee_desired_pos[:, 1], real_ee_desired_pos[:,
                                                      2], linewidth=1, label="real desired")
    '''

    text_data = sim_ee_ctrl_deltas
    text_positions = sim_ee_desired_pos
    metric = calculate_cosine_distances(text_data)
    #metric = np.linalg.norm(text_data, axis=1)
    #for i, pos in enumerate(text_positions[1:-1]):
        #ax.text(pos[0], pos[1], pos[2], str(np.round(metric[i], decimals=4)))

    print("SIM ATE", calculate_ate(sim_ee_pos, sim_ee_desired_pos))
    print("REAL ATE", calculate_ate(real_ee_pos, real_ee_desired_pos))



    plt.legend()
    plt.show()









if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('real_folder', type=str)
    parser.add_argument('sim_folder', type=str)
    args = parser.parse_args()

    main(args.real_folder, args.sim_folder)