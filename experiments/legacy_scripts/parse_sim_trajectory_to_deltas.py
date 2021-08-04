import argparse
import numpy as np



def main(input_folder):
    trajectory_path = input_folder + "/sim_trajectory.csv"
    trajectory = np.genfromtxt(trajectory_path, delimiter=',')
    deltas = trajectory[:, 12:15]

    np.savetxt(f"{input_folder}/parsed_executable_deltas.csv", deltas, delimiter=",", fmt='%f')






if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument('input_folder', type=str)

    args = parser.parse_args()
    
    main(args.input_folder)
