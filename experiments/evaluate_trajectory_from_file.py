import numpy as np
import matplotlib.pyplot as plt

ee_reached = np.genfromtxt('/home/clothmanip/school/osc_ws/src/osc/trajectories/ee_reached.csv', delimiter=',')
ee_targets = np.genfromtxt('/home/clothmanip/school/osc_ws/src/osc/trajectories/ee_targets.csv', delimiter=',')

def main():
    fig = plt.figure(figsize=(30, 30))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.view_init(10, -10)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    #ax1.plot(np.ones(5), np.ones(5), np.ones(5), linewidth=3, label="achieved", color="blue")
    ax1.plot(ee_reached[:, 0], ee_reached[:,1], ee_reached[:, 2], linewidth=3, label="achieved", color="blue")
    ax1.plot(ee_targets[:, 0], ee_targets[:,1], ee_targets[:, 2], linewidth=3, label="target", color="orange")

    ax1.text(ee_reached[0,0], ee_reached[0,1],
                ee_reached[0,2], "start", size=10, zorder=1, color='k')

    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()