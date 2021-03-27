


import numpy as np
import matplotlib.pyplot as plt





ee_velocities = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_velocities.csv', delimiter=',')
ee_velocities_sim = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_velocities_sim.csv', delimiter=',')
ee_accelerations = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_accelerations.csv', delimiter=',')
ee_accelerations_sim = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_accelerations_sim.csv', delimiter=',')
#ee_velocities_sim = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_velocities.csv', delimiter=',')
#ee_accelerations_sim = np.genfromtxt('/home/julius/robotics/osc_ws/src/osc/trajectories/ee_accelerations.csv', delimiter=',')


fig, axs = plt.subplots(6)

start = 18000
end = 20000

start_sim = 14300
end_sim = ee_accelerations_sim.shape[0]


axs[0].plot(range(ee_velocities[start:end,0].shape[0]), ee_velocities[start:end,0], label="real")
axs[0].plot(range(ee_velocities_sim[start_sim:end_sim,0].shape[0]), ee_velocities_sim[start_sim:end_sim,0], label="sim")

axs[1].plot(range(ee_velocities[start:end,1].shape[0]), ee_velocities[start:end,1], label="real")
axs[1].plot(range(ee_velocities_sim[start_sim:end_sim,1].shape[0]), ee_velocities_sim[start_sim:end_sim,1], label="sim")

axs[2].plot(range(ee_velocities[start:end,2].shape[0]), ee_velocities[start:end,2], label="real")
axs[2].plot(range(ee_velocities_sim[start_sim:end_sim,2].shape[0]), ee_velocities_sim[start_sim:end_sim,2], label="sim")

axs[3].plot(range(ee_accelerations[start:end,0].shape[0]), ee_accelerations[start:end,0], label="real")
axs[3].plot(range(ee_accelerations_sim[start_sim:end_sim,0].shape[0]), ee_accelerations_sim[start_sim:end_sim,0], label="sim")

axs[4].plot(range(ee_accelerations[start:end,1].shape[0]), ee_accelerations[start:end,1], label="real")
axs[4].plot(range(ee_accelerations_sim[start_sim:end_sim,1].shape[0]), ee_accelerations_sim[start_sim:end_sim,1], label="sim")

axs[5].plot(range(ee_accelerations[start:end,2].shape[0]), ee_accelerations[start:end,2], label="real")
axs[5].plot(range(ee_accelerations_sim[start_sim:end_sim,2].shape[0]), ee_accelerations_sim[start_sim:end_sim,2], label="sim")

plt.legend()
plt.show()