import matplotlib.pyplot as plt
import pandas as pd

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])

orange = pd.read_csv(
    "/Users/juliushietala/robotics/paper/paper_policies/whenever-4cm-fs1-multi-ctrl-two-run-0-99/orange/0/real_trajectory.csv", header=None)
white = pd.read_csv(
    "/Users/juliushietala/robotics/paper/paper_policies/whenever-4cm-fs1-multi-ctrl-two-run-0-99/white/0/real_trajectory.csv", header=None)
blue = pd.read_csv(
    "/Users/juliushietala/robotics/paper/paper_policies/whenever-4cm-fs1-multi-ctrl-two-run-0-99/blue/0/real_trajectory.csv", header=None)

print(orange.columns)
print(orange[3])

ax.set_xlim(-0.1, 0.1)
ax.set_ylim(-0.2, 0.0)
ax.set_zlim(0.0, 0.2)


ax.plot(orange[6], orange[7], orange[8], label="Orange", color="orange")
ax.plot(white[6], white[7], white[8], label="White", color="grey")
ax.plot(blue[6], blue[7], blue[8], label="Blue", color="blue")
plt.legend()
plt.show()
