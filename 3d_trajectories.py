from posix import listdir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os




def get_stack(filename):
    fixed_df = pd.read_csv(filename)
    stack = np.zeros(3)
    for row in range(len(fixed_df)):
        strr = fixed_df["desired_pos_step_I"][row].strip("[]")
        #strr = fixed_df["ee_position_I"][row].strip("[]")
        arr = [float(st) for st in strr.split(" ") if not st == '']
        add = np.array(arr)
        stack = np.vstack([stack, add])
    return stack


plus = False

'''
if plus:
    orange = pd.read_csv(
        "/home/julius/robotics/real_runs/whenever-4cm-fs1-multi-ctrl-two-run-0-99/999/real_trajectory.csv", header=None)
    white = pd.read_csv(
        "/home/julius/robotics/real_runs/whenever-4cm-fs1-multi-ctrl-two-run-0-99/9999/real_trajectory.csv", header=None)
    blue = pd.read_csv(
        "/home/julius/robotics/real_runs/whenever-4cm-fs1-multi-ctrl-two-run-0-99/99/real_trajectory.csv", header=None)
else:
    orange = pd.read_csv(
        "/home/julius/robotics/real_runs/emo-4cm-fs1-single-ctrl-two-run-0-100/999/real_trajectory.csv", header=None)
    white = pd.read_csv(
        "/home/julius/robotics/real_runs/emo-4cm-fs1-single-ctrl-two-run-0-100/9999/real_trajectory.csv", header=None)
    blue = pd.read_csv(
        "/home/julius/robotics/real_runs/emo-4cm-fs1-single-ctrl-two-run-0-100/99/real_trajectory.csv", header=None)
'''

#ax.plot(orange[0]-orange[0][0], orange[1]-orange[1][0], orange[2]-orange[2][0], label="Orange dels")
#ax.plot(white[0]-white[0][0], white[1]-white[1][0], white[2]-white[2][0], label="White dels")
#ax.plot(blue[0]-blue[0][0], blue[1]-blue[1][0], blue[2]-blue[2][0], label="Blue dels")
#ax.plot(orange[6], orange[7], orange[8], label="Orange ee")
#ax.plot(white[6], white[7], white[8], label="White ee")
#ax.plot(blue[6], blue[7], blue[8], label="Blue ee")



#for fname, title in zip(["/home/julius/robotics/real_runs/thue-4cm-fullstate-run-0-80/0_constant_cloth_trajectory.csv", "/home/julius/robotics/real_runs/whenever-4cm-fs1-multi-ctrl-two-run-0-99/0_constant_cloth_trajectory.csv", "/home/julius/robotics/real_runs/emo-4cm-fs1-single-ctrl-two-run-0-100/0_constant_cloth_trajectory.csv"],["simfixed", "simours+", "simours"]):
#roo = "/home/julius/robotics/real_runs/whenever-4cm-fs1-multi-ctrl-two-run-0-99/policy_eval_runs"
#roo = "/home/julius/robotics/real_runs/emo-4cm-fs1-single-ctrl-two-run-0-100/policy_eval_runs_norand"
#roo = "/home/julius/robotics/real_runs/emo-4cm-fs1-single-ctrl-two-run-0-100/policy_eval_runs_rand"
#roo = "/home/julius/robotics/real_runs/thue-4cm-fullstate-run-0-646/policy_eval_runs_norand"
roo = "/home/julius/robotics/real_runs/thue-4cm-fullstate-run-0-646/policy_eval_runs_rand"
evals = [pat for pat in os.listdir(roo) if os.path.isdir(os.path.join(roo, pat)) and "trajectory.csv" in os.listdir(os.path.join(roo, pat))]
#for i, dir in enumerate(evals):
#    stack = get_stack(os.path.join(roo, dir, "trajectory.csv"))
#    ax.plot(stack[:,0], stack[:,1], stack[:,2], label=str(i))
run_fldr1 = "/home/julius/robotics/real_runs/whenever-4cm-fs1-multi-ctrl-two-run-0-99"
run_fldr2 = "/home/julius/robotics/real_runs/emo-4cm-fs1-single-ctrl-two-run-0-100"
run_fldr3 = "/home/julius/robotics/real_runs/thue-4cm-fullstate-run-0-646"
cloths = ["white", "orange", "blue"]
runs = [run_fldr1, run_fldr2, run_fldr3]
for run_fldr in runs:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.2, 0.0)
    ax.set_zlim(0.0, 0.2)
    for cloth in cloths:
        for i in range(10):
            traj_fodl = os.path.join(run_fldr, cloth, str(i), "real_trajectory.csv")
            traj = pd.read_csv(traj_fodl, header=None)
            if cloth == "white":
                color = "gray"
            else:
                color = cloth
            ax.plot(traj[0]-traj[0][0], traj[1]-traj[1][0], traj[2]-traj[2][0], label=f"{cloth}_{i}", color=color)
    plt.legend()
plt.show()




