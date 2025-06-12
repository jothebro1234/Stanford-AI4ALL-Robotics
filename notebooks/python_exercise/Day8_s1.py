import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# [1] Linear interpolation function
# -------------------------------
def lerp(start, end, alpha):
    """Linear interpolation between start and end by alpha"""
    return (1 - alpha) * start + alpha * end

# -------------------------------
# [2] Initialize PyBullet
# -------------------------------
p.connect(p.GUI)  # launch GUI window
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.8)

# Load ground and robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# Get joint indices for revolute (controllable) joints
num_joints = p.getNumJoints(robot_id)
joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE]

# -------------------------------
# [3] Define start and goal joint configurations
# -------------------------------
start = np.zeros(len(joint_indices))  # all joints at 0 rad
goal = np.array([0.5, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1])  # some arbitrary reachable pose

# -------------------------------
# [4] Plan and execute motion with linear interpolation
# -------------------------------
steps = 120  # number of interpolation steps
history = []  # to record joint angles over time for plotting

print("Executing motion from START to GOAL...")

for step in range(steps):
    alpha = step / (steps - 1)  # goes from 0 → 1
    current = lerp(start, goal, alpha)  # interpolate joint angles
    history.append(current.copy())  # store for visualization

    # Send commands to robot
    for i, j in enumerate(joint_indices):
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL,
                                targetPosition=current[i], force=80)
    p.stepSimulation()
    time.sleep(1./240.)

# Optional pause
time.sleep(1)
p.disconnect()

# -------------------------------
# [5] Plot joint angles over time
# -------------------------------
history = np.array(history)  # convert to numpy for plotting

plt.figure(figsize=(10, 4))
for j in range(history.shape[1]):
    plt.plot(history[:, j], label=f"Joint {j}")

plt.title("Joint Angle Trajectory (Linear Interpolation)")
plt.xlabel("Step (time)")
plt.ylabel("Joint Angle (radians)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
