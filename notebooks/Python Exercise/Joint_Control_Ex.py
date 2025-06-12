
import pybullet as p
import pybullet_data
import time
import math

# Start simulation
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.resetSimulation()

# Load plane and robot
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("simple_arm.urdf", basePosition=[0.5, 0, 0], useFixedBase=True)
print("Robot ID:", robot_id)

# Get number of joints
num_joints = p.getNumJoints(robot_id)
print("Number of joints:", num_joints)

# Create sliders for joint control
slider_ids = []
for joint_idx in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_idx)
    joint_name = joint_info[1].decode("utf-8")
    slider = p.addUserDebugParameter(f"{joint_name} (Joint {joint_idx})", -3.14, 3.14, 0)
    slider_ids.append(slider)

# Main simulation loop
print("Adjust sliders to control joint angles. Press Ctrl+C to exit.")
try:
    while True:
        for joint_idx in range(num_joints):
            target_pos = p.readUserDebugParameter(slider_ids[joint_idx])
            p.setJointMotorControl2(bodyIndex=robot_id,
                                    jointIndex=joint_idx,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_pos,
                                    force=5)
        p.stepSimulation()
        time.sleep(1. / 240.)
except KeyboardInterrupt:
    print("Simulation stopped.")

# Disconnect
p.disconnect()
