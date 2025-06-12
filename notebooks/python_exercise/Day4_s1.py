import pybullet as p
import time

# Start simulation window (Use Pybullet GUI)
physicsClient = p.connect(p.GUI)

urdf_path = "simple_arm.urdf"
robot_id = p.loadURDF(urdf_path, basePosition=[0,0,0], useFixedBase=True)
print("Loaded simple robot with ID:", robot_id)


time.sleep(10)

p.disconnect()