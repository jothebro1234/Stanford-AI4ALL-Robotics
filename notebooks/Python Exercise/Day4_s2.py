import pybullet as p
import pybullet_data
import time

# Start simulation window (Use Pybullet GUI)
physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()  # start with a clean simulation
p.setGravity(0, 0, -10)  # set gravity to earth-like

# Load ground and robot
plane_id = p.loadURDF("plane.urdf")

# We assume the simple_arm.urdf is available (from previous session).
robot_id = p.loadURDF("simple_arm.urdf", basePosition=[0,0,0], useFixedBase=True)
print("Environment ready. Plane ID:", plane_id, " Robot ID:", robot_id)


## Change Robot Base Position
# Remove the old robot first (to avoid cluttering multiple robots)
p.removeBody(robot_id)

# Define a new base position and orientation
new_base_pos = [0.5, 0, 0]  # move the robot 0.5 meters to the right (along x-axis)
# Define a yaw rotation (turn around Z axis) of 90 degrees for the base
import math
yaw_angle = math.radians(90)  # 90 degrees in radians
# Get quaternion from Euler [roll, pitch, yaw] = [0, 0, yaw_angle]
base_orientation = p.getQuaternionFromEuler([0, 0, yaw_angle])

# Load the robot at the new position and orientation
robot_id = p.loadURDF("simple_arm.urdf", basePosition=new_base_pos, baseOrientation=base_orientation, useFixedBase=True)
print("Robot loaded at new base position/orientation. ID:", robot_id)
# Check the base pose
print("New base pose:", p.getBasePositionAndOrientation(robot_id))

## Add Object
cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
cube_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=[1, 0, 0, 1])
cube_start_pos = [0.5, 0, 0.5]
cube_mass = 1.0
cube_id = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_col,
                             baseVisualShapeIndex=cube_vis, basePosition=cube_start_pos)
print("Created cube with ID:", cube_id, "at position", cube_start_pos)


# Start Simulation
print("Starting live simulation: Press Ctrl+C to stop.")
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)  # Frame Rate
except KeyboardInterrupt:
    print("Simulation closed.")

# Finish pybullet
p.disconnect()