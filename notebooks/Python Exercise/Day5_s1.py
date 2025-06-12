import pybullet as p
import pybullet_data
import time
import math

# Start simulation
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -10)

# Load environment and robot
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("simple_arm.urdf", basePosition=[0,0,0], useFixedBase=True)
num_joints = p.getNumJoints(robot_id)
print("Robot loaded with", num_joints, "joints.")

## Part 1: Basic Joint Control
print("\n=== Part 1: Joint Control ===")

# Set target angles
target_angle_joint1 = 1.0  # ~57 degrees
target_angle_joint2 = -0.5  # ~-29 degrees

print(f"Moving to: Joint1={math.degrees(target_angle_joint1):.0f}°, Joint2={math.degrees(target_angle_joint2):.0f}°")

# Apply joint commands
p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, 
                       targetPosition=target_angle_joint1, force=50)
p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, 
                       targetPosition=target_angle_joint2, force=50)

# Let robot move
for _ in range(200):
    p.stepSimulation()
    time.sleep(1./240.)

# Check result
angle1 = p.getJointState(robot_id, 0)[0]
angle2 = p.getJointState(robot_id, 1)[0]
end_pos = p.getLinkState(robot_id, 1, computeForwardKinematics=True)[4]

print(f"Final: Joint1={math.degrees(angle1):.0f}°, Joint2={math.degrees(angle2):.0f}°")
print(f"End-effector: x={end_pos[0]:.2f}, y={end_pos[1]:.2f}, z={end_pos[2]:.2f}")

time.sleep(2)

## Part 2: Inverse Kinematics
print("\n=== Part 2: Inverse Kinematics ===")

# Reset to zero
p.resetJointState(robot_id, 0, 0)
p.resetJointState(robot_id, 1, 0)

# Define target position
target_pos = [0.5, 0, 0.5]
print(f"Target position: {target_pos}")

# Create target marker
marker_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
marker_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 1, 0, 0.8])
marker_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=marker_col,
                             baseVisualShapeIndex=marker_vis, basePosition=target_pos)

# Calculate and apply IK
joint_angles = p.calculateInverseKinematics(robot_id, 1, target_pos)
print(f"IK solution: Joint1={math.degrees(joint_angles[0]):.0f}°, Joint2={math.degrees(joint_angles[1]):.0f}°")

p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, 
                       targetPosition=joint_angles[0], force=50)
p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, 
                       targetPosition=joint_angles[1], force=50)

for _ in range(200):
    p.stepSimulation()
    time.sleep(1./240.)

# Check accuracy
final_pos = p.getLinkState(robot_id, 1, computeForwardKinematics=True)[4]
error = math.sqrt(sum((a-t)**2 for a,t in zip(final_pos, target_pos)))
print(f"Reached: x={final_pos[0]:.3f}, y={final_pos[1]:.3f}, z={final_pos[2]:.3f}")
print(f"Error: {error:.3f}m", "✓" if error < 0.05 else "⚠")

time.sleep(2)

## Part 3: Test Multiple Targets
print("\n=== Part 3: Testing Different Targets ===")

targets = [
    ("Close", [0.8, 0, 0.8]),
    ("High", [0.3, 0, 1.3]),
    ("Far", [1.8, 0, 0.3])
]

for name, pos in targets:
    print(f"\nTrying {name}: {pos}")
    
    p.removeBody(marker_id)
    marker_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=marker_col,
                                 baseVisualShapeIndex=marker_vis, basePosition=pos)
    
    ik_solution = p.calculateInverseKinematics(robot_id, 1, pos)
    p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=ik_solution[0], force=50)
    p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=ik_solution[1], force=50)
    
    for _ in range(150):
        p.stepSimulation()
        time.sleep(1./240.)
    
    result_pos = p.getLinkState(robot_id, 1, computeForwardKinematics=True)[4]
    error = math.sqrt(sum((a-t)**2 for a,t in zip(result_pos, pos)))
    print(f"Error: {error:.3f}m", "✓" if error < 0.1 else "⚠")

print("\n=== Session 1 Complete ===")
print("Learned: Joint control, IK basics, workspace limits")

# Keep running
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    p.disconnect()
    print("Done!")