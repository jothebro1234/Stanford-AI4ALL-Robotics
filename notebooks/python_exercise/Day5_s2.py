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
print("Robot ready with", num_joints, "joints.")

## Exercise 1: Manual Joint Control
print("\n=== Exercise 1: Manual Joint Control ===")

# Reset and set joint angle
p.resetJointState(robot_id, 0, 0)
p.resetJointState(robot_id, 1, 0)

target_angle = 0.785  # 45 degrees
print(f"Setting Joint1 to {math.degrees(target_angle):.0f}°")

p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=target_angle, force=50)

for _ in range(120):
    p.stepSimulation()
    time.sleep(1./240.)

angle = p.getJointState(robot_id, 0)[0]
end_pos = p.getLinkState(robot_id, 1, computeForwardKinematics=True)[4]
print(f"Result: {math.degrees(angle):.0f}°, End-effector: {end_pos[0]:.2f}, {end_pos[1]:.2f}, {end_pos[2]:.2f}")

time.sleep(2)

## Exercise 2: Wave Motion
print("\n=== Exercise 2: Wave Motion ===")

# Wave parameters
amplitude = 0.8
offset = 0.8
num_waves = 2
steps_per_cycle = 60

print(f"Waving {num_waves} times...")

for i in range(num_waves * steps_per_cycle):
    t = i / steps_per_cycle * 2 * math.pi
    target = offset + amplitude * math.sin(t)
    
    p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=target, force=50)
    p.stepSimulation()
    time.sleep(1./240.)

print("Wave complete!")
time.sleep(2)

## Exercise 3: Pose Sequence
print("\n=== Exercise 3: Pose Sequence ===")

poses = [
    ("Up", [0.5, 0.0]),
    ("Bent", [1.0, 0.5]),
    ("Down", [0.0, -0.5]),
    ("Home", [0.0, 0.0])
]

for name, angles in poses:
    print(f"Moving to: {name}")
    p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=angles[0], force=50)
    p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=angles[1], force=50)
    
    for _ in range(120):
        p.stepSimulation()
        time.sleep(1./240.)
    
    # Check position
    final_angles = [p.getJointState(robot_id, j)[0] for j in range(num_joints)]
    print(f"  Reached: Joint1={math.degrees(final_angles[0]):.0f}°, Joint2={math.degrees(final_angles[1]):.0f}°")
    time.sleep(1)

## Exercise 4: IK Reaching
print("\n=== Exercise 4: IK Reaching ===")

targets = [
    ("Table", [0.6, 0.2, 0.3]),
    ("Shelf", [0.4, 0, 1.2]),
    ("Floor", [1.0, 0, 0.1])
]

# Create target markers
markers = []
for i, (name, pos) in enumerate(targets):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, 
                             rgbaColor=[1, i*0.5, 0, 0.7])
    marker = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                              baseVisualShapeIndex=vis, basePosition=pos)
    markers.append(marker)
    print(f"Target: {name} at {pos}")

# Reach each target
for i, (name, pos) in enumerate(targets):
    print(f"\nReaching: {name}")
    
    joint_angles = p.calculateInverseKinematics(robot_id, 1, pos)
    p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=joint_angles[0], force=50)
    p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=joint_angles[1], force=50)
    
    for _ in range(120):
        p.stepSimulation()
        time.sleep(1./240.)
    
    actual_pos = p.getLinkState(robot_id, 1, computeForwardKinematics=True)[4]
    error = math.sqrt(sum((a-t)**2 for a,t in zip(actual_pos, pos)))
    print(f"  Error: {error:.3f}m", "✓" if error < 0.1 else "⚠")
    time.sleep(1)

# Return home
p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=0, force=50)
p.setJointMotorControl2(robot_id, 1, p.POSITION_CONTROL, targetPosition=0, force=50)
for _ in range(60):
    p.stepSimulation()
    time.sleep(1./240.)

print("\n All exercises complete!")
print("You learned: Manual control, waves, sequences, IK, and creative motion!")

# Keep running for experiments
print("\nExperiment time! Press Ctrl+C to stop.")
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    p.disconnect()
    print("Great work! ")