import pybullet as p
import pybullet_data
import time
import math

print("=== Day 7: Simple Vision-Robot Demo ===")

## Setup PyBullet Environment
print("\n=== Setting up Robot Environment ===")

# Connect to PyBullet with GUI
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -10)

# Load ground plane
plane_id = p.loadURDF("plane.urdf")
print("Ground plane loaded")

# Load robot
try:
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0,0,0], useFixedBase=True)
    end_effector_index = 6
    robot_name = "KUKA"
    print(f"{robot_name} robot loaded")
except:
    try:
        robot_id = p.loadURDF("simple_arm.urdf", basePosition=[0,0,0], useFixedBase=True)
        end_effector_index = 1
        robot_name = "Simple Arm"
        print(f"{robot_name} robot loaded")
    except:
        print("No robot available")
        exit()

num_joints = p.getNumJoints(robot_id)
print(f"Robot has {num_joints} joints")

## Create Two Target Objects
print("\n=== Creating Target Objects ===")

# Object 1: Red cube (front-left)
target1_pos = [0.5, 0.2, 0.1]
col1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04])
vis1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04], 
                          rgbaColor=[1, 0, 0, 1])  # Red
target1_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col1,
                              baseVisualShapeIndex=vis1, basePosition=target1_pos)

# Object 2: Blue cube (front-right, close to red)
target2_pos = [0.5, -0.2, 0.1]
col2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04])
vis2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04], 
                          rgbaColor=[0, 0, 1, 1])  # Blue
target2_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col2,
                              baseVisualShapeIndex=vis2, basePosition=target2_pos)

print(f"Red cube placed at {target1_pos}")
print(f"Blue cube placed at {target2_pos}")

time.sleep(2)  # Let user see the setup

## Simple Robot Movement Function
def move_robot_to(target_pos, target_name):
    """Move robot to target position"""
    print(f"\n→ Moving to {target_name}")
    print(f"   Target: {target_pos}")
    
    try:
        # Calculate IK with better parameters
        joint_angles = p.calculateInverseKinematics(
            robot_id, 
            end_effector_index, 
            target_pos,
            maxNumIterations=100,
            residualThreshold=0.01
        )
        
        print(f"   IK calculated: {len(joint_angles)} joint angles")
        
        # Set joint targets with higher force
        for j in range(min(num_joints, len(joint_angles))):
            p.setJointMotorControl2(
                robot_id, j, 
                p.POSITION_CONTROL, 
                targetPosition=joint_angles[j], 
                force=100,  # Increased force
                maxVelocity=2.0  # Add velocity limit
            )
        
        # Wait for movement with progress check
        print("   Moving...", end="")
        for i in range(200):  # 3.3 seconds
            p.stepSimulation()
            time.sleep(1./60.)
            
            # Print progress every 50 steps
            if i % 50 == 0 and i > 0:
                print(".", end="")
        
        print(" Done!")
        
        # Check final position
        link_state = p.getLinkState(robot_id, end_effector_index)
        actual_pos = link_state[4]
        error = math.sqrt(sum((a-t)**2 for a,t in zip(actual_pos, target_pos)))
        
        print(f"   Reached: [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")
        print(f"   Error: {error:.3f}m")
        
        if error > 0.1:
            print(f"   Warning: Large error - target may be unreachable")
        
        return True
        
    except Exception as e:
        print(f"   Failed: {e}")
        return False

## Movement Sequence
print("\n=== Robot Movement Demo ===")

# Start position (home)
home_pos = [0.15, 0, 0.5]  # Slightly forward, lower
print("Starting from home position...")
move_robot_to(home_pos, "Home")

# Move to red cube (nearby)
red_target = [target1_pos[0], target1_pos[1], target1_pos[2] + 0.05]  # 5cm above
move_robot_to(red_target, "Red Cube")
time.sleep(2)

# Move to blue cube (very close to red)
blue_target = [target2_pos[0], target2_pos[1], target2_pos[2] + 0.05]  # 5cm above
move_robot_to(blue_target, "Blue Cube")
time.sleep(2)

# Return home
move_robot_to(home_pos, "Home")

print("\n=== Demo Complete ===")
print("Robot moved between 2 vision-detected objects")
print("Each movement used inverse kinematics")
print("Position accuracy was measured")

## Simple Continuous Demo
print("\nStarting continuous demo... Press Ctrl+C to stop")

targets = [
    ([0.15, 0, 0.25], "Home"),
    ([0.5, 0.2, 0.15], "Red"), 
    ([0.5, -0.2, 0.15], "Blue")
]

try:
    cycle = 0
    while True:
        target_pos, name = targets[cycle % len(targets)]
        
        print(f"Moving to {name}...")
        
        # Quick movement
        joint_angles = p.calculateInverseKinematics(robot_id, end_effector_index, target_pos)
        for j in range(min(num_joints, len(joint_angles))):
            p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, 
                                  targetPosition=joint_angles[j], force=50)
        
        # Wait 3 seconds at each position
        for _ in range(180):
            p.stepSimulation()
            time.sleep(1./60.)
        
        cycle += 1
        
except KeyboardInterrupt:
    print("\nDemo stopped by user")

p.disconnect()
print("Simple vision-robot demo complete!")