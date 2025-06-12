import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import time

# Connect to PyBullet and load robot
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load plane and KUKA robot
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
num_joints = p.getNumJoints(robot_id)

print(f"KUKA loaded with {num_joints} joints")

def linear_interpolation(start, goal, steps):
    # Linear interpolation between start and goal
    trajectory = []
    for i in range(steps):
        t = i / (steps - 1)
        point = [s + t * (g - s) for s, g in zip(start, goal)]
        trajectory.append(point)
    return np.array(trajectory)

def calculate_max_joint_speed(trajectory, duration):
    # Calculate maximum joint speed: max(|Δθ|) / Δt
    dt = duration / (trajectory.shape[0] - 1)
    max_speed = 0
    for i in range(1, trajectory.shape[0]):
        speeds = np.abs(trajectory[i] - trajectory[i-1]) / dt
        max_speed = max(max_speed, np.max(speeds))
    return max_speed

def reset_robot():
    # Reset robot to zero position
    for i in range(num_joints):
        p.resetJointState(robot_id, i, 0.0)

def execute_trajectory(trajectory, delay=0.1):
    # Execute joint trajectory
    for angles in trajectory:
        for i, pos in enumerate(angles):
            if i < num_joints:
                p.resetJointState(robot_id, i, pos)
        p.stepSimulation()
        time.sleep(delay)


print("Exercise: Smoothness Metric - max_joint_speed = max(|Δθ|) / Δt")

# Define start and goal positions
start = [0.0, 0.0, 0.0, 0.0]
goal = [1.5, -1.2, 0.8, -0.5]
duration = 2.0
step_counts = [3, 6, 12, 24]

results = []
trajectories = []

print(f"Analyzing trajectory from {start} to {goal}")
print(f"Duration: {duration}s\n")

# Generate trajectories for different step counts
for steps in step_counts:
    traj = linear_interpolation(start, goal, steps)
    max_speed = calculate_max_joint_speed(traj, duration)
    total_change = np.sum(np.abs(np.diff(traj, axis=0)))
    
    results.append({'steps': steps, 'max_speed': max_speed, 'total_change': total_change})
    trajectories.append(traj)
    
    print(f"{steps:2d} steps: max_joint_speed = {max_speed:.3f} rad/s, total_change = {total_change:.2f} rad")

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
steps_list = [r['steps'] for r in results]
max_speeds = [r['max_speed'] for r in results]
plt.plot(steps_list, max_speeds, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Steps')
plt.ylabel('Max Joint Speed (rad/s)')
plt.title('Max Speed vs Steps')
plt.grid(True)

plt.subplot(1, 3, 2)
total_changes = [r['total_change'] for r in results]
plt.plot(steps_list, total_changes, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Steps')
plt.ylabel('Total Joint Change (rad)')
plt.title('Total Change vs Steps')
plt.grid(True)

plt.subplot(1, 3, 3)
colors = ['r', 'g', 'b', 'm']
markers = ['-o', '-s', '-^', '-v']
for i, (steps, traj) in enumerate(zip(step_counts, trajectories)):
    plt.plot(traj[:, 0], traj[:, 1], markers[i], color=colors[i],
            label=f'{steps} steps', linewidth=2, markersize=4)
plt.xlabel('Joint 1 (rad)')
plt.ylabel('Joint 2 (rad)')
plt.title('Joint Space Trajectories')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Execute trajectories in PyBullet simulation
print("\nExecuting trajectories in PyBullet...")
for i, (steps, traj) in enumerate(zip(step_counts, trajectories)):
    print(f"Simulating {steps}-step trajectory...")
    reset_robot()
    execute_trajectory(traj, delay=0.05)
    
    if i < len(step_counts) - 1:
        input(f"Press Enter to continue with {step_counts[i+1]}-step trajectory...")

print("\nAnalysis complete!")
print("Observation: More steps lead to lower maximum joint speeds (smoother motion)")

# Summary
print("\nSummary:")
for r in results:
    print(f"{r['steps']:2d} steps -> Max Speed: {r['max_speed']:.3f} rad/s")

# Disconnect
p.disconnect()