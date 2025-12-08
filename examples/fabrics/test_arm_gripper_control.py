"""
Test script for CS63 arm + DG3F gripper control.

Demonstrates 4 control modes:
1. Hold initial joint configuration from YAML
2. Move to arbitrary joint positions
3. End-effector control using apply_action (IK-based delta pose)
4. End-effector control using go_to_goal (IK-based absolute pose)

Each control mode saves a video for visualization.
"""
import os
import yaml
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

import genesis as gs
from throw_env import Manipulator

# Get script directory
SCRIPT_DIR = Path(__file__).parent.resolve()


def load_joint_config(yaml_path: str | Path) -> tuple[list, list]:
    """Load joint configuration from YAML file."""
    yaml_path = Path(yaml_path)
    if not yaml_path.is_absolute():
        yaml_path = SCRIPT_DIR / yaml_path
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract arm joints in order
    arm_joints = [
        config['arm_joints']['shoulder_pan_joint'],
        config['arm_joints']['shoulder_lift_joint'],
        config['arm_joints']['elbow_joint'],
        config['arm_joints']['wrist_1_joint'],
        config['arm_joints']['wrist_2_joint'],
        config['arm_joints']['wrist_3_joint'],
    ]
    
    # Extract gripper joints in order
    gripper_joints = [
        config['gripper_joints']['F1M1'],
        config['gripper_joints']['F1M2'],
        config['gripper_joints']['F1M3'],
        config['gripper_joints']['F1M4'],
        config['gripper_joints']['F2M1'],
        config['gripper_joints']['F2M2'],
        config['gripper_joints']['F2M3'],
        config['gripper_joints']['F2M4'],
        config['gripper_joints']['F3M1'],
        config['gripper_joints']['F3M2'],
        config['gripper_joints']['F3M3'],
        config['gripper_joints']['F3M4'],
    ]
    
    return arm_joints, gripper_joints


def test_1_initial_config(scene, manipulator, camera, output_dir, device):
    """Test 1: Hold initial joint configuration from YAML using PD (v_des = 0)."""
    print("\n" + "=" * 60)
    print("Test 1: Hold Initial Joint Configuration")
    print("=" * 60)
    
    # Load config from YAML
    arm_joints, gripper_joints = load_joint_config("initial_joint_config.yaml")
    
    print(f"Target arm joints (deg): {np.degrees(arm_joints)}")
    print(f"Target gripper joints (deg): {np.degrees(gripper_joints)}")
    
    # Convert to tensors
    arm_target = torch.tensor(arm_joints, device=device)
    gripper_target = torch.tensor(gripper_joints, device=device)
    
    # Prepare PD targets: desired position with zero velocity for stability
    arm_pos = arm_target.unsqueeze(0)  # [1, 6]
    grip_pos = gripper_target.unsqueeze(0)  # [1, 12]
    arm_vel = torch.zeros_like(arm_pos, device=device)
    grip_vel = torch.zeros_like(grip_pos, device=device)
    
    # Start recording
    video_path = os.path.join(output_dir, "01_initial_config.mp4")
    camera.start_recording()
    
    # Run simulation with PD control (position + zero velocity)
    total_steps = 300
    print(f"Running {total_steps} steps...")
    for step in range(total_steps):
        # PD control at each step: position + zero velocity
        manipulator.command_arm_position_velocity(position=arm_pos, velocity=arm_vel)
        manipulator.command_gripper_position_velocity(position=grip_pos, velocity=grip_vel)
        scene.step()
        camera.render()
        
        if step % 100 == 0:
            arm_qpos = manipulator.arm_qpos[0]
            gripper_qpos = manipulator.gripper_qpos[0]
            arm_error = torch.norm(arm_qpos - arm_target).item()
            gripper_error = torch.norm(gripper_qpos - gripper_target).item()
            print(f"Step {step}: arm_error={arm_error:.4f}, gripper_error={gripper_error:.4f}")
    
    camera.stop_recording(save_to_filename=video_path)
    print(f"✅ Saved video: {video_path}")


def main():
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"tmp/arm_gripper_control_test/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create scene
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -1.5, 1.8),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=40,
        ),
        sim_options=gs.options.SimOptions(dt=0.01, substeps=10),
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
            enable_self_collision=True,
            enable_adjacent_collision=False,
        ),
        show_viewer=False,
    )
    
    # Add ground
    scene.add_entity(gs.morphs.Plane())
    
    # Add manipulator (base elevated to 0.6m to avoid ground collision)
    device = "cuda:0"
    manipulator = Manipulator(scene=scene, num_envs=1, device=device, base_height=0.6)
    
    # Add camera
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(1.5, -1.5, 1.8),
        lookat=(0.0, 0.0, 1.0),
        fov=40,
        GUI=False,
    )
    
    # Build scene
    scene.build(n_envs=1)
    
    # Set PD gains
    manipulator.set_pd_gains()
    
    print("\n" + "=" * 60)
    print("CS63 Arm + DG3F Gripper Control Tests")
    print("=" * 60)
    
    # Run tests
    test_1_initial_config(scene, manipulator, camera, output_dir, device)
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print(f"📁 All videos saved to: {output_dir}")


if __name__ == "__main__":
    main()

