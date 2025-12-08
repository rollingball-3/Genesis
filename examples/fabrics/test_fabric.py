"""
Run multiple FABRICS tests with different configurations.

This script runs 5 different tests:
1. No palm target, no finger forces - maintain initial pose
2. Palm position changes from initial position
3. Palm position and orientation both change
4. Finger forces applied, palm unchanged
5. Both palm target and finger forces applied
"""
import os
import time
import numpy as np
import torch
from pathlib import Path

from genesis.utils.geom import quat_to_xyz, xyz_to_quat, quat_to_R

# Import utilities
from test_fabric_utils import (
    FabricTestConfig,
    load_initial_joint_config,
    setup_genesis_simulation,
    setup_fabrics_controller,
    to_fabrics,
    to_genesis,
    world_to_base_frame,
    update_palm_target_marker,
    update_finger_force_markers,
    plot_control_data,
)
from fabrics.throw_env import Manipulator


class TestConfig(FabricTestConfig):
    """Extended config for batch testing"""
    total_time = 10.0  # seconds per test
    
    # Test 1: No motion
    test1_name = "test1_maintain_initial_pose"
    
    # Test 2: Palm position change
    test2_name = "test2_palm_position_change"
    test2_palm_motion_radius = 0.05  # meters
    test2_palm_motion_height = 0.02  # meters
    test2_palm_motion_frequency = 0.15  # Hz
    
    # Test 3: Palm position and orientation change
    test3_name = "test3_palm_pose_change"
    test3_palm_motion_radius = 0.05
    test3_palm_motion_height = 0.02
    test3_palm_motion_frequency = 0.15
    test3_orientation_amplitude = 0.2  # radians
    test3_orientation_frequency = 0.1  # Hz
    
    # Test 4: Finger forces only
    test4_name = "test4_finger_forces_only"
    test4_force_period = 2.0  # seconds
    test4_force_hold_time = 1.0  # seconds
    test4_force_max = 1.0 # N
    
    # Test 5: Both palm and forces
    test5_name = "test5_palm_and_forces"
    test5_palm_motion_radius = 0.05
    test5_palm_motion_height = 0.02
    test5_palm_motion_frequency = 0.15
    test5_force_period = 2.0
    test5_force_hold_time = 1.0
    test5_force_max = 5.0


def quat_mult_torch(q1, q2):
    """Multiply two quaternions: q_result = q1 * q2"""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=1)


def run_single_test(
    test_name,
    cfg,
    scene,
    robot:Manipulator,
    camera,
    palm_target_marker,
    finger_force_markers,
    cs63_fabric,
    cs63_integrator,
    object_ids,
    object_indicator,
    output_dir,
    palm_motion_fn=None,
    force_fn=None,
):
    """
    Run a single test with specified motion and force functions.
    
    Args:
        palm_motion_fn: Function(time, palm_target_center, palm_quat_base) -> (palm_pos, palm_quat)
        force_fn: Function(time) -> finger_forces [B, 3, 3]
    """
    print(f"\n{'='*60}")
    print(f"Running Test: {test_name}")
    print(f"{'='*60}")
    
    # Reset simulation
    robot.reset()
    
    # Load initial configuration
    initial_config = load_initial_joint_config()
    q_initial = torch.tensor(initial_config, device=cfg.genesis_device).unsqueeze(0)
    
    # Step simulation once to update palm pose
    scene.step()
    
    # Now read the actual palm pose at this configuration
    current_palm_pose_world = robot.palm_pose  # [B, 7] in world frame
    current_palm_pose_base = world_to_base_frame(current_palm_pose_world, cfg.base_height)
    
    # Initialize FABRICS state with initial configuration
    fabric_q = to_fabrics(q_initial.clone(), cfg)
    fabric_qd = torch.zeros(cfg.num_envs, 18, device=cfg.fabrics_device)
    fabric_qdd = torch.zeros(cfg.num_envs, 18, device=cfg.fabrics_device)
    
    # Set initial position and orientation from actual palm pose (Genesis base frame)
    initial_pos_base = current_palm_pose_base[:, :3]                # on Genesis device
    initial_quat = current_palm_pose_base[:, 3:7]                   # on Genesis device

    # Palm target (position + Euler) for visualization and logging (FABRICS device)
    initial_euler = quat_to_xyz(initial_quat, rpy=True, degrees=False)
    palm_target = torch.zeros(cfg.num_envs, 6, device=cfg.fabrics_device)
    palm_target[:, 0:3] = to_fabrics(initial_pos_base, cfg)
    palm_target[:, 3:6] = to_fabrics(initial_euler, cfg)

    # Palm pose for FABRICS (position + rotation matrix on FABRICS device)
    palm_position = palm_target[:, 0:3].clone()
    initial_rotmat = quat_to_R(initial_quat)                        # still on Genesis device
    palm_matrix = to_fabrics(initial_rotmat, cfg)

    # Store base values for motion (FABRICS device for position, FABRICS device for quat)
    palm_target_base_center = palm_target[:, :3].clone()           # FABRICS device
    palm_target_base_quat = to_fabrics(initial_quat, cfg)          # FABRICS device
    
    # Initialize finger forces
    finger_forces = torch.zeros(cfg.num_envs, 3, 3, device=cfg.fabrics_device)
    
    # Start video recording
    video_path = os.path.join(output_dir, f"{test_name}.mp4")
    print(f"Recording video to: {video_path}")
    camera.start_recording()
    
    # Initialize data logging
    log_data = {
        'times': [],
        'q_actual': [],
        'q_desired': [],
        'qd_actual': [],
        'qd_desired': [],
        'palm_targets': [],
        'palm_targets_control_pos': [],
        'palm_targets_control_matrix': [],
        'palm_targets_actual': [],
    }
    
    num_control_steps = int(cfg.total_time / cfg.control_dt)
    
    for control_step in range(num_control_steps):
        current_time = control_step * cfg.control_dt
        
        # Update palm target
        if palm_motion_fn is not None:
            palm_pos, palm_quat = palm_motion_fn(
                current_time,
                palm_target_base_center,
                palm_target_base_quat,
                cfg
            )
            # Visualization / logging uses Euler representation
            palm_target[:, 0:3] = palm_pos
            palm_target[:, 3:6] = quat_to_xyz(palm_quat, rpy=True, degrees=False)

            # FABRICS palm pose uses position + rotation matrix
            palm_position = palm_pos
            palm_matrix = quat_to_R(palm_quat)
        
        # Update finger forces
        if force_fn is not None:
            finger_forces = force_fn(current_time, cfg)
        else:
            finger_forces.zero_()
        
        # Read state from Genesis
        q_genesis = torch.cat([robot.arm_qpos, robot.gripper_qpos], dim=1)
        qd_genesis = torch.cat([robot.arm_qvel, robot.gripper_qvel], dim=1)
        
        # Log control palm target actually sent to FABRICS (position + rotation matrix)
        log_data['palm_targets_control_pos'].append(
            palm_position[0].detach().cpu().numpy()
        )
        log_data['palm_targets_control_matrix'].append(
            palm_matrix[0].detach().cpu().numpy()
        )
        
        # Log actual palm pose from Genesis (base frame, position + quaternion wxyz)
        current_palm_pose_world_step = robot.palm_pose
        current_palm_pose_base_step = world_to_base_frame(
            current_palm_pose_world_step, cfg.base_height
        )
        log_data['palm_targets_actual'].append(
            current_palm_pose_base_step[0].detach().cpu().numpy()
        )
        
        # Set FABRICS features
        # print(finger_forces)
        cs63_fabric.set_features(
            finger_forces, palm_position, palm_matrix,
            fabric_q.detach(), fabric_qd.detach(),
            object_ids, object_indicator
        )
        
        # FABRICS integration
        for _ in range(cfg.fabric_decimation):
            fabric_q_new, fabric_qd_new, fabric_qdd_new = cs63_integrator.step(
                fabric_q.detach(), fabric_qd.detach(), fabric_qdd.detach(), cfg.fabrics_dt
            )
            fabric_q.copy_(fabric_q_new)
            fabric_qd.copy_(fabric_qd_new)
            fabric_qdd.copy_(fabric_qdd_new)
        
        # Transfer control commands to Genesis
        q_desired = to_genesis(fabric_q, cfg)
        qd_desired = to_genesis(fabric_qd, cfg)
        
        # Send PD commands
        q_cmd = q_desired
        qd_cmd = torch.zeros_like(q_desired)
        robot.command_arm_position_velocity(
            position=q_cmd[:, :6],
            velocity=qd_cmd[:, :6],
        )
        robot.command_gripper_position_velocity(
            position=q_cmd[:, 6:],
            velocity=qd_cmd[:, 6:],
        )
        
        # Step simulation
        scene.step()
        camera.render()
        
        # Update visualization markers
        if cfg.show_palm_target:
            update_palm_target_marker(palm_target_marker, palm_target, cfg)
        
        if cfg.show_finger_forces:
            update_finger_force_markers(
                finger_force_markers,
                robot.finger_force_frames_pose,
                finger_forces,
                cfg
            )
        
        # Log data
        log_data['times'].append(current_time)
        log_data['q_actual'].append(q_genesis[0].cpu().numpy())
        log_data['q_desired'].append(q_cmd[0].cpu().numpy())
        log_data['qd_actual'].append(qd_genesis[0].cpu().numpy())
        log_data['qd_desired'].append(qd_cmd[0].cpu().numpy())
        log_data['palm_targets'].append(palm_target[0].cpu().numpy())
    
    # Stop recording
    camera.stop_recording(save_to_filename=video_path)
    print(f"✅ Video saved: {video_path}")
    
    # Convert lists to arrays
    for key in [
        'times',
        'q_actual',
        'q_desired',
        'qd_actual',
        'qd_desired',
        'palm_targets',
        'palm_targets_control_pos',
        'palm_targets_control_matrix',
        'palm_targets_actual',
    ]:
        log_data[key] = np.array(log_data[key])
    
    # Plot data
    if cfg.plot_control_data:
        plot_path = os.path.join(output_dir, f"{test_name}_summary.png")
        plot_control_data(log_data, plot_path)
    
    print(f"✅ Test completed: {test_name}\n")


# ============ Test Motion Functions ============

def test1_motion(time, palm_center, palm_quat, cfg):
    """Test 1: No motion - maintain initial pose"""
    return palm_center, palm_quat


def test2_motion(time, palm_center, palm_quat, cfg):
    """Test 2: Palm position changes"""
    t_tensor = torch.tensor(time, device=cfg.fabrics_device, dtype=palm_center.dtype)
    angle = t_tensor * (2 * np.pi * cfg.test2_palm_motion_frequency)
    
    dx = cfg.test2_palm_motion_radius * torch.sin(angle)
    dy = cfg.test2_palm_motion_radius * (torch.cos(angle) - 1.0)
    dz = cfg.test2_palm_motion_height * torch.sin(2 * angle)
    
    palm_pos = palm_center.clone()
    palm_pos[:, 0] += dx
    palm_pos[:, 1] += dy
    palm_pos[:, 2] += dz
    
    # Keep orientation unchanged: return original quaternion
    return palm_pos, palm_quat


def test3_motion(time, palm_center, palm_quat, cfg):
    """Test 3: Palm position and orientation change (using rotation matrix)"""
    # Position change
    t_tensor = torch.tensor(time, device=cfg.fabrics_device, dtype=palm_center.dtype)
    angle = t_tensor * (2 * np.pi * cfg.test3_palm_motion_frequency)
    
    dx = cfg.test3_palm_motion_radius * torch.sin(angle)
    dy = cfg.test3_palm_motion_radius * (torch.cos(angle) - 1.0)
    dz = cfg.test3_palm_motion_height * torch.sin(2 * angle)
    
    palm_pos = palm_center.clone()
    palm_pos[:, 0] += dx
    palm_pos[:, 1] += dy
    palm_pos[:, 2] += dz
    
    # Orientation change: use rotation matrix multiplication
    # Convert quaternion to rotation matrix
    R_base = quat_to_R(palm_quat)  # (b, 3, 3)
    
    # Create small rotation matrices for each axis
    t_dev = torch.tensor(time, device=R_base.device, dtype=R_base.dtype)
    angle_x = cfg.test3_orientation_amplitude * torch.sin(t_dev * (2 * np.pi * cfg.test3_orientation_frequency))
    angle_y = cfg.test3_orientation_amplitude * torch.sin(t_dev * (2 * np.pi * cfg.test3_orientation_frequency * 1.3))
    angle_z = cfg.test3_orientation_amplitude * torch.sin(t_dev * (2 * np.pi * cfg.test3_orientation_frequency * 0.7))
    
    # Rotation around X-axis
    cos_x, sin_x = torch.cos(angle_x), torch.sin(angle_x)
    R_x = torch.eye(3, device=R_base.device, dtype=R_base.dtype).unsqueeze(0).repeat(R_base.shape[0], 1, 1)
    R_x[:, 1, 1] = cos_x
    R_x[:, 1, 2] = -sin_x
    R_x[:, 2, 1] = sin_x
    R_x[:, 2, 2] = cos_x
    
    # Rotation around Y-axis
    cos_y, sin_y = torch.cos(angle_y), torch.sin(angle_y)
    R_y = torch.eye(3, device=R_base.device, dtype=R_base.dtype).unsqueeze(0).repeat(R_base.shape[0], 1, 1)
    R_y[:, 0, 0] = cos_y
    R_y[:, 0, 2] = sin_y
    R_y[:, 2, 0] = -sin_y
    R_y[:, 2, 2] = cos_y
    
    # Rotation around Z-axis
    cos_z, sin_z = torch.cos(angle_z), torch.sin(angle_z)
    R_z = torch.eye(3, device=R_base.device, dtype=R_base.dtype).unsqueeze(0).repeat(R_base.shape[0], 1, 1)
    R_z[:, 0, 0] = cos_z
    R_z[:, 0, 1] = -sin_z
    R_z[:, 1, 0] = sin_z
    R_z[:, 1, 1] = cos_z
    
    # Apply perturbations: R_new = R_z @ R_y @ R_x @ R_base
    R_new = torch.bmm(R_z, torch.bmm(R_y, torch.bmm(R_x, R_base)))
    
    # Convert back to quaternion
    from genesis.utils.geom import R_to_quat
    rotated_quat = R_to_quat(R_new)
    
    # Return new pose as position + quaternion
    return palm_pos, rotated_quat


def test4_forces(time, cfg):
    """Test 4: Finger forces only"""
    finger_forces = torch.zeros(cfg.num_envs, 3, 3, device=cfg.fabrics_device)
    
    total_period = cfg.test4_force_period + 2 * cfg.test4_force_hold_time
    phase = (time % total_period) / total_period
    
    ramp_fraction = cfg.test4_force_period / total_period / 2
    hold_fraction = cfg.test4_force_hold_time / total_period
    
    if phase < ramp_fraction:
        force_unit = phase / ramp_fraction
    elif phase < ramp_fraction + hold_fraction:
        force_unit = 1.0
    elif phase < ramp_fraction * 2 + hold_fraction:
        force_unit = 1.0 - 2 * (phase - ramp_fraction - hold_fraction) / ramp_fraction
    elif phase < ramp_fraction * 2 + hold_fraction * 2:
        force_unit = -1.0
    else:
        force_unit = -1.0 + (phase - ramp_fraction * 2 - hold_fraction * 2) / ramp_fraction
    
    force_value = force_unit * cfg.test4_force_max
    force_tensor = torch.tensor(force_value, device=cfg.fabrics_device, dtype=finger_forces.dtype)
    finger_forces[:, :, 2] = force_tensor
    # finger_forces[:, 1, 2] = force_tensor
    # finger_forces[:, 2, 0] = force_tensor
    
    return finger_forces


def test5_forces(time, cfg):
    """Test 5: Same forces as test 4"""
    finger_forces = torch.zeros(cfg.num_envs, 3, 3, device=cfg.fabrics_device)
    
    total_period = cfg.test5_force_period + 2 * cfg.test5_force_hold_time
    phase = (time % total_period) / total_period
    
    ramp_fraction = cfg.test5_force_period / total_period / 2
    hold_fraction = cfg.test5_force_hold_time / total_period
    
    if phase < ramp_fraction:
        force_unit = phase / ramp_fraction
    elif phase < ramp_fraction + hold_fraction:
        force_unit = 1.0
    elif phase < ramp_fraction * 2 + hold_fraction:
        force_unit = 1.0 - 2 * (phase - ramp_fraction - hold_fraction) / ramp_fraction
    elif phase < ramp_fraction * 2 + hold_fraction * 2:
        force_unit = -1.0
    else:
        force_unit = -1.0 + (phase - ramp_fraction * 2 - hold_fraction * 2) / ramp_fraction
    
    force_value = force_unit * cfg.test5_force_max
    force_tensor = torch.tensor(force_value, device=cfg.fabrics_device, dtype=finger_forces.dtype)
    finger_forces[:, :, 2] = force_tensor
    # finger_forces[:, 1, 2] = force_tensor
    # finger_forces[:, 2, 0] = force_tensor
    
    return finger_forces


def main():
    cfg = TestConfig()
    
    # Create timestamped output directory
    timestamp = int(time.time())
    output_dir = os.path.join(cfg.base_dir, f"batch_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print("\n" + "=" * 60)
    print("Genesis + FABRICS - Batch Test Suite")
    print(f"  Genesis (Taichi): {cfg.genesis_device}")
    print(f"  FABRICS (Warp):   {cfg.fabrics_device}")
    print("=" * 60 + "\n")
    
    # Setup simulation (only once)
    scene, robot, camera, palm_target_marker, finger_force_markers = setup_genesis_simulation(cfg)
    
    # Setup FABRICS controller (only once)
    cs63_fabric, cs63_integrator, object_ids, object_indicator = setup_fabrics_controller(cfg)
    
    # Run all tests
    tests = [
        (cfg.test1_name, None, None),  # Test 1: No motion
        (cfg.test2_name, test2_motion, None),  # Test 2: Palm position
        (cfg.test3_name, test3_motion, None),  # Test 3: Palm pose
        (cfg.test4_name, test1_motion, test4_forces),  # Test 4: Forces only
        (cfg.test5_name, test2_motion, test5_forces),  # Test 5: Both
    ]
    
    for test_name, palm_fn, force_fn in tests:
        run_single_test(
            test_name=test_name,
            cfg=cfg,
            scene=scene,
            robot=robot,
            camera=camera,
            palm_target_marker=palm_target_marker,
            finger_force_markers=finger_force_markers,
            cs63_fabric=cs63_fabric,
            cs63_integrator=cs63_integrator,
            object_ids=object_ids,
            object_indicator=object_indicator,
            output_dir=output_dir,
            palm_motion_fn=palm_fn,
            force_fn=force_fn,
        )
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
