"""
Utility functions for Genesis + FABRICS integration tests.

This module contains common functions for:
- Genesis simulation setup
- FABRICS controller initialization
- Coordinate transformations
- Visualization
- Data logging and plotting
"""
import os
import sys
import yaml
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Genesis imports
import genesis as gs
from genesis.utils.geom import quat_to_xyz, xyz_to_quat, transform_by_quat

# Import Manipulator class
sys.path.append(str(Path(__file__).parent.parent))
from fabrics.throw_env import Manipulator

# FABRICS imports
sys.path.append(str(Path(__file__).parent.parent.parent / "genesis/ext/FABRICS/src"))
from fabrics_sim.fabrics.cs63_tesollo_fabric import CS63TesolloFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel


# ============ Configuration ============
class FabricTestConfig:
    """Simulation and control configuration"""
    # Genesis simulation
    num_envs = 1
    sim_dt = 1.0 / 60.0  # 60 Hz
    genesis_device = "cuda:0"
    
    # FABRICS
    fabrics_device = "cuda:1"
    cuda_graph = False
    fabric_decimation = 2
    fabrics_dt = sim_dt / fabric_decimation
    
    # Control timing
    control_dt = 1.0 / 60.0  # 60 Hz
    
    # Recording
    save_video = True
    base_dir = "tmp/fabrics_test"
    
    # Data logging
    save_control_data = True
    plot_control_data = True
    
    # Visualization
    show_palm_target = True
    show_finger_forces = False
    force_scale = 0.1
    base_height = 0.6


# ============ Initial Config Loading ============
def load_initial_joint_config():
    """Load initial joint configuration from YAML file"""
    config_path = Path(__file__).parent / "initial_joint_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse arm joints
    arm_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    arm_positions = [config["arm_joints"][name] for name in arm_joint_names]
    
    # Parse gripper joints
    gripper_joint_names = [
        "F1M1", "F1M2", "F1M3", "F1M4",
        "F2M1", "F2M2", "F2M3", "F2M4",
        "F3M1", "F3M2", "F3M3", "F3M4"
    ]
    gripper_positions = [config["gripper_joints"][name] for name in gripper_joint_names]
    
    return arm_positions + gripper_positions


# ============ Genesis Simulation Setup ============
def setup_genesis_simulation(cfg: FabricTestConfig):
    """Initialize Genesis simulation with CS63 + DG3F"""
    print(f"Setting up Genesis simulation on {cfg.genesis_device}...")
    gs.init(backend=gs.gpu)
    
    # Create scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg.sim_dt, substeps=2),
        rigid_options=gs.options.RigidOptions(
            dt=cfg.sim_dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,  # Disabled to avoid coordinate frame confusion
            world_frame_size=0.5,
            show_link_frame=False,   # Don't show link frames
        ),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=60,
            camera_pos=(2.0, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=False,
    )
    
    # Add ground
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    
    # Create Manipulator
    robot = Manipulator(
        scene=scene,
        num_envs=cfg.num_envs,
        device=cfg.genesis_device,
        base_height=cfg.base_height,
    )
    
    # Add camera
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(2.0, 0.0, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=40,
        GUI=False,
    )
    
    # Add visualization markers
    palm_target_marker = None
    if cfg.show_palm_target:
        axis_length = 0.05
        axis_radius = 0.003
        sphere_radius = 0.01
        
        # Z-axis (blue) - show palm orientation
        axis_z = scene.add_entity(
            gs.morphs.Cylinder(
                radius=axis_radius,
                height=axis_length,
                pos=(0.0, 0.0, 0.5),
                collision=False,
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.0, 0.0, 1.0, 1.0)),
            visualize_contact=False,
        )
        # Position sphere (white) - show palm target position
        palm_sphere = scene.add_entity(
            gs.morphs.Sphere(
                radius=sphere_radius,
                pos=(0.0, 0.0, 0.5),
                collision=False,
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0, 1.0)),
            visualize_contact=False,
        )
        palm_target_marker = [axis_z, palm_sphere]
    
    # Finger force arrows
    finger_force_markers = []
    if cfg.show_finger_forces:
        colors = [(0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0)]
        for i, color in enumerate(colors):
            arrow = scene.add_entity(
                gs.morphs.Cylinder(
                    radius=0.005,
                    height=0.05,
                    pos=(0.0, 0.0, 0.5),
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=color),
                visualize_contact=False,
            )
            finger_force_markers.append(arrow)
    
    # Build scene
    scene.build(n_envs=cfg.num_envs)
    robot.set_pd_gains()
    
    print(f"✅ Genesis simulation ready on {cfg.genesis_device}")
    
    return scene, robot, camera, palm_target_marker, finger_force_markers


# ============ FABRICS Controller Setup ============
def setup_fabrics_controller(cfg: FabricTestConfig):
    """Initialize FABRICS controller"""
    print(f"Setting up FABRICS controller on {cfg.fabrics_device}...")
    
    # Initialize warp
    fabrics_device_int = int(cfg.fabrics_device.split(":")[-1])
    initialize_warp(str(fabrics_device_int))
    
    # Create world model
    world_model = WorldMeshesModel(
        batch_size=cfg.num_envs,
        max_objects_per_env=20,
        device=cfg.fabrics_device,
        world_filename='floor'
    )
    object_ids, object_indicator = world_model.get_object_ids()
    
    # Create CS63-Tesollo fabric
    cs63_fabric = CS63TesolloFabric(
        batch_size=cfg.num_envs,
        device=cfg.fabrics_device,
        timestep=cfg.fabrics_dt,
        num_arm_joints=6,
        num_gripper_joints=12,
        num_fingers=3,
        graph_capturable=cfg.cuda_graph
    )
    
    # Create integrator
    cs63_integrator = DisplacementIntegrator(cs63_fabric)
    
    print(f"✅ FABRICS controller ready on {cfg.fabrics_device}")
    
    return cs63_fabric, cs63_integrator, object_ids, object_indicator


# ============ Device Transfer Helpers ============
def to_fabrics(tensor: torch.Tensor, cfg: FabricTestConfig) -> torch.Tensor:
    """Transfer tensor from Genesis device to FABRICS device"""
    return tensor.to(cfg.fabrics_device)


def to_genesis(tensor: torch.Tensor, cfg: FabricTestConfig) -> torch.Tensor:
    """Transfer tensor from FABRICS device to Genesis device"""
    return tensor.to(cfg.genesis_device)


# ============ Coordinate Transform Helpers ============
def world_to_base_frame(world_pose: torch.Tensor, base_height: float) -> torch.Tensor:
    """Convert pose from world frame to robot base frame"""
    base_pose = world_pose.clone()
    base_pose[:, 2] -= base_height
    return base_pose


def base_to_world_frame(base_pose: torch.Tensor, base_height: float) -> torch.Tensor:
    """Convert pose from robot base frame to world frame"""
    world_pose = base_pose.clone()
    world_pose[:, 2] += base_height
    return world_pose


# ============ Visualization Helpers ============
def quat_multiply(q1, q2):
    """Multiply two quaternions: q_result = q1 * q2"""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quat_from_axis_angle(axis, angle):
    """Create quaternion from axis-angle representation"""
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    sin_half = np.sin(half_angle)
    x, y, z = axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half
    return np.array([w, x, y, z])


def update_palm_target_marker(markers, palm_target, cfg):
    """Update palm target marker: Z-axis + position sphere"""
    if markers is None or len(markers) != 2:
        return
    
    pos = palm_target[0, :3].detach().cpu().numpy()
    pos[2] += cfg.base_height
    
    # Convert Euler ZYX to quaternion
    euler_zyx = palm_target[0, 3:6].detach().cpu().numpy()
    quat = xyz_to_quat(euler_zyx, rpy=True, degrees=False)
    
    # Z-axis (blue) - shows orientation
    markers[0].set_pos(pos)
    markers[0].set_quat(quat)
    
    # Position sphere - shows tracked palm position
    markers[1].set_pos(pos)


def update_finger_force_markers(markers, finger_force_frames_pose, finger_forces, cfg):
    """Update finger force arrow markers"""
    if not markers:
        return
    
    forces_cpu = finger_forces[0].cpu().numpy()
    force_frames_pos = finger_force_frames_pose[0, :, :3].cpu().numpy()
    force_frames_quat = finger_force_frames_pose[0, :, 3:].cpu().numpy()
    
    for i, marker in enumerate(markers):
        frame_pos = force_frames_pos[i]
        force_local = forces_cpu[i]
        frame_quat = force_frames_quat[i]
        
        force = transform_by_quat(force_local, frame_quat)
        # Swap Y and Z axes for visualization
        force = np.array([force[0], force[2], force[1]])

        force_mag = np.linalg.norm(force)
        
        if force_mag > 0.01:
            # Calculate arrow parameters based on force magnitude
            arrow_length = force_mag * cfg.force_scale
            arrow_radius = max(0.003, force_mag * 0.001)  # Radius increases with force magnitude
            
            # Update cylinder dimensions
            marker._radius = arrow_radius
            marker._height = arrow_length
            
            # Calculate arrow end position and orientation
            force_dir = force / force_mag
            arrow_end = frame_pos + force_dir * arrow_length
            mid_pos = (frame_pos + arrow_end) / 2
            marker.set_pos(mid_pos)
            
            # Calculate arrow orientation
            z_axis = np.array([0, 0, 1])
            if np.abs(np.dot(force_dir, z_axis)) < 0.999:
                axis = np.cross(z_axis, force_dir)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.dot(z_axis, force_dir))
                quat = np.array([
                    np.cos(angle/2),
                    axis[0] * np.sin(angle/2),
                    axis[1] * np.sin(angle/2),
                    axis[2] * np.sin(angle/2)
                ])
                marker.set_quat(quat)
        else:
            # Hide marker
            marker.set_pos(np.array([100.0, 100.0, 100.0]))


# ============ Data Logging and Plotting ============
def plot_control_data(data_dict, save_path):
    """Plot control commands and tracking errors"""
    times = data_dict['times']
    q_actual = data_dict['q_actual']
    q_desired = data_dict['q_desired']
    qd_actual = data_dict['qd_actual']
    qd_desired = data_dict['qd_desired']
    palm_targets = data_dict['palm_targets']
    palm_targets_actual = data_dict.get('palm_targets_actual', None)
    
    num_arm_joints = 6
    num_gripper_joints = 12
    
    arm_joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
    
    # Figure 1: Arm Joint Positions
    fig1 = plt.figure(figsize=(16, 12))
    fig1.suptitle('Arm Joint Positions - Individual Tracking', fontsize=16, fontweight='bold')
    
    for i in range(num_arm_joints):
        ax = plt.subplot(3, 2, i+1)
        ax.plot(times, q_actual[:, i], label='Actual', linewidth=2, color='blue')
        ax.plot(times, q_desired[:, i], label='Desired', linewidth=2, linestyle='--', color='red')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Position (rad)', fontsize=10)
        ax.set_title(f'Joint {i}: {arm_joint_names[i]}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1_path = save_path.replace('.png', '_arm_positions.png')
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"✅ Arm positions plot saved to: {fig1_path}")
    plt.close()
    
    # Figure 2: Summary Statistics
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle('Tracking Summary & Errors', fontsize=16, fontweight='bold')
    
    # Position errors
    ax1 = plt.subplot(2, 3, 1)
    q_errors = np.abs(q_actual - q_desired)
    for i in range(num_arm_joints):
        ax1.plot(times, q_errors[:, i], label=f'J{i}: {arm_joint_names[i]}', linewidth=1.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position Error (rad)')
    ax1.set_title('Arm Position Errors (per joint)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Velocity errors
    ax2 = plt.subplot(2, 3, 2)
    qd_errors = np.abs(qd_actual - qd_desired)
    for i in range(num_arm_joints):
        ax2.plot(times, qd_errors[:, i], label=f'J{i}: {arm_joint_names[i]}', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity Error (rad/s)')
    ax2.set_title('Arm Velocity Errors (per joint)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Average errors
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times, q_errors[:, :num_arm_joints].mean(axis=1), 
            label='Arm avg error', linewidth=2)
    ax3.plot(times, q_errors[:, num_arm_joints:].mean(axis=1), 
            label='Gripper avg error', linewidth=2)
    ax3.plot(times, q_errors.mean(axis=1), 
            label='Total avg error', linewidth=2, linestyle='--')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position Error (rad)')
    ax3.set_title('Average Position Tracking Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Palm target position
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(times, palm_targets[:, 0], label='X_target', linewidth=2)
    ax4.plot(times, palm_targets[:, 1], label='Y_target', linewidth=2)
    ax4.plot(times, palm_targets[:, 2], label='Z_target', linewidth=2)
    if palm_targets_actual is not None and palm_targets_actual.shape[1] >= 3:
        ax4.plot(times, palm_targets_actual[:, 0], label='X_actual', linewidth=1.5, linestyle='--')
        ax4.plot(times, palm_targets_actual[:, 1], label='Y_actual', linewidth=1.5, linestyle='--')
        ax4.plot(times, palm_targets_actual[:, 2], label='Z_actual', linewidth=1.5, linestyle='--')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('Palm Target Position')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Palm target orientation
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(times, np.rad2deg(palm_targets[:, 3]), label='Euler Z', linewidth=2)
    ax5.plot(times, np.rad2deg(palm_targets[:, 4]), label='Euler Y', linewidth=2)
    ax5.plot(times, np.rad2deg(palm_targets[:, 5]), label='Euler X', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angle (deg)')
    ax5.set_title('Palm Target Orientation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Gripper errors per finger
    ax6 = plt.subplot(2, 3, 6)
    for finger in range(3):
        finger_errors = q_errors[:, num_arm_joints + finger*4 : num_arm_joints + (finger+1)*4].mean(axis=1)
        ax6.plot(times, finger_errors, label=f'Finger {finger+1}', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Position Error (rad)')
    ax6.set_title('Gripper Errors (avg per finger)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Summary plot saved to: {save_path}")
    plt.close()
