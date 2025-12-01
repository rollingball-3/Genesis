"""
Basic integration between Genesis simulation and FABRICS control.

This script demonstrates:
1. Genesis simulation setup with CS63 arm + DG3F gripper (using Manipulator class)
2. FABRICS controller initialization
3. Bidirectional data flow: Genesis state -> FABRICS -> Genesis PD control
4. Video recording of the simulation
5. Visualization of palm target and finger forces

Note: Genesis and FABRICS use separate CUDA devices to avoid Taichi/Warp conflicts.
- Genesis (Taichi): cuda:0
- FABRICS (Warp): cuda:1

Usage:
    python genesis_fabric_basic.py
"""
import os
import sys
import time
import yaml
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Genesis imports
import genesis as gs
from genesis.utils.geom import quat_to_xyz, xyz_to_quat

# Import Manipulator class from throw_env
sys.path.append(str(Path(__file__).parent.parent))
from manipulation.throw_env import Manipulator

# FABRICS imports
sys.path.append(str(Path(__file__).parent.parent.parent / "genesis/ext/FABRICS/src"))
from fabrics_sim.fabrics.cs63_tesollo_fabric import CS63TesolloFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel


# ============ Load Initial Config ============
def load_initial_joint_config():
    """Load initial joint configuration from YAML file
    
    NOTE: set_qpos expects finger-by-finger order (F1M1-4, F2M1-4, F3M1-4)
    but gripper_qpos returns entity order (F1M1,F2M1,F3M1, F1M2,F2M2,F3M2, ...)
    """
    config_path = Path(__file__).parent.parent / "manipulation" / "initial_joint_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse arm joints in order
    arm_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    arm_positions = [config["arm_joints"][name] for name in arm_joint_names]
    
    # Parse gripper joints in finger-by-finger order (for set_qpos)
    gripper_joint_names = [
        "F1M1", "F1M2", "F1M3", "F1M4",
        "F2M1", "F2M2", "F2M3", "F2M4",
        "F3M1", "F3M2", "F3M3", "F3M4"
    ]
    gripper_positions = [config["gripper_joints"][name] for name in gripper_joint_names]
    
    return arm_positions + gripper_positions


# ============ Configuration ============
class Config:
    """Simulation and control configuration
    
    Unified 60 Hz timing:
    - sim_dt = 1/60 (60 Hz physics simulation)
    - fabrics_dt = 1/60 (60 Hz FABRICS integration)
    - control_dt = 1/60 (60 Hz control updates)
    - All components run at the same frequency
    """
    # Genesis simulation (Taichi backend)
    num_envs = 1
    sim_dt = 1.0 / 60.0  # 60 Hz physics simulation
    total_time = 10.0  # seconds
    genesis_device = "cuda:0"  # Genesis uses cuda:0
    
    # FABRICS (Warp backend) - use separate device to avoid conflicts
    fabrics_device = "cuda:1"  # FABRICS uses cuda:1
    cuda_graph = False  # Start with False for debugging
    
    # FABRICS timing - higher frequency than simulation
    fabric_decimation = 2  # Number of FABRICS integration steps per physics step
    fabrics_dt = sim_dt / fabric_decimation  # FABRICS runs at 120 Hz when sim is 60 Hz
    
    # Control timing - same as simulation
    control_dt = 1.0 / 60.0  # 60 Hz control updates (same as sim_dt)
    
    # Recording
    save_video = True
    base_dir = "tmp/fabrics_test"  # Base directory for all results
    
    # Data logging
    save_control_data = True  # Save control commands to file
    plot_control_data = True  # Generate control plots
    
    # Testing/Debugging mode
    test_mode_simple = False  # Simplified control targets for debugging
    # When test_mode_simple=True:
    #   - Finger forces are set to zero
    #   - Palm target is fixed (not updated during simulation)
    
    # Visualization
    show_palm_target = True  # Show palm target marker
    show_finger_forces = True  # Show finger force arrows
    force_scale = 0.1  # Scale factor for force visualization (m/N)
    
    # Palm target offset (in meters)
    palm_target_offset_x = -0.20  # X direction offset: -20cm backward
    palm_target_offset_y = 0.05  # Y direction offset: +5cm to the side
    palm_target_offset_z = 0.35  # Z direction offset: +35cm upward (compensate base_height)


# ============ Genesis Simulation Setup ============
def setup_genesis_simulation(cfg: Config):
    """Initialize Genesis simulation with CS63 + DG3F using Manipulator class"""
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
            show_world_frame=True,
            world_frame_size=0.5,
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
    
    # Create Manipulator instance (handles arm + gripper setup)
    robot = Manipulator(
        scene=scene,
        num_envs=cfg.num_envs,
        device=cfg.genesis_device,
        base_height=0.6,  # Elevated to avoid ground collision
    )
    
    # Add camera for recording
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(2.0, 0.0, 1.5),
        lookat=(0.0, 0.0, 0.5),
        fov=40,
        GUI=False,
    )
    
    # Add visualization markers (visual only, no collision, fixed)
    # Palm target marker (red sphere)
    palm_target_marker = None
    if cfg.show_palm_target:
        palm_target_marker = scene.add_entity(
            gs.morphs.Sphere(
                radius=0.02,
                pos=(0.0, 0.0, 0.5),
                collision=False,  # Disable collision - visual only
                fixed=True,  # Fixed in space, no dynamics
            ),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0, 0.8)),  # Red, semi-transparent
            visualize_contact=False,
        )
    
    # Finger force arrows (3 arrows for 3 fingers)
    finger_force_markers = []
    if cfg.show_finger_forces:
        colors = [(0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0)]  # Green, Blue, Yellow
        for i, color in enumerate(colors):
            # Create arrow as cylinder (visual only, no collision, fixed)
            arrow = scene.add_entity(
                gs.morphs.Cylinder(
                    radius=0.005,
                    height=0.05,
                    pos=(0.0, 0.0, 0.5),
                    collision=False,  # Disable collision - visual only
                    fixed=True,  # Fixed in space, no dynamics
                ),
                surface=gs.surfaces.Default(color=color),
                visualize_contact=False,
            )
            finger_force_markers.append(arrow)
    
    # Build scene
    scene.build(n_envs=cfg.num_envs)
    
    # Set PD gains (must be called after scene.build)
    robot.set_pd_gains()
    
    print(f"✅ Genesis simulation ready on {cfg.genesis_device}")
    
    return scene, robot, camera, palm_target_marker, finger_force_markers


# ============ FABRICS Controller Setup ============
def setup_fabrics_controller(cfg: Config):
    """Initialize FABRICS controller on separate device"""
    print(f"Setting up FABRICS controller on {cfg.fabrics_device}...")
    
    # Initialize warp on FABRICS device
    fabrics_device_int = int(cfg.fabrics_device.split(":")[-1])
    initialize_warp(str(fabrics_device_int))
    
    # Create world model for collision avoidance
    world_model = WorldMeshesModel(
        batch_size=cfg.num_envs,
        max_objects_per_env=20,
        device=cfg.fabrics_device,
        world_filename='floor'
    )
    object_ids, object_indicator = world_model.get_object_ids()
    
    # Create CS63-Tesollo fabric (use fabrics_dt for integration timestep)
    cs63_fabric = CS63TesolloFabric(
        batch_size=cfg.num_envs,
        device=cfg.fabrics_device,
        timestep=cfg.fabrics_dt,  # Use smaller timestep for better accuracy
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
def to_fabrics(tensor: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Transfer tensor from Genesis device to FABRICS device"""
    return tensor.to(cfg.fabrics_device)


def to_genesis(tensor: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Transfer tensor from FABRICS device to Genesis device"""
    return tensor.to(cfg.genesis_device)


# ============ Data Logging and Plotting ============
def plot_control_data(data_dict, save_path):
    """Plot control commands and tracking errors - each joint separately
    
    Args:
        data_dict: Dictionary containing logged data
        save_path: Path to save the plot
    """
    times = data_dict['times']
    q_actual = data_dict['q_actual']  # [T, 18]
    q_desired = data_dict['q_desired']  # [T, 18]
    qd_actual = data_dict['qd_actual']  # [T, 18]
    qd_desired = data_dict['qd_desired']  # [T, 18]
    palm_targets = data_dict['palm_targets']  # [T, 6]
    
    num_arm_joints = 6
    num_gripper_joints = 12
    
    # Joint names for labeling
    arm_joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
    
    # ========== Figure 1: Individual Arm Joint Positions (3x2 grid) ==========
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
    
    # ========== Figure 2: Individual Arm Joint Velocities (3x2 grid) ==========
    fig2 = plt.figure(figsize=(16, 12))
    fig2.suptitle('Arm Joint Velocities - Individual Tracking', fontsize=16, fontweight='bold')
    
    for i in range(num_arm_joints):
        ax = plt.subplot(3, 2, i+1)
        ax.plot(times, qd_actual[:, i], label='Actual', linewidth=2, color='blue')
        ax.plot(times, qd_desired[:, i], label='Desired', linewidth=2, linestyle='--', color='red')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Velocity (rad/s)', fontsize=10)
        ax.set_title(f'Joint {i}: {arm_joint_names[i]}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2_path = save_path.replace('.png', '_arm_velocities.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"✅ Arm velocities plot saved to: {fig2_path}")
    plt.close()
    
    # ========== Figure 3: Individual Gripper Joint Positions (4x3 grid) ==========
    fig3 = plt.figure(figsize=(18, 16))
    fig3.suptitle('Gripper Joint Positions - Individual Tracking', fontsize=16, fontweight='bold')
    
    for i in range(num_gripper_joints):
        ax = plt.subplot(4, 3, i+1)
        ax.plot(times, q_actual[:, num_arm_joints + i], label='Actual', linewidth=2, color='green')
        ax.plot(times, q_desired[:, num_arm_joints + i], label='Desired', linewidth=2, linestyle='--', color='orange')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Position (rad)', fontsize=9)
        finger_num = i // 4 + 1
        motor_num = i % 4 + 1
        ax.set_title(f'F{finger_num}M{motor_num} (Joint {num_arm_joints + i})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig3_path = save_path.replace('.png', '_gripper_positions.png')
    plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gripper positions plot saved to: {fig3_path}")
    plt.close()
    
    # ========== Figure 4: Summary Statistics ==========
    fig4 = plt.figure(figsize=(16, 10))
    fig4.suptitle('Tracking Summary & Errors', fontsize=16, fontweight='bold')
    
    # 1. Position tracking errors per joint
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
    
    # 2. Velocity tracking errors per joint
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
    
    # 3. Average tracking errors
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
    
    # 4. Palm target position
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(times, palm_targets[:, 0], label='X', linewidth=2)
    ax4.plot(times, palm_targets[:, 1], label='Y', linewidth=2)
    ax4.plot(times, palm_targets[:, 2], label='Z', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('Palm Target Position')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Palm target orientation (logged but not used for control)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(times, np.rad2deg(palm_targets[:, 3]), label='Euler Z', linewidth=2)
    ax5.plot(times, np.rad2deg(palm_targets[:, 4]), label='Euler Y', linewidth=2)
    ax5.plot(times, np.rad2deg(palm_targets[:, 5]), label='Euler X', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angle (deg)')
    ax5.set_title('Palm Target Orientation (Not Used)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.text(0.5, 0.95, 'Orientation kept for API compatibility only', 
             ha='center', va='top', transform=ax5.transAxes, 
             fontsize=8, style='italic', color='red')
    
    # 6. Gripper position errors (average per finger)
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
    
    # ========== Figure 5: FABRICS Output - Arm Positions (3x2 grid) ==========
    if 'q_fabrics' in data_dict:
        q_fabrics = data_dict['q_fabrics']  # [T, 18]
        qd_fabrics = data_dict['qd_fabrics']  # [T, 18]
        qdd_fabrics = data_dict['qdd_fabrics']  # [T, 18]
        
        fig5 = plt.figure(figsize=(16, 12))
        fig5.suptitle('FABRICS Output - Arm Joint Positions', fontsize=16, fontweight='bold')
        
        for i in range(num_arm_joints):
            ax = plt.subplot(3, 2, i+1)
            ax.plot(times, q_actual[:, i], label='Actual (Robot)', linewidth=2, color='blue', alpha=0.7)
            ax.plot(times, q_fabrics[:, i], label='FABRICS Output', linewidth=2, color='red', linestyle='--')
            ax.plot(times, q_desired[:, i], label='PD Command', linewidth=1.5, color='green', linestyle=':')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Position (rad)', fontsize=10)
            ax.set_title(f'Joint {i}: {arm_joint_names[i]}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig5_path = save_path.replace('.png', '_fabrics_arm_positions.png')
        plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
        print(f"✅ FABRICS arm positions plot saved to: {fig5_path}")
        plt.close()
        
        # ========== Figure 6: FABRICS Output - Arm Velocities (3x2 grid) ==========
        fig6 = plt.figure(figsize=(16, 12))
        fig6.suptitle('FABRICS Output - Arm Joint Velocities', fontsize=16, fontweight='bold')
        
        for i in range(num_arm_joints):
            ax = plt.subplot(3, 2, i+1)
            ax.plot(times, qd_actual[:, i], label='Actual (Robot)', linewidth=2, color='blue', alpha=0.7)
            ax.plot(times, qd_fabrics[:, i], label='FABRICS Output', linewidth=2, color='red', linestyle='--')
            ax.plot(times, qd_desired[:, i], label='PD Command', linewidth=1.5, color='green', linestyle=':')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Velocity (rad/s)', fontsize=10)
            ax.set_title(f'Joint {i}: {arm_joint_names[i]}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig6_path = save_path.replace('.png', '_fabrics_arm_velocities.png')
        plt.savefig(fig6_path, dpi=150, bbox_inches='tight')
        print(f"✅ FABRICS arm velocities plot saved to: {fig6_path}")
        plt.close()
        
        # ========== Figure 7: FABRICS Output - Arm Accelerations (3x2 grid) ==========
        fig7 = plt.figure(figsize=(16, 12))
        fig7.suptitle('FABRICS Output - Arm Joint Accelerations', fontsize=16, fontweight='bold')
        
        for i in range(num_arm_joints):
            ax = plt.subplot(3, 2, i+1)
            ax.plot(times, qdd_fabrics[:, i], label='FABRICS qdd', linewidth=2, color='purple')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Acceleration (rad/s²)', fontsize=10)
            ax.set_title(f'Joint {i}: {arm_joint_names[i]}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig7_path = save_path.replace('.png', '_fabrics_arm_accelerations.png')
        plt.savefig(fig7_path, dpi=150, bbox_inches='tight')
        print(f"✅ FABRICS arm accelerations plot saved to: {fig7_path}")
        plt.close()
        
        # ========== Figure 8: FABRICS vs PD Command Comparison ==========
        fig8 = plt.figure(figsize=(18, 12))
        fig8.suptitle('FABRICS Output vs PD Command - All Joints', fontsize=16, fontweight='bold')
        
        # Position comparison
        ax1 = plt.subplot(3, 1, 1)
        for i in range(num_arm_joints):
            fabrics_pd_diff = q_fabrics[:, i] - q_desired[:, i]
            ax1.plot(times, fabrics_pd_diff, label=f'J{i}: {arm_joint_names[i]}', linewidth=1.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position Difference (rad)')
        ax1.set_title('Position: FABRICS Output - PD Command (Arm Joints)')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
        
        # Velocity comparison
        ax2 = plt.subplot(3, 1, 2)
        for i in range(num_arm_joints):
            fabrics_pd_diff = qd_fabrics[:, i] - qd_desired[:, i]
            ax2.plot(times, fabrics_pd_diff, label=f'J{i}: {arm_joint_names[i]}', linewidth=1.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity Difference (rad/s)')
        ax2.set_title('Velocity: FABRICS Output - PD Command (Arm Joints)')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
        
        # Acceleration magnitude
        ax3 = plt.subplot(3, 1, 3)
        for i in range(num_arm_joints):
            ax3.plot(times, np.abs(qdd_fabrics[:, i]), label=f'J{i}: {arm_joint_names[i]}', linewidth=1.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('|Acceleration| (rad/s²)')
        ax3.set_title('FABRICS Acceleration Magnitude (Arm Joints)')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig8_path = save_path.replace('.png', '_fabrics_comparison.png')
        plt.savefig(fig8_path, dpi=150, bbox_inches='tight')
        print(f"✅ FABRICS comparison plot saved to: {fig8_path}")
        plt.close()


# ============ Visualization Helpers ============
def update_palm_target_marker(marker, palm_target, cfg):
    """Update palm target marker position"""
    if marker is None:
        return
    # palm_target is [B, 6] on FABRICS device: [x, y, z, euler_z, euler_y, euler_x]
    pos = palm_target[0, :3].cpu().numpy()
    marker.set_pos(pos)


def update_finger_force_markers(markers, finger_tips_pose, finger_forces, cfg):
    """Update finger force arrow markers"""
    if not markers:
        return
    # finger_tips_pose: [B, 3, 7] on Genesis device
    # finger_forces: [B, 3, 3] on FABRICS device
    forces_cpu = finger_forces[0].cpu().numpy()  # [3, 3]
    tips_pos = finger_tips_pose[0, :, :3].cpu().numpy()  # [3, 3]
    
    for i, marker in enumerate(markers):
        # Get fingertip position
        tip_pos = tips_pos[i]
        force = forces_cpu[i]
        force_mag = np.linalg.norm(force)
        
        if force_mag > 0.01:  # Only show if force is significant
            # Arrow end position: tip + force * scale
            arrow_end = tip_pos + force * cfg.force_scale
            # Position arrow at midpoint
            mid_pos = (tip_pos + arrow_end) / 2
            marker.set_pos(mid_pos)
            
            # Calculate arrow orientation from force direction
            force_dir = force / force_mag
            # Create rotation to align Z-axis with force direction
            # Simple approach: use angle-axis
            z_axis = np.array([0, 0, 1])
            if np.abs(np.dot(force_dir, z_axis)) < 0.999:
                axis = np.cross(z_axis, force_dir)
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.dot(z_axis, force_dir))
                # Convert to quaternion (w, x, y, z)
                quat = np.array([
                    np.cos(angle/2),
                    axis[0] * np.sin(angle/2),
                    axis[1] * np.sin(angle/2),
                    axis[2] * np.sin(angle/2)
                ])
                marker.set_quat(quat)
        else:
            # Hide marker by moving it far away
            marker.set_pos(np.array([100.0, 100.0, 100.0]))


# ============ Main Control Loop ============
def main():
    cfg = Config()
    
    # Create timestamped output directory
    timestamp = int(time.time())
    output_dir = os.path.join(cfg.base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print("\n" + "=" * 60)
    print("Genesis + FABRICS Integration (Dual GPU)")
    print(f"  Genesis (Taichi): {cfg.genesis_device}")
    print(f"  FABRICS (Warp):   {cfg.fabrics_device}")
    print("=" * 60 + "\n")
    
    # Setup Genesis simulation (using Manipulator class)
    scene, robot, camera, palm_target_marker, finger_force_markers = setup_genesis_simulation(cfg)
    
    # Setup FABRICS controller on separate device
    cs63_fabric, cs63_integrator, object_ids, object_indicator = setup_fabrics_controller(cfg)
    print(f"Number of joints: {cs63_fabric.num_joints}")
    # Load initial joint states from YAML config
    initial_config = load_initial_joint_config()
    q_initial = torch.tensor(initial_config, device=cfg.genesis_device).unsqueeze(0)
    
    print(f"Initial joint config (from YAML):")
    print(f"  Arm: {initial_config[:6]}")
    print(f"  Gripper: {initial_config[6:]}")
    
    # FABRICS state tensors - on FABRICS device
    # Initialize FABRICS state from initial joint configuration
    fabric_q = to_fabrics(q_initial.clone(), cfg)  # [B, 18]
    fabric_qd = torch.zeros(cfg.num_envs, 18, device=cfg.fabrics_device)
    fabric_qdd = torch.zeros(cfg.num_envs, 18, device=cfg.fabrics_device)
    
    # Set initial positions in Genesis using Manipulator
    robot.reset()
    
    # FABRICS control targets - on FABRICS device
    # 1. Palm target: current palm pose (6D: position + orientation)
    # Note: Only position is used for control, orientation is kept for API compatibility
    current_palm_pose = robot.palm_pose  # [B, 7] (pos + quat) on Genesis device
    palm_target = torch.zeros(cfg.num_envs, 6, device=cfg.fabrics_device)  # 6D pose (API compatible)
    
    # Read current position and add configurable offset
    current_position = current_palm_pose[:, :3]  # [B, 3] on Genesis device
    palm_target[:, 0] = to_fabrics(current_position[:, 0], cfg) + cfg.palm_target_offset_x
    palm_target[:, 1] = to_fabrics(current_position[:, 1], cfg) + cfg.palm_target_offset_y
    palm_target[:, 2] = to_fabrics(current_position[:, 2], cfg) + cfg.palm_target_offset_z
    
    # Keep orientation as current (convert quaternion to Euler ZYX) - not used for control
    current_quat = current_palm_pose[:, 3:7]  # [w, x, y, z] format
    current_euler = quat_to_xyz(current_quat, rpy=True, degrees=False)  # On Genesis device
    palm_target[:, 3:6] = to_fabrics(current_euler, cfg)  # Transfer to FABRICS device
    
    # 2. Finger forces - on FABRICS device
    finger_forces = torch.zeros(cfg.num_envs, 3, 3, device=cfg.fabrics_device)
    if not cfg.test_mode_simple:
        finger_forces[:, 0, 0] = 1.0  # First finger (index 0), X direction, 1.0 N
    
    print(f"\n=== Control Mode ===")
    if cfg.test_mode_simple:
        print("  Mode: SIMPLIFIED (test_mode_simple=True)")
        print("  - Finger forces: ALL ZERO")
        print("  - Palm target: FIXED (no updates during simulation)")
    else:
        print("  Mode: FULL CONTROL")
        print("  - Finger forces: ACTIVE")
        print("  - Palm target: FIXED OFFSET (current position + ΔX, ΔY)")
    
    # Prepare video recording
    if cfg.save_video:
        video_path = os.path.join(output_dir, "simulation.mp4")
        print(f"Will save video to: {video_path}")
        camera.start_recording()
    
    # Print initial control targets
    print("\n=== FABRICS Control Targets ===")
    print(f"Initial palm position: {current_palm_pose[0, :3].cpu().numpy()}")
    print(f"Initial palm quat:     {current_palm_pose[0, 3:7].cpu().numpy()}")
    print(f"Target palm position:  {palm_target[0, :3].cpu().numpy()}")
    print(f"Target palm ori (Euler ZYX): {palm_target[0, 3:6].cpu().numpy()}")
    print(f"Note: Only position is used for control, orientation is kept for API compatibility")
    print(f"\nPalm target offsets:")
    print(f"  ΔX: {cfg.palm_target_offset_x:+.3f} m")
    print(f"  ΔY: {cfg.palm_target_offset_y:+.3f} m")
    print(f"  ΔZ: {cfg.palm_target_offset_z:+.3f} m")
    print(f"  Total offset: {np.sqrt(cfg.palm_target_offset_x**2 + cfg.palm_target_offset_y**2 + cfg.palm_target_offset_z**2):.3f} m")
    print(f"\nFinger forces:")
    print(f"  Finger 1: {finger_forces[0, 0, :].cpu().numpy()} N")
    print(f"  Finger 2: {finger_forces[0, 1, :].cpu().numpy()} N")
    print(f"  Finger 3: {finger_forces[0, 2, :].cpu().numpy()} N")
    
    # === FIXED PD CONTROL TARGET (for testing) ===
    # Store initial joint configuration as fixed target
    q_fixed_target = q_initial.clone()  # [B, 18] unified joint order
    qd_fixed_target = torch.zeros(cfg.num_envs, 18, device=cfg.genesis_device)  # Zero velocity
    print(f"\n=== FIXED PD TARGET (TEST MODE) ===")
    print(f"Position target: {q_fixed_target[0].cpu().numpy()}")
    print(f"Velocity target: {qd_fixed_target[0].cpu().numpy()}")
    print("PD controller will use this fixed target throughout simulation")
    
    # Main control loop
    # Multi-rate timing (similar to IsaacLab):
    # - Physics simulation: 60 Hz
    # - FABRICS integration: 120 Hz (2x decimation)
    # - Control updates: 60 Hz
    print("\nStarting control loop...")
    print(f"  Physics simulation: {1/cfg.sim_dt:.0f} Hz (dt={cfg.sim_dt:.6f}s)")
    print(f"  FABRICS integration: {1/cfg.fabrics_dt:.0f} Hz (dt={cfg.fabrics_dt:.6f}s)")
    print(f"  Fabric decimation: {cfg.fabric_decimation}x")
    print(f"  Control updates: {1/cfg.control_dt:.0f} Hz")
    
    num_control_steps = int(cfg.total_time / cfg.control_dt)
    start_time = time.time()
    
    # Initialize data logging
    if cfg.save_control_data or cfg.plot_control_data:
        log_data = {
            'times': [],
            'q_actual': [],
            'q_desired': [],  # PD command sent to robot
            'qd_actual': [],
            'qd_desired': [],  # PD command sent to robot
            'q_fabrics': [],  # FABRICS output
            'qd_fabrics': [],  # FABRICS output
            'qdd_fabrics': [],  # FABRICS output
            'palm_targets': [],
        }
    
    # Error tracking
    error_occurred = False
    error_message = ""
    error_step = -1
    
    # Diagnostics tracking
    diagnostics_log = {
        'fabric_q_norm': [],
        'fabric_qd_norm': [],
        'fabric_qdd_norm': [],
        'fabric_q_max': [],
        'fabric_qd_max': [],
        'fabric_q_has_nan': [],
        'fabric_qd_has_nan': [],
    }
    
    # Detailed integration trace (for offline reproduction)
    integration_trace = {
        'control_step': [],
        'decimation_step': [],
        'timestamp': [],
        'dt': [],
        # State before integration
        'q_before': [],
        'qd_before': [],
        'qdd_before': [],
        # State after integration
        'q_after': [],
        'qd_after': [],
        'qdd_after': [],
        # Palm target
        'palm_target_pos': [],
        'palm_target_ori': [],
        # Finger forces
        'finger_forces': [],
    }
    
    for control_step in range(num_control_steps):
        try:
            # === Read state from Genesis (on Genesis device) - ONLY for logging ===
            q_genesis = torch.cat([robot.arm_qpos, robot.gripper_qpos], dim=1)
            qd_genesis = torch.cat([robot.arm_qvel, robot.gripper_qvel], dim=1)
            
            # === Pre-integration diagnostics ===
            fabric_q_has_nan = torch.isnan(fabric_q).any().item()
            fabric_qd_has_nan = torch.isnan(fabric_qd).any().item()
            
            if fabric_q_has_nan or fabric_qd_has_nan:
                print(f"\n⚠️  WARNING at step {control_step}: NaN detected BEFORE integration!")
                print(f"  fabric_q has NaN: {fabric_q_has_nan}")
                print(f"  fabric_qd has NaN: {fabric_qd_has_nan}")
                if fabric_q_has_nan:
                    print(f"  fabric_q: {fabric_q[0].cpu().numpy()}")
                if fabric_qd_has_nan:
                    print(f"  fabric_qd: {fabric_qd[0].cpu().numpy()}")
            
            # Log diagnostics
            diagnostics_log['fabric_q_norm'].append(torch.norm(fabric_q).item())
            diagnostics_log['fabric_qd_norm'].append(torch.norm(fabric_qd).item())
            diagnostics_log['fabric_qdd_norm'].append(torch.norm(fabric_qdd).item())
            diagnostics_log['fabric_q_max'].append(torch.max(torch.abs(fabric_q)).item())
            diagnostics_log['fabric_qd_max'].append(torch.max(torch.abs(fabric_qd)).item())
            diagnostics_log['fabric_q_has_nan'].append(fabric_q_has_nan)
            diagnostics_log['fabric_qd_has_nan'].append(fabric_qd_has_nan)
            
            # === FABRICS computation (on FABRICS device) ===
            # Set features for FABRICS (targets are already on FABRICS device)
            cs63_fabric.set_features(
                finger_forces, palm_target, "euler_zyx",
                fabric_q.detach(), fabric_qd.detach(),
                object_ids, object_indicator
            )
            
            # Multiple FABRICS integration steps per physics step (fabric decimation)
            for decimation_step in range(cfg.fabric_decimation):
                # Record state BEFORE integration
                q_before = fabric_q.clone()
                qd_before = fabric_qd.clone()
                qdd_before = fabric_qdd.clone()
                
                # Perform integration step
                fabric_q_new, fabric_qd_new, fabric_qdd_new = cs63_integrator.step(
                    fabric_q.detach(), fabric_qd.detach(), fabric_qdd.detach(), cfg.fabrics_dt
                )
                
                # Record integration trace (detailed log for offline reproduction)
                integration_trace['control_step'].append(control_step)
                integration_trace['decimation_step'].append(decimation_step)
                integration_trace['timestamp'].append(control_step * cfg.control_dt + decimation_step * cfg.fabrics_dt)
                integration_trace['dt'].append(cfg.fabrics_dt)
                
                # State before
                integration_trace['q_before'].append(q_before[0].cpu().numpy())
                integration_trace['qd_before'].append(qd_before[0].cpu().numpy())
                integration_trace['qdd_before'].append(qdd_before[0].cpu().numpy())
                
                # State after
                integration_trace['q_after'].append(fabric_q_new[0].cpu().numpy())
                integration_trace['qd_after'].append(fabric_qd_new[0].cpu().numpy())
                integration_trace['qdd_after'].append(fabric_qdd_new[0].cpu().numpy())
                
                # Control targets (constant during decimation loop)
                integration_trace['palm_target_pos'].append(palm_target[0, :3].cpu().numpy())
                integration_trace['palm_target_ori'].append(palm_target[0, 3:6].cpu().numpy())
                integration_trace['finger_forces'].append(finger_forces[0].cpu().numpy().flatten())
                
                # Update FABRICS state in-place (persistent state evolution)
                fabric_q.copy_(fabric_q_new)
                fabric_qd.copy_(fabric_qd_new)
                fabric_qdd.copy_(fabric_qdd_new)
            
            # === Post-integration diagnostics ===
            fabric_q_has_nan_post = torch.isnan(fabric_q).any().item()
            fabric_qd_has_nan_post = torch.isnan(fabric_qd).any().item()
            
            if fabric_q_has_nan_post or fabric_qd_has_nan_post:
                print(f"\n⚠️  WARNING at step {control_step}: NaN detected AFTER integration!")
                print(f"  fabric_q has NaN: {fabric_q_has_nan_post}")
                print(f"  fabric_qd has NaN: {fabric_qd_has_nan_post}")
            
            # === Transfer control commands back to Genesis device ===
            q_desired = to_genesis(fabric_q, cfg)
            qd_desired = to_genesis(fabric_qd, cfg)
        
        except Exception as e:
            # Capture error and stop simulation immediately
            error_occurred = True
            error_message = str(e)
            error_step = control_step
            print(f"\n{'='*60}")
            print(f"❌ ERROR at control step {control_step} (t={control_step * cfg.control_dt:.3f}s)")
            print(f"{'='*60}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print(f"{'='*60}")
            
            # Print FABRICS state diagnostics at error
            print("\n🔍 FABRICS State at Error:")
            print(f"  fabric_q norm:  {torch.norm(fabric_q).item():.6f}")
            print(f"  fabric_qd norm: {torch.norm(fabric_qd).item():.6f}")
            print(f"  fabric_qdd norm: {torch.norm(fabric_qdd).item():.6f}")
            print(f"  fabric_q max:   {torch.max(torch.abs(fabric_q)).item():.6f}")
            print(f"  fabric_qd max:  {torch.max(torch.abs(fabric_qd)).item():.6f}")
            print(f"  fabric_q has NaN: {torch.isnan(fabric_q).any().item()}")
            print(f"  fabric_qd has NaN: {torch.isnan(fabric_qd).any().item()}")
            print(f"\n  fabric_q [0]:  {fabric_q[0].cpu().numpy()}")
            print(f"  fabric_qd [0]: {fabric_qd[0].cpu().numpy()}")
            print(f"  fabric_qdd [0]: {fabric_qdd[0].cpu().numpy()}")
            
            # Print Genesis state for comparison
            print(f"\n🤖 Genesis State at Error:")
            print(f"  q_genesis:  {q_genesis[0].cpu().numpy()}")
            print(f"  qd_genesis: {qd_genesis[0].cpu().numpy()}")
            
            # Print palm target
            print(f"\n🎯 Palm Target at Error:")
            print(f"  Target position: {palm_target[0, :3].cpu().numpy()}")
            print(f"  Target orientation: {palm_target[0, 3:6].cpu().numpy()}")
            
            # Print recent diagnostics trend
            if len(diagnostics_log['fabric_q_norm']) > 1:
                print(f"\n📊 Recent Diagnostics Trend (last 5 steps):")
                start_idx = max(0, len(diagnostics_log['fabric_q_norm']) - 5)
                print(f"  Step | q_norm  | qd_norm | qdd_norm | q_max   | qd_max")
                print(f"  " + "-"*60)
                for i in range(start_idx, len(diagnostics_log['fabric_q_norm'])):
                    step_num = i
                    print(f"  {step_num:4d} | {diagnostics_log['fabric_q_norm'][i]:7.4f} | "
                          f"{diagnostics_log['fabric_qd_norm'][i]:7.4f} | "
                          f"{diagnostics_log['fabric_qdd_norm'][i]:8.4f} | "
                          f"{diagnostics_log['fabric_q_max'][i]:7.4f} | "
                          f"{diagnostics_log['fabric_qd_max'][i]:7.4f}")
            
            print(f"{'='*60}")
            print("Stopping simulation and saving all data for debugging...")
            print(f"{'='*60}\n")
            
            # Exit the control loop immediately
            break
        
        # === Send PD control commands to Genesis ===
        # Choose one of the following control modes (uncomment ONE):
        
        # ===== Case 1: Use FABRICS position output with zero velocity =====
        q_cmd = q_desired
        qd_cmd = torch.zeros_like(q_desired)
        robot.command_full_state(
            arm_position=q_cmd[:, :6],
            arm_velocity=qd_cmd[:, :6],
            gripper_position=q_cmd[:, 6:],
            gripper_velocity=qd_cmd[:, 6:]
        )
        
        # ===== Case 2: Use FABRICS position output with velocity =====
        # q_cmd = q_desired
        # qd_cmd = qd_desired
        # robot.command_full_state(
        #     arm_position=q_cmd[:, :6],
        #     arm_velocity=qd_cmd[:, :6],
        #     gripper_position=q_cmd[:, 6:],
        #     gripper_velocity=qd_cmd[:, 6:]
        # )
        
        # ===== Case 3: Use pre-defined sinusoidal trajectory (oscillating joints) =====
        # current_time = control_step * cfg.control_dt
        # q_cmd = q_initial.clone()
        # # Joint 0 (shoulder_pan): oscillate ±30° around initial, period 4s
        # q_cmd[:, 0] = q_initial[:, 0] + 0.52 * torch.sin(torch.tensor(2 * np.pi * current_time / 4.0, device=cfg.genesis_device))
        # # Joint 1 (shoulder_lift): oscillate ±20° around initial, period 3s
        # q_cmd[:, 1] = q_initial[:, 1] + 0.35 * torch.sin(torch.tensor(2 * np.pi * current_time / 3.0, device=cfg.genesis_device))
        # # Joint 2 (elbow): oscillate ±25° around initial, period 5s
        # q_cmd[:, 2] = q_initial[:, 2] + 0.44 * torch.sin(torch.tensor(2 * np.pi * current_time / 5.0, device=cfg.genesis_device))
        # # Joint 5 (wrist_3): oscillate ±45° around initial, period 2s (fast rotation)
        # q_cmd[:, 5] = q_initial[:, 5] + 0.79 * torch.sin(torch.tensor(2 * np.pi * current_time / 2.0, device=cfg.genesis_device))
        # qd_cmd = torch.zeros_like(q_cmd)  # Zero velocity command
        # robot.command_full_state(
        #     arm_position=q_cmd[:, :6],
        #     arm_velocity=qd_cmd[:, :6],
        #     gripper_position=q_cmd[:, 6:],
        #     gripper_velocity=qd_cmd[:, 6:]
        # )
        
        # ===== Case 4: Use fixed target position (hold initial pose) =====
        # q_cmd = q_fixed_target
        # qd_cmd = qd_fixed_target
        # robot.command_full_state(
        #     arm_position=q_cmd[:, :6],
        #     arm_velocity=qd_cmd[:, :6],
        #     gripper_position=q_cmd[:, 6:],
        #     gripper_velocity=qd_cmd[:, 6:]
        # )
        
        # === Step Genesis simulation once (60 Hz) ===
        scene.step()
        
        # Render camera frame
        if cfg.save_video:
            camera.render()
        
        # === Update visualization markers ===
        if cfg.show_palm_target:
            update_palm_target_marker(palm_target_marker, palm_target, cfg)
        
        if cfg.show_finger_forces:
            update_finger_force_markers(
                finger_force_markers, 
                robot.finger_tips_pose,  # [B, 3, 7]
                finger_forces, 
                cfg
            )
        
        # Note: fabric_q, fabric_qd, fabric_qdd are already updated in-place above
        # No need to manually update them here
        
        # === Log control data ===
        if cfg.save_control_data or cfg.plot_control_data:
            log_data['times'].append(control_step * cfg.control_dt)
            log_data['q_actual'].append(q_genesis[0].cpu().numpy())
            log_data['q_desired'].append(q_cmd[0].cpu().numpy())  # PD command sent
            log_data['qd_actual'].append(qd_genesis[0].cpu().numpy())
            log_data['qd_desired'].append(qd_cmd[0].cpu().numpy())  # PD command sent
            log_data['q_fabrics'].append(q_desired[0].cpu().numpy())  # FABRICS output
            log_data['qd_fabrics'].append(qd_desired[0].cpu().numpy())  # FABRICS output
            # Compute qdd from FABRICS (transfer to Genesis device for consistency)
            qdd_desired = to_genesis(fabric_qdd, cfg)
            log_data['qdd_fabrics'].append(qdd_desired[0].cpu().numpy())  # FABRICS output
            log_data['palm_targets'].append(palm_target[0].cpu().numpy())
        
        # === Palm target tracking (log every 1 second) ===
        if control_step % 60 == 0 and not cfg.test_mode_simple:
            current_time = control_step * cfg.control_dt
            current_palm_pose = robot.palm_pose
            print(f"\n[Palm Tracking] Time {current_time:.2f}s")
            print(f"  Current palm pos: {current_palm_pose[0, :3].cpu().numpy()}")
            print(f"  Target palm pos:  {palm_target[0, :3].cpu().numpy()}")
            print(f"  Position error:   {torch.norm(to_fabrics(current_palm_pose[0, :3], cfg) - palm_target[0, :3]).item():.4f} m")
        
        # === Logging (every 1 second at 60 Hz) ===
        if control_step % 60 == 0:
            elapsed = time.time() - start_time
            sim_time = control_step * cfg.control_dt
            # Compute errors against the actual command sent
            pos_error = torch.norm(q_genesis - q_cmd).item()
            vel_error = torch.norm(qd_genesis - qd_cmd).item()
            print(f"Control {control_step:4d}/{num_control_steps} | "
                  f"Time: {sim_time:5.2f}s | "
                  f"Wall: {elapsed:5.2f}s | "
                  f"q_err: {pos_error:.4f} | qd_err: {vel_error:.4f}")
    
    # Save video (even if error occurred)
    if cfg.save_video:
        if error_occurred:
            print(f"\nSaving partial video to {video_path}...")
        else:
            print(f"\nSaving video to {video_path}...")
        try:
            camera.stop_recording(save_to_filename=video_path)
            if error_occurred:
                print(f"✅ Partial video saved to: {video_path} (up to step {error_step})")
            else:
                print(f"✅ Video saved to: {video_path}")
        except Exception as e:
            print(f"⚠️  Failed to save video: {e}")
    
    # Process and save control data
    if cfg.save_control_data or cfg.plot_control_data:
        if error_occurred:
            print(f"\nProcessing partial control data (up to step {error_step})...")
        else:
            print("\nProcessing control data...")
        
        # Check if we have any data
        if len(log_data['times']) == 0:
            print("⚠️  No data logged, skipping data processing")
        else:
            try:
                # Convert lists to numpy arrays
                for key in log_data:
                    log_data[key] = np.array(log_data[key])
                
                # Save raw data
                if cfg.save_control_data:
                    data_path = os.path.join(output_dir, "control_data.npz")
                    np.savez(data_path, **log_data)
                    if error_occurred:
                        print(f"✅ Partial control data saved to: {data_path}")
                    else:
                        print(f"✅ Control data saved to: {data_path}")
                
                # Save integration trace to CSV for offline reproduction
                if len(integration_trace['control_step']) > 0:
                    print("\nSaving detailed integration trace to CSV...")
                    csv_path = os.path.join(output_dir, "integration_trace.csv")
                    
                    try:
                        import csv
                        with open(csv_path, 'w', newline='') as csvfile:
                            # Header
                            fieldnames = [
                                'control_step', 'decimation_step', 'timestamp', 'dt',
                            ]
                            # Add q, qd, qdd columns (18 joints each)
                            for i in range(18):
                                fieldnames.extend([f'q_before_{i}', f'qd_before_{i}', f'qdd_before_{i}'])
                            for i in range(18):
                                fieldnames.extend([f'q_after_{i}', f'qd_after_{i}', f'qdd_after_{i}'])
                            # Add palm target (3 pos + 3 ori)
                            for i in range(3):
                                fieldnames.append(f'palm_pos_{i}')
                            for i in range(3):
                                fieldnames.append(f'palm_ori_{i}')
                            # Add finger forces (3 fingers x 3 components = 9)
                            for i in range(9):
                                fieldnames.append(f'finger_force_{i}')
                            
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()
                            
                            # Write data rows
                            num_rows = len(integration_trace['control_step'])
                            for idx in range(num_rows):
                                row = {
                                    'control_step': integration_trace['control_step'][idx],
                                    'decimation_step': integration_trace['decimation_step'][idx],
                                    'timestamp': integration_trace['timestamp'][idx],
                                    'dt': integration_trace['dt'][idx],
                                }
                                
                                # Add state before integration
                                q_before = integration_trace['q_before'][idx]
                                qd_before = integration_trace['qd_before'][idx]
                                qdd_before = integration_trace['qdd_before'][idx]
                                for i in range(18):
                                    row[f'q_before_{i}'] = q_before[i]
                                    row[f'qd_before_{i}'] = qd_before[i]
                                    row[f'qdd_before_{i}'] = qdd_before[i]
                                
                                # Add state after integration
                                q_after = integration_trace['q_after'][idx]
                                qd_after = integration_trace['qd_after'][idx]
                                qdd_after = integration_trace['qdd_after'][idx]
                                for i in range(18):
                                    row[f'q_after_{i}'] = q_after[i]
                                    row[f'qd_after_{i}'] = qd_after[i]
                                    row[f'qdd_after_{i}'] = qdd_after[i]
                                
                                # Add palm target
                                palm_pos = integration_trace['palm_target_pos'][idx]
                                palm_ori = integration_trace['palm_target_ori'][idx]
                                for i in range(3):
                                    row[f'palm_pos_{i}'] = palm_pos[i]
                                    row[f'palm_ori_{i}'] = palm_ori[i]
                                
                                # Add finger forces
                                forces = integration_trace['finger_forces'][idx]
                                for i in range(9):
                                    row[f'finger_force_{i}'] = forces[i]
                                
                                writer.writerow(row)
                        
                        print(f"✅ Integration trace saved to: {csv_path}")
                        print(f"   Total integration steps: {num_rows}")
                        print(f"   Columns: {len(fieldnames)}")
                        
                    except Exception as e:
                        print(f"⚠️  Failed to save integration trace CSV: {e}")
                
                # Generate plots
                if cfg.plot_control_data:
                    plot_path = os.path.join(output_dir, "control_plot.png")
                    plot_control_data(log_data, plot_path)
                    
                    # Print statistics
                    if error_occurred:
                        print(f"\n=== Control Statistics (up to error at step {error_step}) ===")
                    else:
                        print("\n=== Control Statistics ===")
                    q_errors = np.abs(log_data['q_actual'] - log_data['q_desired'])
                    qd_errors = np.abs(log_data['qd_actual'] - log_data['qd_desired'])
                    print(f"Position tracking error (mean): {q_errors.mean():.6f} rad")
                    print(f"Position tracking error (max):  {q_errors.max():.6f} rad")
                    print(f"Velocity tracking error (mean): {qd_errors.mean():.6f} rad/s")
                    print(f"Velocity tracking error (max):  {qd_errors.max():.6f} rad/s")
                    print(f"Arm position error (mean):      {q_errors[:, :6].mean():.6f} rad")
                    print(f"Gripper position error (mean):  {q_errors[:, 6:].mean():.6f} rad")
            
            except Exception as e:
                print(f"⚠️  Error during data processing: {e}")
                print("Attempting to save raw log data...")
                try:
                    import pickle
                    raw_data_path = os.path.join(output_dir, "control_data_raw.pkl")
                    with open(raw_data_path, 'wb') as f:
                        pickle.dump(log_data, f)
                    print(f"✅ Raw log data saved to: {raw_data_path}")
                except Exception as e2:
                    print(f"❌ Failed to save raw data: {e2}")
    
    # Print error summary if error occurred
    if error_occurred:
        print("\n" + "="*60)
        print("❌ ERROR SUMMARY")
        print("="*60)
        print(f"Error occurred at step: {error_step}")
        print(f"Error time: {error_step * cfg.control_dt:.3f}s")
        print(f"Total steps planned: {num_control_steps}")
        print(f"Steps completed: {error_step}")
        print(f"Completion: {error_step/num_control_steps*100:.1f}%")
        print(f"\nError message: {error_message}")
        print("="*60)
        
        # Save error report
        error_report_path = os.path.join(output_dir, "error_report.txt")
        try:
            with open(error_report_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("FABRICS ERROR REPORT\n")
                f.write("="*60 + "\n")
                f.write(f"Error occurred at step: {error_step}\n")
                f.write(f"Error time: {error_step * cfg.control_dt:.3f}s\n")
                f.write(f"Total steps planned: {num_control_steps}\n")
                f.write(f"Completion: {error_step/num_control_steps*100:.1f}%\n")
                f.write(f"\nError message:\n{error_message}\n")
                f.write("\n" + "="*60 + "\n")
                f.write("Configuration:\n")
                f.write(f"  fabric_decimation: {cfg.fabric_decimation}\n")
                f.write(f"  fabrics_dt: {cfg.fabrics_dt:.6f}s ({1/cfg.fabrics_dt:.1f} Hz)\n")
                f.write(f"  sim_dt: {cfg.sim_dt:.6f}s ({1/cfg.sim_dt:.1f} Hz)\n")
                f.write(f"  control_dt: {cfg.control_dt:.6f}s ({1/cfg.control_dt:.1f} Hz)\n")
                f.write(f"  num_envs: {cfg.num_envs}\n")
                f.write(f"  test_mode_simple: {cfg.test_mode_simple}\n")
                f.write("\n" + "="*60 + "\n")
                f.write("State at error (if available in log_data):\n")
                if len(log_data['times']) > 0:
                    # Get last logged state
                    last_idx = min(error_step, len(log_data['times']) - 1)
                    f.write(f"  Last logged step: {last_idx}\n")
                    f.write(f"  Last logged time: {log_data['times'][last_idx]:.3f}s\n")
                    if last_idx < len(log_data['q_actual']):
                        f.write(f"  q_actual norm: {np.linalg.norm(log_data['q_actual'][last_idx]):.6f}\n")
                        f.write(f"  qd_actual norm: {np.linalg.norm(log_data['qd_actual'][last_idx]):.6f}\n")
                        f.write(f"  q_fabrics norm: {np.linalg.norm(log_data['q_fabrics'][last_idx]):.6f}\n")
                        f.write(f"  qd_fabrics norm: {np.linalg.norm(log_data['qd_fabrics'][last_idx]):.6f}\n")
                else:
                    f.write("  No state logged before error\n")
                f.write("\n" + "="*60 + "\n")
                f.write("FABRICS State Diagnostics at Error:\n")
                f.write(f"  fabric_q norm:  {diagnostics_log['fabric_q_norm'][error_step] if error_step < len(diagnostics_log['fabric_q_norm']) else 'N/A'}\n")
                f.write(f"  fabric_qd norm: {diagnostics_log['fabric_qd_norm'][error_step] if error_step < len(diagnostics_log['fabric_qd_norm']) else 'N/A'}\n")
                f.write(f"  fabric_qdd norm: {diagnostics_log['fabric_qdd_norm'][error_step] if error_step < len(diagnostics_log['fabric_qdd_norm']) else 'N/A'}\n")
                f.write(f"  fabric_q max:   {diagnostics_log['fabric_q_max'][error_step] if error_step < len(diagnostics_log['fabric_q_max']) else 'N/A'}\n")
                f.write(f"  fabric_qd max:  {diagnostics_log['fabric_qd_max'][error_step] if error_step < len(diagnostics_log['fabric_qd_max']) else 'N/A'}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("Diagnostics Trend (all steps):\n")
                f.write("Step | q_norm  | qd_norm | qdd_norm | q_max   | qd_max  | NaN?\n")
                f.write("-" * 70 + "\n")
                for i in range(len(diagnostics_log['fabric_q_norm'])):
                    nan_flag = "⚠️ NaN" if (diagnostics_log['fabric_q_has_nan'][i] or diagnostics_log['fabric_qd_has_nan'][i]) else ""
                    f.write(f"{i:4d} | {diagnostics_log['fabric_q_norm'][i]:7.4f} | "
                           f"{diagnostics_log['fabric_qd_norm'][i]:7.4f} | "
                           f"{diagnostics_log['fabric_qdd_norm'][i]:8.4f} | "
                           f"{diagnostics_log['fabric_q_max'][i]:7.4f} | "
                           f"{diagnostics_log['fabric_qd_max'][i]:7.4f} | {nan_flag}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("Debugging suggestions:\n")
                f.write("  1. Check diagnostics trend above for explosive growth\n")
                f.write("  2. If q_norm/qd_norm grows rapidly → reduce fabric_decimation\n")
                f.write("  3. If NaN appears → check initial configuration validity\n")
                f.write("  4. Check control_plot*.png for trajectory anomalies\n")
                f.write("  5. Review video to see robot behavior before error\n")
                f.write("  6. Try increasing damping in cs63_tesollo_params.yaml\n")
                f.write("  7. Try reducing palm_target offsets\n")
                f.write("="*60 + "\n")
            print(f"✅ Error report saved to: {error_report_path}")
        except Exception as e:
            print(f"⚠️  Failed to save error report: {e}")
    
    elapsed_total = time.time() - start_time
    
    if error_occurred:
        print("\n" + "="*60)
        print("❌ Simulation stopped due to error")
        print("="*60)
        print(f"Elapsed time: {elapsed_total:.2f}s")
        print(f"Simulated time: {error_step * cfg.control_dt:.2f}s / {cfg.total_time:.2f}s")
        print(f"Real-time factor: {(error_step * cfg.control_dt)/elapsed_total:.2f}x")
        print("\n📁 Saved files for debugging:")
        print(f"  - Video: simulation.mp4 (partial, up to error)")
        print(f"  - Data: control_data.npz (partial)")
        print(f"  - Plots: control_plot*.png")
        print(f"  - Error report: error_report.txt")
        
        # Quick diagnosis based on error pattern
        print(f"\n🔧 Quick Diagnosis:")
        if error_step < 20:
            print(f"  ⚠️  Error occurred very early (step {error_step})")
            print(f"  → Likely cause: Initial configuration or FABRICS setup issue")
            print(f"  → Try: Check initial_joint_config.yaml for validity")
            print(f"  → Try: Verify all FABRICS modules are properly enabled")
        
        if len(diagnostics_log['fabric_q_norm']) >= 2:
            q_growth = diagnostics_log['fabric_q_norm'][-1] / (diagnostics_log['fabric_q_norm'][0] + 1e-6)
            qd_growth = diagnostics_log['fabric_qd_norm'][-1] / (diagnostics_log['fabric_qd_norm'][0] + 1e-6)
            
            if q_growth > 2.0 or qd_growth > 2.0:
                print(f"  ⚠️  State explosion detected:")
                print(f"     q_norm growth: {q_growth:.2f}x")
                print(f"     qd_norm growth: {qd_growth:.2f}x")
                print(f"  → Likely cause: Numerical instability in integration")
                print(f"  → Try: Reduce fabric_decimation from {cfg.fabric_decimation} to 1")
                print(f"  → Try: Increase damping in cs63_tesollo_params.yaml")
        
        if any(diagnostics_log['fabric_q_has_nan']) or any(diagnostics_log['fabric_qd_has_nan']):
            nan_step = next((i for i, (q, qd) in enumerate(zip(diagnostics_log['fabric_q_has_nan'], 
                                                               diagnostics_log['fabric_qd_has_nan'])) 
                            if q or qd), -1)
            print(f"  ⚠️  NaN detected at step {nan_step}")
            print(f"  → Likely cause: Division by zero or invalid operation")
            print(f"  → Try: Check palm_target offsets are reasonable")
            print(f"  → Try: Verify joint limits are not violated")
        
        if "singular" in error_message.lower():
            print(f"  ⚠️  Matrix singularity (cannot invert metric)")
            print(f"  → Likely cause: Missing geometric fabrics for some joints")
            print(f"  → Try: Enable cspace_energy in cs63_tesollo_fabric.py")
            print(f"  → Try: Check that palm_pose attractor covers all arm joints")
        
        print(f"\n💡 Check error_report.txt for detailed diagnostics and full trend data")
        print("="*60)
    else:
        print("\n✅ Simulation complete!")
        print(f"Total time: {elapsed_total:.2f}s for {cfg.total_time:.2f}s simulation")
        print(f"Real-time factor: {cfg.total_time/elapsed_total:.2f}x")


if __name__ == "__main__":
    main()
