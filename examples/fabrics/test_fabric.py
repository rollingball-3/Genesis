
import os
import time
from datetime import datetime
import numpy as np
import torch
from pathlib import Path
import genesis as gs

# Import quat_to_R explicitly to avoid any import issues
from genesis.utils.geom import quat_to_R

# Import utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_initial_joint_config

from throw_env import Manipulator
from fabrics_sim.fabrics.cs63_tesollo_fabric import CS63TesolloFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

class Config:
    total_time = 10.0
    genesis_device = "cuda:0"
    sim_dt = 1.0 / 60.0
    fabrics_device = "cuda:1"
    fabrics_decimation = 2
    fabrics_dt = sim_dt / fabrics_decimation
    num_envs = 1
    fabric_decimation = 5
    
    base_height = 0.6

def to_fabrics(tensor: torch.Tensor, cfg) -> torch.Tensor:
    """Transfer tensor from Genesis device to FABRICS device"""
    return tensor.to(cfg.fabrics_device)

def to_genesis(tensor: torch.Tensor, cfg) -> torch.Tensor:
    """Transfer tensor from FABRICS device to Genesis device"""
    return tensor.to(cfg.genesis_device)

def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"tmp/fabrics_test/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    cfg = Config()
    
    ################################## Genesis Config ##################################
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=cfg.sim_dt, substeps=2),
        show_viewer=False,
    )
    # Add ground
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    # Add manipulator
    manipulator = Manipulator(scene=scene, num_envs=1, device=cfg.genesis_device, base_height=0.6)
    # Add camera
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(1.5, 0.0, 1.2),
        lookat=(0.0, 0.0, 0.5),
        fov=40,
        GUI=False,
    )
    # Build scene
    scene.build(n_envs=1)
    manipulator.set_pd_gains()
    
    ################################## Fabrics Config ##################################
    fabrics_device_int = int(cfg.fabrics_device.replace("cuda:", ""))
    initialize_warp(str(fabrics_device_int))
    world_model = WorldMeshesModel(
        batch_size=1,
        max_objects_per_env=20,
        device=cfg.fabrics_device,
        world_filename='floor'
    )
    object_ids, object_indicator = world_model.get_object_ids()
    
    # Create CS63-Tesollo fabric
    cs63_fabric = CS63TesolloFabric(
        batch_size=1,
        device=cfg.fabrics_device,
        timestep=cfg.fabrics_dt,
        num_arm_joints=6,
        num_gripper_joints=12,
        num_fingers=3,
        graph_capturable=False
    )
    
    # Create integrator
    cs63_integrator = DisplacementIntegrator(cs63_fabric)
    
    # start simulation
    manipulator.reset()
    # Load initial configuration
    initial_config = load_initial_joint_config()
    q_initial = torch.tensor(initial_config, device=cfg.genesis_device).unsqueeze(0)
    
    # Now read the actual palm pose at this configuration
    current_palm_pose_world = manipulator.palm_pose  # [B, 7] in world frame
    current_palm_pose_base = current_palm_pose_world.clone()
    current_palm_pose_base[:,2] -= cfg.base_height
    
    # Initialize FABRICS state with initial configuration
    fabric_q = to_fabrics(q_initial.clone(), cfg)
    fabric_qd = torch.zeros(1, 18, device=cfg.fabrics_device)
    fabric_qdd = torch.zeros(1, 18, device=cfg.fabrics_device)
    
    # Set initial position and orientation from actual palm pose (Genesis base frame)
    initial_pos_base = current_palm_pose_base[:, :3]                # on Genesis device
    initial_quat = current_palm_pose_base[:, 3:7]                   # on Genesis device

    palm_target_position = to_fabrics(initial_pos_base, cfg)       # FABRICS device
    initial_rotmat = quat_to_R(initial_quat)         # still on Genesis device
    palm_target_matrix = to_fabrics(initial_rotmat, cfg)     # FABRICS device
    
    # Initialize finger forces
    finger_forces = torch.zeros(1, 3, 3, device=cfg.fabrics_device)
    
    # Start video recording
    video_path = os.path.join(output_dir, f"1.mp4")
    print(f"Recording video to: {video_path}")
    camera.start_recording()
    
    num_control_steps = int(cfg.total_time / cfg.sim_dt)
    
    for control_step in range(num_control_steps):
        current_time = control_step * cfg.sim_dt
        
        # Create time-varying finger forces (sine wave pattern)
        force_magnitude = torch.sin(torch.tensor(current_time))
        finger_forces = torch.zeros(cfg.num_envs, 3, 3, device=cfg.fabrics_device)
        # Set forces in the y-direction of each finger's local frame
        finger_forces[:, :, 2] = force_magnitude
        
        # Add smooth oscillations to palm desired pose
        # Create time-varying parameter with very low frequency (0.2Hz) for extremely slow changes
        time_param = current_time * 1
        
        # Create rotation around x and y axes (smooth changes)
        rot_x = np.sin(time_param) * 0.2
        rot_y = np.sin(time_param) * 0.2
        rot_z = np.sin(time_param) * 0.2
        
        # Create rotation matrix from small angles using Rodrigues' formula
        from scipy.spatial.transform import Rotation as R
        small_rot = R.from_euler('xyz', [rot_x, rot_y, rot_z]).as_matrix()
        
        # Apply small rotation relative to initial rotation (not cumulative)
        initial_rotmat_np = initial_rotmat[0].cpu().numpy()
        new_rotmat = small_rot @ initial_rotmat_np
        # new_rotmat = initial_rotmat_np
        # Update target matrix
        palm_target_matrix[0] = torch.tensor(new_rotmat, device=cfg.fabrics_device)
        
        # Add position changes to palm target (oscillate around initial position)
        pos_x = (np.cos(time_param) - 1) * 0.1  # 10cm oscillation in x
        pos_y = np.sin(time_param) * 0.1  # 10cm oscillation in y
        pos_z = np.sin(time_param * 0.7) * 0.02  # 5cm oscillation in z with different frequency
        
        # Create position offset vector and ensure it's on the same device as initial_pos_base
        pos_offset = torch.tensor([pos_x, pos_y, pos_z], device=initial_pos_base.device)
        
        # Set target position relative to initial position (not cumulative)
        new_pos = initial_pos_base[0] + pos_offset
        # new_pos = initial_pos_base[0]
        palm_target_position[0] = to_fabrics(new_pos, cfg)
        
        # FABRICS set target
        cs63_fabric.set_features(
            finger_forces, palm_target_position, palm_target_matrix,
            fabric_q.detach(), fabric_qd.detach(),
            object_ids, object_indicator
        )
        
        # Clear all debug objects first to prevent persistence
        if scene.visualizer:
            scene.clear_debug_objects()
        
        # Draw palm target frame
        if scene.visualizer:
            # Convert palm target to world frame
            palm_pos = palm_target_position.cpu().numpy()[0] + np.array([0, 0, cfg.base_height])
            palm_rotmat = palm_target_matrix.cpu().numpy()[0]
            
            # Create transformation matrix for the frame
            T = np.eye(4)
            T[:3, :3] = palm_rotmat
            T[:3, 3] = palm_pos
            
            # Draw the debug frame
            scene.draw_debug_frame(T, axis_length=0.1, origin_size=0.01, axis_radius=0.005)
        
        # Visualize force frames
        # Get finger force frames pose in world frame
        force_frames_pose = manipulator.finger_force_frames_pose  # [B, 3, 7]
        
        # Draw the force frames' coordinate systems
        for finger_idx in range(3):
            # Get position and quaternion for this finger's force frame
            pos = force_frames_pose[0, finger_idx, :3]
            quat = force_frames_pose[0, finger_idx, 3:7]
            
            # Convert quaternion to rotation matrix
            rotmat = quat_to_R(quat)
            
            # Create transformation matrix for the force frame
            T = np.eye(4)
            T[:3, :3] = rotmat.cpu().numpy()
            T[:3, 3] = pos.cpu().numpy()
            
            # Draw the coordinate system for this force frame
            if scene.visualizer:
                scene.draw_debug_frame(
                    T,
                    axis_length=0.05,  # Shorter axes for force frames
                    origin_size=0.005,
                    axis_radius=0.003
                )
        
        # Visualize gripper_base_link (palm_link) coordinate system
        if scene.visualizer:
            # Get palm link pose
            palm_pose = manipulator.palm_pose.cpu().numpy()[0]
            palm_pos = palm_pose[:3]
            palm_quat = palm_pose[3:7]
            
            # Convert quaternion to rotation matrix
            palm_rotmat = quat_to_R(palm_quat)
            
            # Create transformation matrix for palm link
            T_palm = np.eye(4)
            T_palm[:3, :3] = palm_rotmat
            T_palm[:3, 3] = palm_pos
            
            # Draw palm link coordinate system
            scene.draw_debug_frame(
                T_palm,
                axis_length=0.1,  # Same as palm target frame
                origin_size=0.01,
                axis_radius=0.005
            )
        
        # Visualize world coordinate system
        if scene.visualizer:
            # World coordinate system is at origin (0,0,0) with standard axes
            T_world = np.eye(4)
            
            # Draw world coordinate system
            scene.draw_debug_frame(
                T_world,
                axis_length=0.2,  # Larger axes for world frame
                origin_size=0.02,
                axis_radius=0.007
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
        manipulator.command_arm_position_velocity(
            position=q_cmd[:, :6],
            velocity=qd_cmd[:, :6],
        )
        manipulator.command_gripper_position_velocity(
            position=q_cmd[:, 6:],
            velocity=qd_cmd[:, 6:],
        )
        # Step simulation
        scene.step()
        camera.render()
        
    # Stop recording
    camera.stop_recording(save_to_filename=video_path)
    print(f"✅ Video saved: {video_path}")
    
if __name__ == "__main__":
    main()
