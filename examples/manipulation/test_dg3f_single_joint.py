"""
Test script for DG3F gripper single joint oscillation.

Tests all 12 joints in parallel using a 3x4 grid layout.
One camera captures all grippers in a single video.

Usage:
    python test_dg3f_single_joint.py
"""
import os
import numpy as np
import torch
from datetime import datetime

import genesis as gs


# PD control parameters
GRIPPER_KP = 120
GRIPPER_KV = 10.0
GRIPPER_FORCE_LIMIT = 10.0

# Grid layout parameters
GRID_ROWS = 3      # 3 rows (M1, M2, M3)
GRID_COLS = 4      # 4 columns (M1-M4)
SPACING_X = 0.35   # Spacing between grippers in X direction (increased from 0.15)
SPACING_Y = 0.35   # Spacing between grippers in Y direction (increased from 0.15)
BASE_HEIGHT = 0.09 # Height of gripper base

# Test configuration: (joint_index, joint_name, oscillation_range, oscillation_center, row, col)
TEST_CONFIGS = [
    # Row 0: M1 joints (finger spread) - F1M1, F2M1, F3M1, placeholder
    (0, "F1M1", 0.5, 0.0, 0, 0),
    (4, "F2M1", 0.5, 0.0, 0, 1),
    (8, "F3M1", 0.5, 0.0, 0, 2),
    # Row 1: M2 joints (finger base flexion) - F1M2, F2M2, F3M2, placeholder
    (1, "F1M2", 0.8, 0.0, 1, 0),
    (5, "F2M2", 0.8, 0.0, 1, 1),
    (9, "F3M2", 0.8, 0.0, 1, 2),
    # Row 2: M3 joints (main finger bend) - F1M3, F2M3, F3M3, placeholder
    (2, "F1M3", 1.2, 0.5, 2, 0),
    (6, "F2M3", 1.2, 0.5, 2, 1),
    (10, "F3M3", 1.2, 0.5, 2, 2),
    # Row 3: M4 joints (finger tip) - F1M4, F2M4, F3M4, placeholder
    (3, "F1M4", 0.6, 0.3, 3, 0),
    (7, "F2M4", 0.6, 0.3, 3, 1),
    (11, "F3M4", 0.6, 0.3, 3, 2),
]


def disable_collision_between_links(entity, link_name_a, link_name_b):
    """Disable collision between two links by modifying collision pair validity."""
    solver = entity.solver
    link_a = entity.get_link(link_name_a)
    link_b = entity.get_link(link_name_b)
    
    # Get geom indices for each link
    geoms_link_idx = solver.geoms_info.link_idx.to_numpy()
    
    for i_ga in range(solver.n_geoms):
        for i_gb in range(i_ga + 1, solver.n_geoms):
            i_la = geoms_link_idx[i_ga]
            i_lb = geoms_link_idx[i_gb]
            
            # Check if this pair matches our target links
            if (i_la == link_a.idx and i_lb == link_b.idx) or \
               (i_la == link_b.idx and i_lb == link_a.idx):
                # Disable this collision pair
                solver.collider._collider_info.collision_pair_validity[i_ga, i_gb] = 0


def main():
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"tmp/dg3f_single_joint_test/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    n_tests = len(TEST_CONFIGS)
    print(f"\n{'='*60}")
    print(f"DG3F Single Joint Oscillation Test")
    print(f"Testing {n_tests} joints in 3x4 grid layout")
    print(f"PD gains: kp={GRIPPER_KP}, kv={GRIPPER_KV}, force_limit=±{GRIPPER_FORCE_LIMIT}")
    print(f"{'='*60}")
    
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Calculate grid center for camera positioning
    grid_center_x = (GRID_COLS - 1) * SPACING_X / 2
    grid_center_y = (GRID_ROWS - 1) * SPACING_Y / 2
    
    # Create scene with wide-angle camera
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(grid_center_x + 0.8, grid_center_y - 1.5, 1.2),
            camera_lookat=(grid_center_x, grid_center_y, BASE_HEIGHT),
            camera_fov=70,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=10,
        ),
        show_viewer=False,
    )
    
    # Add ground
    scene.add_entity(gs.morphs.Plane())
    
    # Add ONE gripper template - will be replicated in grid layout
    gripper = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/DG3F/urdf/delto_gripper_3f.urdf",
            merge_fixed_links=False,
            fixed=True,
            pos=(0.0, 0.0, BASE_HEIGHT),  # Base position, will be offset by env_pos
            decompose_robot_error_threshold=0.15,
        ),
        vis_mode="collision",
    )
    
    # Add ONE camera to capture all grippers
    camera = scene.add_camera(
        res=(1280, 720),  # High resolution for grid view
        pos=(grid_center_x + 0.8, grid_center_y - 1.5, 1.2),
        lookat=(grid_center_x, grid_center_y, BASE_HEIGHT),
        fov=70,
        GUI=False,
    )
    
    # Build scene with n_tests environments in grid layout
    scene.build(
        n_envs=n_tests,
        env_spacing=(SPACING_X, SPACING_Y),  # Spacing between environments
        n_envs_per_row=GRID_COLS,  # 4 environments per row
        center_envs_at_origin=False,  # Don't center, keep grid aligned
    )
    
    print(f"✅ Grid layout: {GRID_ROWS} rows × {GRID_COLS} cols")
    print(f"   Spacing: {SPACING_X}m × {SPACING_Y}m")
    
    # Disable collision
    for finger in ["F1", "F2", "F3"]:
        disable_collision_between_links(gripper, f"{finger}_02", "delto_base_link")
    print("✅ Collision filtering applied")
    
    # Get joint indices
    device = "cuda:0"
    gripper_joints_name = (
        "F1M1", "F1M2", "F1M3", "F1M4",
        "F2M1", "F2M2", "F2M3", "F2M4",
        "F3M1", "F3M2", "F3M3", "F3M4",
    )
    gripper_dof_idx = torch.tensor(
        [gripper.get_joint(name).dofs_idx_local[0] for name in gripper_joints_name],
        device=device
    )
    
    # Set PD gains
    gripper_kp = torch.tensor([GRIPPER_KP] * 12, device=device)
    gripper_kv = torch.tensor([GRIPPER_KV] * 12, device=device)
    gripper_force_min = torch.tensor([-GRIPPER_FORCE_LIMIT] * 12, device=device)
    gripper_force_max = torch.tensor([GRIPPER_FORCE_LIMIT] * 12, device=device)
    
    gripper.set_dofs_kp(gripper_kp, gripper_dof_idx)
    gripper.set_dofs_kv(gripper_kv, gripper_dof_idx)
    gripper.set_dofs_force_range(gripper_force_min, gripper_force_max, gripper_dof_idx)
    
    # Prepare data recording
    time_steps = []
    target_angles_all = [[] for _ in range(n_tests)]
    actual_angles_all = [[] for _ in range(n_tests)]
    
    # Start recording
    camera.start_recording()
    
    # Oscillation parameters
    oscillation_period = 50
    total_cycles = 3
    total_steps = oscillation_period * total_cycles * 2
    
    print(f"\nRunning {total_steps} steps...")
    
    for step in range(total_steps):
        # Create target positions for all environments [n_envs, 12]
        targets = torch.zeros(n_tests, 12, device=device)
        
        for env_idx, (joint_idx, joint_name, osc_range, osc_center, row, col) in enumerate(TEST_CONFIGS):
            # Calculate target angle
            phase = (step / oscillation_period) * np.pi
            target_angle = osc_center + osc_range * np.sin(phase)
            
            # Set target for this environment's test joint
            targets[env_idx, joint_idx] = target_angle
            
            # Record target
            target_angles_all[env_idx].append(target_angle)
        
        # Apply PD control
        gripper.control_dofs_position(
            position=targets,
            dofs_idx_local=gripper_dof_idx
        )
        
        # Step simulation
        scene.step()
        
        # Render camera
        camera.render()
        
        # Record actual positions
        actual_qpos_all = gripper.get_dofs_position(gripper_dof_idx)  # [n_envs, 12]
        for env_idx, (joint_idx, _, _, _, _, _) in enumerate(TEST_CONFIGS):
            actual_angles_all[env_idx].append(actual_qpos_all[env_idx, joint_idx].cpu().item())
        
        # Record time
        time_steps.append(step)
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step:4d}/{total_steps}")
    
    # Stop recording and save video
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    video_path = os.path.join(output_dir, "all_joints_grid.mp4")
    camera.stop_recording(save_to_filename=video_path)
    print(f"✅ Video saved: {video_path}")
    
    # Process results
    results = []
    for env_idx, (joint_idx, joint_name, osc_range, osc_center, row, col) in enumerate(TEST_CONFIGS):
        
        
        # Calculate statistics
        error_array = np.abs(np.array(target_angles_all[env_idx]) - np.array(actual_angles_all[env_idx]))
        mean_error = np.degrees(np.mean(error_array))
        max_error = np.degrees(np.max(error_array))
        rms_error = np.degrees(np.sqrt(np.mean(error_array**2)))
        
        print(f"Statistics: Mean={mean_error:5.2f}°, Max={max_error:5.2f}°, RMS={rms_error:5.2f}°")
        
        results.append({
            "joint_name": joint_name,
            "joint_idx": joint_idx,
            "row": row,
            "col": col,
            "mean_error": mean_error,
            "max_error": max_error,
            "rms_error": rms_error,
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary of All Joints:")
    print(f"{'='*60}")
    print(f"{'Joint':<8} {'Row':>4} {'Col':>4} {'Mean Error':<12} {'Max Error':<12} {'RMS Error':<12}")
    print(f"{'-'*70}")
    for res in results:
        print(f"{res['joint_name']:<8} {res['row']:>4} {res['col']:>4} "
              f"{res['mean_error']:>10.2f}°  {res['max_error']:>10.2f}°  {res['rms_error']:>10.2f}°")
    
    # Group by joint type
    print(f"\n{'='*60}")
    print("Grouped by Joint Type:")
    print(f"{'='*60}")
    
    for joint_type in ["M1", "M2", "M3", "M4"]:
        type_results = [r for r in results if joint_type in r["joint_name"]]
        if type_results:
            print(f"\n{joint_type} Joints:")
            avg_mean = np.mean([r["mean_error"] for r in type_results])
            avg_max = np.mean([r["max_error"] for r in type_results])
            avg_rms = np.mean([r["rms_error"] for r in type_results])
            print(f"  Average Mean Error: {avg_mean:5.2f}°")
            print(f"  Average Max Error:  {avg_max:5.2f}°")
            print(f"  Average RMS Error:  {avg_rms:5.2f}°")
    
    # Create summary plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle('DG3F Joint Control Test - Grid View', fontsize=16, fontweight='bold')
        
        time_array = np.array(time_steps) * 0.01
        
        for env_idx, (joint_idx, joint_name, osc_range, osc_center, row, col) in enumerate(TEST_CONFIGS):
            ax = axes[row, col]
            
            target_deg = np.degrees(target_angles_all[env_idx])
            actual_deg = np.degrees(actual_angles_all[env_idx])
            error_deg = np.abs(target_deg - actual_deg)
            
            # Plot target and actual
            ax.plot(time_array, target_deg, 'k-', label='Target', linewidth=1.5, alpha=0.7)
            ax.plot(time_array, actual_deg, 'r-', label='Actual', linewidth=1.5)
            
            # Title with statistics
            res = results[env_idx]
            ax.set_title(f'{joint_name}\nMean: {res["mean_error"]:.1f}° RMS: {res["rms_error"]:.1f}°',
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Angle (°)', fontsize=8)
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
        
        # Hide unused subplots (column 3 in rows 0-3)
        for row in range(4):
            axes[row, 3].axis('off')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "all_joints_summary.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Summary plot saved: {plot_path}")
        
    except ImportError:
        print("⚠️ matplotlib not available, skipping plots")
    
    print(f"\n{'='*60}")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"   - Video: all_joints_grid.mp4 (grid view of all joints)")
    print(f"   - Plot:  all_joints_summary.png (tracking performance)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
