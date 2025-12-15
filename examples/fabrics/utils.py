"""
Utility functions for Genesis + FABRICS integration.

This module contains common functions for:
- Genesis simulation setup
- FABRICS controller initialization
- Coordinate transformations
- Visualization
- Data logging and plotting
"""
import yaml
from pathlib import Path


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

def load_throwable_objects_config():
    """Load throwable objects configuration from YAML file"""
    config_path = Path(__file__).parent / "throwable_objects.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config["objects"]
    