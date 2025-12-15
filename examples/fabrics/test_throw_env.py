#!/usr/bin/env python3

import os
import sys
import torch
import yaml
from datetime import datetime
import genesis as gs

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from examples.fabrics.throw_env import ThrowEnv
    print("✓ Successfully imported ThrowEnv class")
except ImportError as e:
    print(f"✗ Failed to import ThrowEnv: {e}")
    sys.exit(1)

try:
    # Create a minimal config for testing
    env_cfg = {
        "num_envs": 16,
        "num_obs": 41,  # 6(arm) + 12(gripper) + 7(palm) + 3(obj_pos) + 4(obj_quat) + 6(obj_vel) + 3(distance) = 41
        "num_actions": 16,  # 3(palm_pos_delta) + 4(palm_quat_delta) + 9(finger_forces) = 16
        "ctrl_dt": 1.0/60,
        "use_rasterizer": True,
        "object_size": [0.05, 0.05, 0.05],
        "object_collision": True,
        "action_scales": [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Scales for all 16 actions (3+4+9)
        "max_episode_length": 1000,
    }
    
    # Use empty reward config since no reward functions are implemented yet
    reward_cfg = {}
    
    robot_cfg = {
        "base_height": 0.6,
    }
    
    print("Testing ThrowEnv initialization...")
    # Initialize Genesis before creating the environment
    gs.init(backend=gs.gpu)
    # Use headless mode since no display is available on server
    env = ThrowEnv(env_cfg, reward_cfg, robot_cfg, show_viewer=False)
    print("✓ Successfully initialized ThrowEnv")
    
    # Test reset
    print("Testing reset()...")
    obs, extras = env.reset()
    print(f"✓ Reset successful, observation shape: {obs.shape}")
    
    # Test step - do nothing and let robot and object fall to the ground
    print("Testing step() - letting robot and object fall to the ground...")
    
    # Create action tensor of zeros (do nothing)
    actions = torch.zeros((env_cfg["num_envs"], env_cfg["num_actions"]), device=env.device)
    
    # Create output directory for video with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"tmp/throw_env_test/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Run for a few seconds to let things fall
    num_steps = int(10.0 / env_cfg["ctrl_dt"])  # 10 seconds
    print(f"Running for {num_steps} steps ({10.0} seconds)...")
    
    # Start recording
    env.global_camera.start_recording()
    
    for step in range(num_steps):
        obs, rewards, resets, extras = env.step(actions)
        
        # Render camera for recording
        env.global_camera.render()
        
        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"  Step {step}/{num_steps}")
    
    print("✓ Step test completed - robot and object should have fallen to the ground")
    
    # Stop recording and save video
    video_path = os.path.join(output_dir, "fall_test.mp4")
    env.global_camera.stop_recording(save_to_filename=video_path)
    print(f"✓ Video saved to {video_path}")
    
    print("\nAll tests passed! ThrowEnv is working correctly.")
    
except Exception as e:
    print(f"✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
