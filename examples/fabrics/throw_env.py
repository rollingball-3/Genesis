"""
Throw environment that reuses the CS63 arm + DG3F gripper connection
validated in ``test_arm_gripper_connection.py``.

Includes ThrowEnv for RL training and Manipulator class for programmatic control.
"""
from typing import Sequence

import numpy as np
import torch
import yaml

import genesis as gs
from utils import load_initial_joint_config, disable_collision_between_links, load_throwable_objects_config

# Load default joint angles from config file
DEFAULT_JOINT_ANGLES = load_initial_joint_config()


## ------------ ThrowEnv ----------------
class ThrowEnv:
    """
    Throwing environment for RL training with CS63 arm + DG3F gripper.
    Similar structure to GraspEnv but adapted for throwing tasks.
    """
    
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        # TODO define obs
        self.num_obs = env_cfg["num_obs"]
        # TODO define privileged obs
        self.num_privileged_obs = None
        # TODO define actions
        self.num_actions = env_cfg["num_actions"]
        # TODO define image shape
        self.image_width = env_cfg.get("image_resolution", [1280, 720])[0]
        self.image_height = env_cfg.get("image_resolution", [1280, 720])[1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = env_cfg["max_episode_length"]

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg

        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=10),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, self.num_envs)))),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        # == add ground ==
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # == add robot ==
        self.robot:Manipulator = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            device=self.device,
            base_height=robot_cfg.get("base_height", 0.0),
        )

        # Load throwable objects configuration
        throwable_objects_config = load_throwable_objects_config()
        # Add throwable objects from the configuration
        self.objects = []
        for i, obj_config in enumerate(throwable_objects_config):
            obj_name = obj_config["name"]  # Get object name from config
            urdf_path = obj_config["urdf_path"]
            # Add the object to the scene
            obj = self.scene.add_entity(
                gs.morphs.URDF(
                    file=urdf_path,
                    fixed=False,  # Make the object movable
                ),
                surface=gs.surfaces.Rough(),
                vis_mode = "collision",
            )
            self.objects.append(obj)
        
        # Add a simple Box object for testing purposes
        box_obj = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.2, 0.2, 0.2),  # 0.2m x 0.2m x 0.2m box
                fixed=False,  # Make the box movable
            ),
            surface=gs.surfaces.Rough(),
            vis_mode = "collision",
        )
        self.objects.append(box_obj)

        # Add observation cameras (for RL agent's visual input)
        self.observation_cameras = {
            'right': self.scene.add_camera(
                model='pinhole',
                res=(self.image_width, self.image_height),
                pos=(1.5, -1.5, 3.0),  # Right side view
                lookat=(1.5, 0.5, 0.5),  # Look at the center of the scene
                fov=60,
                GUI=False,  # Not shown in GUI, only for observation
            ),
            'front': self.scene.add_camera(
                model='pinhole',
                res=(self.image_width, self.image_height),
                pos=(3.0, 0.0, 3.0),  # Front view
                lookat=(1.5, 0.5, 0.5),
                fov=60,
                GUI=False,
            ),
            'left': self.scene.add_camera(
                model='pinhole',
                res=(self.image_width, self.image_height),
                pos=(1.5, 1.5, 3.0),  # Left side view
                lookat=(1.5, 0.5, 0.5),
                fov=60,
                GUI=False,
            ),
            'top': self.scene.add_camera(
                model='pinhole',
                res=(self.image_width, self.image_height),
                pos=(1.5, 0.0, 6.0),  # Top-down view
                lookat=(1.5, 0.5, 0.5),
                fov=60,
                GUI=False,
            ),
        }

        # Add global visualization camera (for debugging and monitoring)
        self.global_camera = self.scene.add_camera(
            model='pinhole',
            res=(self.image_width, self.image_height),
            pos=(-3.0, -3.0, 4.0),  # High global view
            lookat=(1.5, 1.5, 1.0),  # Look at the center of the scene
            fov=45,
            GUI=False,  # Show in GUI for visualization
            debug=True,  # Mark as debug camera to avoid interfering with observation cameras
        )

        # build scene
        self.scene.build(n_envs=env_cfg["num_envs"], env_spacing=(2.0, 2.0))
        
        # set pd gains (must be called after scene.build)
        self.robot.set_pd_gains()

        # prepare reward functions
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # == init buffers ==
        self._init_buffers()
        self.reset()

    def _init_buffers(self) -> None:
        """Initialize episode tracking buffers"""
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        """Reset specific environments"""
        if len(envs_idx) == 0:
            return
        
        # Sanitize envs_idx using scene's method to ensure it's 1D
        envs_idx = self.scene._sanitize_envs_idx(envs_idx)
        
        self.episode_length_buf[envs_idx] = 0

        # reset robot
        self.robot.reset(envs_idx)

        # reset objects
        # Select a random object from self.objects to be thrown
        selected_object_idx = torch.randint(0, len(self.objects), (1,), device=self.device).item()
        thrown_object = self.objects[selected_object_idx]
        
        # Set thrown object position at x = [-1.0, 1.0] ,y = [-0.5, 2.5], z = [1.0, 2.0]
        # Create object_pos with shape [len(envs_idx), 3]
        thrown_object_pos = torch.zeros((len(envs_idx), 3), device=self.device)
        thrown_object_pos[:, 0] = -1.0 + torch.rand((len(envs_idx),), device=self.device) * 2.0  # x: [-1, 1]
        thrown_object_pos[:, 1] = -0.5 + torch.rand((len(envs_idx),), device=self.device) * 3.0  # y: [-0.5, 2.5]
        thrown_object_pos[:, 2] = 1.0 + torch.rand((len(envs_idx),), device=self.device) * 1.0  # z
        
        # Set position of thrown object - use envs_idx directly
        thrown_object.set_pos(thrown_object_pos, envs_idx=envs_idx)
        
        # 3. give thrown object initial velocity to throw toward robot, robot base is [0, 0, 0]
        # Calculate direction from object to robot base
        direction_to_robot = torch.zeros_like(thrown_object_pos)
        direction_to_robot[:, 0] = -thrown_object_pos[:, 0]  # x direction: from object to robot (0,0,0)
        direction_to_robot[:, 1] = -thrown_object_pos[:, 1]  # y direction: from object to robot (0,0,0)
        direction_to_robot[:, 2] = -thrown_object_pos[:, 2] * 0.5  # z direction: slight downward motion
        
        # Normalize direction vector
        direction_magnitude = torch.norm(direction_to_robot, dim=1, keepdim=True) + 1e-8
        normalized_direction = direction_to_robot / direction_magnitude
        
        # Set initial velocity with magnitude between 1.0 and 3.0
        velocity_magnitude = 1.0 + torch.rand((len(envs_idx), 1), device=self.device) * 2.0
        linear_vel = normalized_direction * velocity_magnitude
        
        # Create 6-DOF velocity tensor (3 linear, 3 angular with zero)
        angular_vel = torch.zeros_like(linear_vel)
        object_vel = torch.cat([linear_vel, angular_vel], dim=1)
        
        # Set velocity of thrown object with correct 6 DOFs
        thrown_object.set_dofs_velocity(object_vel, envs_idx=envs_idx)
        
        # 4. Set all other objects to a position that doesn't interfere with the throwing task
        # Place them far away from the main task area
        for i, obj in enumerate(self.objects):
            if i != selected_object_idx:
                # Place non-thrown objects far away (x=10.0, y=10.0, z=5.0)
                non_thrown_pos = torch.full((len(envs_idx), 3), 10.0, device=self.device)
                non_thrown_pos[:, 2] = 5.0  # Set z to 5.0
                obj.set_pos(non_thrown_pos, envs_idx=envs_idx)
                
                # Set their velocity to zero
                zero_vel = torch.zeros((len(envs_idx), 6), device=self.device)
                obj.set_dofs_velocity(zero_vel, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.max_episode_length
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environments"""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, self.extras = self.get_observations()
        return obs, self.extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment"""
        # update time
        self.episode_length_buf += 1

        # apply action
        # TODO: Implement action application
        # arm_action = actions[:, :6]  # ee pose delta
        # gripper_action = actions[:, 6:]  # gripper joint targets
        # self.robot.apply_action(arm_action, gripper_action)
        
        self.scene.step()

        # check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward
        reward = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations
        obs, self.extras = self.get_observations()

        return obs, reward, self.reset_buf, self.extras

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Get observations for the policy"""
        # Get visual observations from all cameras
        visual_obs = {}
        for camera_name, camera in self.observation_cameras.items():
            # Render image from camera
            rgb_image = camera.render()
            visual_obs[camera_name] = rgb_image
        
        # Get robot state observations
        robot_joint_pos = self.robot.get_joint_positions()
        robot_joint_vel = self.robot.get_joint_velocities()
        robot_ee_pose = self.robot.ee_pose
        
        # Get object state observations
        object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        object_vel = torch.zeros((self.num_envs, 6), device=self.device)
        for obj in self.objects:
            object_pos += obj.get_pos()
            object_vel += obj.get_dofs_velocity()
        object_pos /= len(self.objects)
        object_vel /= len(self.objects)
        
        # Concatenate all non-visual observations into a tensor
        obs_components = [
            robot_joint_pos,
            robot_joint_vel,
            robot_ee_pose.view(self.num_envs, -1),  # Flatten pose
            object_pos,
            object_vel,
        ]
        
        # Ensure all components have the right shape (num_envs, ...)
        for i, component in enumerate(obs_components):
            if component.dim() == 1:
                obs_components[i] = component.unsqueeze(1)
        
        # Concatenate all components
        obs_tensor = torch.cat(obs_components, dim=1)[:, :self.num_obs]  # Truncate to expected size
        
        # Update extras with observations
        self.extras["observations"]["critic"] = obs_tensor
        self.extras["observations"]["visual"] = visual_obs
        self.extras["observations"]["robot_state"] = {
            "joint_pos": robot_joint_pos,
            "joint_vel": robot_joint_vel,
            "ee_pose": robot_ee_pose,
        }
        self.extras["observations"]["object_state"] = {
            "pos": object_pos,
            "vel": object_vel,
        }
        
        return obs_tensor, self.extras

    def get_privileged_observations(self) -> None:
        """Get privileged observations (if any)"""
        return None

    def is_episode_complete(self) -> torch.Tensor:
        """Check if episodes are complete"""
        time_out_buf = self.episode_length_buf > self.max_episode_length
        
        # TODO: Add task-specific termination conditions
        # - object landed successfully
        # - object out of bounds
        # - robot safety violations
        
        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]



    # ------------ Reward functions (to be implemented) ----------------
    def _reward_tracking(self) -> torch.Tensor:
        """Reward for tracking target trajectory"""
        # TODO: Implement
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_velocity(self) -> torch.Tensor:
        """Reward for achieving target release velocity"""
        # TODO: Implement
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_accuracy(self) -> torch.Tensor:
        """Reward for throwing accuracy (landing near target)"""
        # TODO: Implement
        return torch.zeros(self.num_envs, device=self.device)


## ------------ Manipulator ----------------
class Manipulator:

    def __init__(
        self, 
        scene: gs.Scene, 
        num_envs: int, 
        device: str = "cuda:0",
        base_height: float = 0.0,
    ) -> None:
        self._scene = scene
        self._num_envs = num_envs
        self._device = device
        
        from genesis.engine.entities.rigid_entity import RigidEntity
        self.robot:RigidEntity = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/cs63_tesollo/cs63_tesollo.urdf",
                merge_fixed_links=False,
                fixed=True,
                pos=(0.0, 0.0, base_height),
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
            vis_mode = "collision",
        )

        self._init_joint_indices()
        self._arm_command = None
        self._gripper_command = None
        
        # Store reference to scene for post-build operations
        self._scene_built = False

    def _init_joint_indices(self) -> None:
        arm_joints_name = (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        )
        gripper_joints_name = (
            "F1M1",
            "F1M2",
            "F1M3",
            "F1M4",
            "F2M1",
            "F2M2",
            "F2M3",
            "F2M4",
            "F3M1",
            "F3M2",
            "F3M3",
            "F3M4",
        )
        
        # Arm DOF indices
        self._arm_dof_dim = 6
        self._arm_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dofs_idx_local[0] for name in arm_joints_name],
            device=self._device
        )
        
        # Gripper DOF indices
        self._gripper_dof_dim = 12
        self._gripper_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dofs_idx_local[0] for name in gripper_joints_name],
            device=self._device
        )
        
        # End-effector link (flange from arm)
        self._ee_link = self.robot.get_link("wrist_3_link")
        
        # Gripper finger tip links
        self._finger1_tip_link = self.robot.get_link("F1_TIP")
        self._finger2_tip_link = self.robot.get_link("F2_TIP")
        self._finger3_tip_link = self.robot.get_link("F3_TIP")
        
        # Force frame links (defined in cs63_tesollo.urdf)
        self._tip1_force_frame = self.robot.get_link("tip1_force_frame")
        self._tip2_force_frame = self.robot.get_link("tip2_force_frame")
        self._tip3_force_frame = self.robot.get_link("tip3_force_frame")
        
        # Palm link (gripper base control frame)
        self._palm_link = self.robot.get_link("palm_link")
        
        # Default joint angles
        self._default_joint_angles = DEFAULT_JOINT_ANGLES

    def set_pd_gains(self) -> None:
        """Set PD control gains for arm and gripper (must be called after scene.build)"""
        # Arm (cs63) PD control parameters
        arm_kp = torch.tensor([3500, 3500, 2500, 2500, 1200, 1200], device=self._device)
        arm_kv = torch.tensor([350, 350, 250, 250, 120, 120], device=self._device)
        arm_force_min = torch.tensor([-56, -56, -28, -28, -12, -12], device=self._device)
        arm_force_max = torch.tensor([56, 56, 28, 28, 12, 12], device=self._device)

        # DG3F gripper PD control parameters (12 joints: 3 fingers × 4 joints each)
        # All joints use same gains based on ROS controller config (p:1.2, d:0.1)
        # Scaled up for Genesis simulation environment
        gripper_kp = torch.tensor([120] * 12, device=self._device)
        gripper_kv = torch.tensor([10] * 12, device=self._device)
        gripper_force_min = torch.tensor([-10] * 12, device=self._device)
        gripper_force_max = torch.tensor([10] * 12, device=self._device)

        self.robot.set_dofs_kp(arm_kp, self._arm_dof_idx)
        self.robot.set_dofs_kv(arm_kv, self._arm_dof_idx)
        self.robot.set_dofs_force_range(arm_force_min, arm_force_max, self._arm_dof_idx)
        
        self.robot.set_dofs_kp(gripper_kp, self._gripper_dof_idx)
        self.robot.set_dofs_kv(gripper_kv, self._gripper_dof_idx)
        self.robot.set_dofs_force_range(gripper_force_min, gripper_force_max, self._gripper_dof_idx)
        
        # Disable collision between F*_02 links and delto_base_link (only once after build)
        if not self._scene_built:
            for finger in ["F1", "F2", "F3"]:
                disable_collision_between_links(self.robot, f"{finger}_02", "delto_base_link")
            print("✅ Disabled collision between F*_02 links and delto_base_link")
            self._scene_built = True

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        """Reset arm and gripper to initial positions"""
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)
        if len(envs_idx) == 0:
            return
        
        default_joint_angles = torch.tensor(
            self._default_joint_angles, dtype=torch.float32, device=self._device
        ).repeat(len(envs_idx), 1)
        
        # Set arm positions
        self.robot.set_qpos(default_joint_angles[:, :self._arm_dof_dim], qs_idx_local=self._arm_dof_idx, envs_idx=envs_idx)
        
        # Set gripper positions
        self.robot.set_qpos(default_joint_angles[:, self._arm_dof_dim:], qs_idx_local=self._gripper_dof_idx, envs_idx=envs_idx)

    def command_arm(self, target: Sequence[float] | torch.Tensor) -> None:
        """Command arm joints to target positions using PD control"""
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32, device=self._device)
        if target.ndim == 1:
            target = target.unsqueeze(0).repeat(self._num_envs, 1)
        self.robot.control_dofs_position(target, self._arm_dof_idx)

    def command_gripper(self, target: Sequence[float] | torch.Tensor) -> None:
        """Command gripper joints to target positions using PD control"""
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32, device=self._device)
        if target.ndim == 1:
            target = target.unsqueeze(0).repeat(self._num_envs, 1)
        self.robot.control_dofs_position(target, self._gripper_dof_idx)

    def command_arm_force(self, force: Sequence[float] | torch.Tensor) -> None:
        """Apply force/torque control to arm joints"""
        if not isinstance(force, torch.Tensor):
            force = torch.tensor(force, dtype=torch.float32, device=self._device)
        if force.ndim == 1:
            force = force.unsqueeze(0).repeat(self._num_envs, 1)
        self.robot.control_dofs_force(force, self._arm_dof_idx)

    def command_gripper_force(self, force: Sequence[float] | torch.Tensor) -> None:
        """Apply force/torque control to gripper joints"""
        if not isinstance(force, torch.Tensor):
            force = torch.tensor(force, dtype=torch.float32, device=self._device)
        if force.ndim == 1:
            force = force.unsqueeze(0).repeat(self._num_envs, 1)
        self.robot.control_dofs_force(force, self._gripper_dof_idx)

    def command_arm_position_velocity(
        self, 
        position: Sequence[float] | torch.Tensor,
        velocity: Sequence[float] | torch.Tensor
    ) -> None:
        """Command arm joints with both position and velocity targets for PD control"""
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=torch.float32, device=self._device)
        if position.ndim == 1:
            position = position.unsqueeze(0).repeat(self._num_envs, 1)
        
        if not isinstance(velocity, torch.Tensor):
            velocity = torch.tensor(velocity, dtype=torch.float32, device=self._device)
        if velocity.ndim == 1:
            velocity = velocity.unsqueeze(0).repeat(self._num_envs, 1)
        
        self.robot.control_dofs_position_velocity(position, velocity, self._arm_dof_idx)

    def command_gripper_position_velocity(
        self,
        position: Sequence[float] | torch.Tensor,
        velocity: Sequence[float] | torch.Tensor
    ) -> None:
        """Command gripper joints with both position and velocity targets for PD control"""
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=torch.float32, device=self._device)
        if position.ndim == 1:
            position = position.unsqueeze(0).repeat(self._num_envs, 1)
        
        if not isinstance(velocity, torch.Tensor):
            velocity = torch.tensor(velocity, dtype=torch.float32, device=self._device)
        if velocity.ndim == 1:
            velocity = velocity.unsqueeze(0).repeat(self._num_envs, 1)
        
        self.robot.control_dofs_position_velocity(position, velocity, self._gripper_dof_idx)
    
    def get_joint_positions(self) -> torch.Tensor:
        """Get current joint positions of arm and gripper"""
        return self.robot.get_dofs_position()[..., torch.cat([self._arm_dof_idx, self._gripper_dof_idx])]
    
    def get_joint_velocities(self) -> torch.Tensor:
        """Get current joint velocities of arm and gripper"""
        return self.robot.get_dofs_velocity()[..., torch.cat([self._arm_dof_idx, self._gripper_dof_idx])]
    


    # ============ Properties ============
    @property
    def base_pos(self) -> torch.Tensor:
        """Robot base position"""
        return self.robot.get_pos().to(self._device)
    
    @property
    def ee_pose(self) -> torch.Tensor:
        """End-effector pose [B, 7] (position + quaternion)"""
        pos = self._ee_link.get_pos()
        quat = self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1).to(self._device)
    
    @property
    def finger_tips_pose(self) -> torch.Tensor:
        """All three finger tips poses [B, 3, 7] (position + quaternion for each finger).
        Returns stacked poses: [finger1, finger2, finger3]
        """
        poses = []
        for finger_link in [self._finger1_tip_link, self._finger2_tip_link, self._finger3_tip_link]:
            pos = finger_link.get_pos()
            quat = finger_link.get_quat()
            pose = torch.cat([pos, quat], dim=-1)
            poses.append(pose)
        return torch.stack(poses, dim=1).to(self._device)  # [B, 3, 7]
    
    @property
    def arm_qpos(self) -> torch.Tensor:
        """Arm joint positions [B, 6]"""
        return self.robot.get_dofs_position(self._arm_dof_idx).to(self._device)
    
    @property
    def gripper_qpos(self) -> torch.Tensor:
        """Gripper joint positions [B, 12]"""
        return self.robot.get_dofs_position(self._gripper_dof_idx).to(self._device)
    
    @property
    def arm_qvel(self) -> torch.Tensor:
        """Arm joint velocities [B, 6]"""
        return self.robot.get_dofs_velocity(self._arm_dof_idx).to(self._device)
    
    @property
    def gripper_qvel(self) -> torch.Tensor:
        """Gripper joint velocities [B, 12]"""
        return self.robot.get_dofs_velocity(self._gripper_dof_idx).to(self._device)
    
    @property
    def palm_pose(self) -> torch.Tensor:
        """Palm link pose [B, 7] (position + quaternion)"""
        pos = self._palm_link.get_pos()
        quat = self._palm_link.get_quat()
        return torch.cat([pos, quat], dim=-1).to(self._device)
    
    @property
    def finger_force_frames_pose(self) -> torch.Tensor:
        """
        All three finger force frames poses [B, 3, 7] (position + quaternion for each frame).
        Returns stacked poses: [frame1, frame2, frame3]
        """
        poses = []
        for frame_link in [self._tip1_force_frame, self._tip2_force_frame, self._tip3_force_frame]:
            pos = frame_link.get_pos()
            quat = frame_link.get_quat()
            pose = torch.cat([pos, quat], dim=-1)
            poses.append(pose)
        return torch.stack(poses, dim=1).to(self._device)  # [B, 3, 7]
