"""
Throw environment that reuses the CS63 arm + DG3F gripper connection
validated in ``test_arm_gripper_connection.py``.

Includes ThrowEnv for RL training and Manipulator class for programmatic control.
"""
from typing import Sequence, Literal
import os
from pathlib import Path
import math

import numpy as np
import torch
import yaml

import genesis as gs
from genesis.utils.geom import (
    xyz_to_quat,
    transform_quat_by_quat,
    transform_by_quat,
)

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


## ------------ Load initial config ----------------
def load_initial_config(config_path: str = "examples/manipulation/initial_joint_config.yaml"):
    """Load initial joint configuration from YAML file."""
    # Try relative path first, then absolute
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(__file__), "initial_joint_config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse arm joints in order
    arm_joint_names = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    arm_positions = [config["arm_joints"][name] for name in arm_joint_names]
    
    # Parse gripper joints in order
    gripper_joint_names = [
        "F1M1", "F1M2", "F1M3", "F1M4",
        "F2M1", "F2M2", "F2M3", "F2M4",
        "F3M1", "F3M2", "F3M3", "F3M4"
    ]
    gripper_positions = [config["gripper_joints"][name] for name in gripper_joint_names]
    
    return arm_positions, gripper_positions

# Load default joint angles from config file
ARM_INITIAL_POSITIONS, GRIPPER_INITIAL_POSITIONS = load_initial_config()
DEFAULT_JOINT_ANGLES = ARM_INITIAL_POSITIONS + GRIPPER_INITIAL_POSITIONS


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
        self.num_obs = env_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.image_width = env_cfg.get("image_resolution", [128, 128])[0]
        self.image_height = env_cfg.get("image_resolution", [128, 128])[1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        # configs
        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
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
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            ik_method=robot_cfg.get("ik_method", "gs_ik"),
            device=self.device,
            base_height=robot_cfg.get("base_height", 0.0),
            action_scale=robot_cfg.get("action_scale", 1.0),
            dls_lambda=robot_cfg.get("dls_lambda", 0.01),
        )

        # == add throwable object ==
        self.object = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=env_cfg.get("ball_radius", 0.05),
                pos=(0.0, 0.0, 0.5),
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
        )

        # == add cameras (optional) ==
        if env_cfg.get("use_camera", False):
            self.camera = self.scene.add_camera(
                res=(self.image_width, self.image_height),
                pos=(1.5, 0.0, 1.0),
                lookat=(0.0, 0.0, 0.5),
                fov=60,
                GUI=False,
            )

        # build scene
        self.scene.build(n_envs=env_cfg["num_envs"])
        
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
        self.episode_length_buf[envs_idx] = 0

        # reset robot
        self.robot.reset(envs_idx)

        # reset object to gripper position
        # TODO: Implement object reset logic
        pass

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
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
        actions = self.rescale_action(actions)
        
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
        # TODO: Implement observation logic
        # Example observation components:
        # - robot state (joint positions, velocities)
        # - object state (position, velocity)
        # - goal state (target landing position)
        obs_tensor = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.extras["observations"]["critic"] = obs_tensor
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

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Rescale actions by action scales"""
        rescaled_action = action * self.action_scales
        return rescaled_action

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
        ik_method: Literal["gs_ik", "dls_ik"] = "gs_ik",
        device: str = "cuda:0",
        base_height: float = 0.0,
        action_scale: float = 1.0,
        dls_lambda: float = 0.01
    ) -> None:
        self._scene = scene
        self._num_envs = num_envs
        self._device = device
        self._ik_method = ik_method
        self._action_scale = action_scale
        self._dls_lambda = dls_lambda
        
        from genesis.engine.entities.rigid_entity import RigidEntity
        self.arm_entity:RigidEntity = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/cs63/cs63.urdf",
                merge_fixed_links=False,
                fixed=True,
                pos=(0.0, 0.0, base_height),
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        self.gripper_entity:RigidEntity = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/DG3F/urdf/delto_gripper_3f.urdf",
                merge_fixed_links=False,
                fixed=True,
                decompose_robot_error_threshold=0.15, # Force better decomposition
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )
        scene.link_entities(
            parent_entity=self.arm_entity,
            child_entity=self.gripper_entity,
            parent_link_name="delto_base_link_cs63",
            child_link_name="delto_base_link",
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
            [self.arm_entity.get_joint(name).dofs_idx_local[0] for name in arm_joints_name],
            device=self._device
        )
        
        # Gripper DOF indices
        self._gripper_dof_dim = 12
        self._gripper_dof_idx = torch.tensor(
            [self.gripper_entity.get_joint(name).dofs_idx_local[0] for name in gripper_joints_name],
            device=self._device
        )
        
        # End-effector link (flange from arm)
        self._ee_link = self.arm_entity.get_link("wrist_3_link")
        
        # Gripper finger tip links
        self._finger1_tip_link = self.gripper_entity.get_link("F1_TIP")
        self._finger2_tip_link = self.gripper_entity.get_link("F2_TIP")
        self._finger3_tip_link = self.gripper_entity.get_link("F3_TIP")
        
        # Palm link (gripper base control frame)
        self._palm_link = self.gripper_entity.get_link("palm_link")
        
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
        gripper_kp = torch.tensor([60] * 12, device=self._device)
        gripper_kv = torch.tensor([5] * 12, device=self._device)
        gripper_force_min = torch.tensor([-5] * 12, device=self._device)
        gripper_force_max = torch.tensor([5] * 12, device=self._device)

        self.arm_entity.set_dofs_kp(arm_kp, self._arm_dof_idx)
        self.arm_entity.set_dofs_kv(arm_kv, self._arm_dof_idx)
        self.arm_entity.set_dofs_force_range(arm_force_min, arm_force_max, self._arm_dof_idx)
        
        self.gripper_entity.set_dofs_kp(gripper_kp, self._gripper_dof_idx)
        self.gripper_entity.set_dofs_kv(gripper_kv, self._gripper_dof_idx)
        self.gripper_entity.set_dofs_force_range(gripper_force_min, gripper_force_max, self._gripper_dof_idx)
        
        # Disable collision between F*_02 links and delto_base_link (only once after build)
        if not self._scene_built:
            for finger in ["F1", "F2", "F3"]:
                disable_collision_between_links(self.gripper_entity, f"{finger}_02", "delto_base_link")
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
        self.arm_entity.set_qpos(default_joint_angles[:, :self._arm_dof_dim], qs_idx_local=self._arm_dof_idx, envs_idx=envs_idx)
        
        # Set gripper positions
        self.gripper_entity.set_qpos(default_joint_angles[:, self._arm_dof_dim:], qs_idx_local=self._gripper_dof_idx, envs_idx=envs_idx)

    def apply_action(
        self, 
        arm_action: torch.Tensor, 
        gripper_action: torch.Tensor
    ) -> None:
        """
        Apply action to robot.
        
        Parameters
        ----------
        arm_action : torch.Tensor
            End-effector pose delta [B, 6] (position delta + orientation delta in RPY)
        gripper_action : torch.Tensor
            Gripper joint position targets [B, 12]
        """
        # Scale action for stronger effect
        scaled_action = arm_action * self._action_scale
        
        # Compute arm joint positions using IK
        if self._ik_method == "gs_ik":
            arm_qpos = self._gs_ik(scaled_action)
        elif self._ik_method == "dls_ik":
            arm_qpos = self._dls_ik(scaled_action)
        else:
            raise ValueError(f"Invalid IK method: {self._ik_method}")
        
        # Apply arm control
        self.arm_entity.control_dofs_position(position=arm_qpos, dofs_idx_local=self._arm_dof_idx)
        
        # Apply gripper control
        self.gripper_entity.control_dofs_position(position=gripper_action, dofs_idx_local=self._gripper_dof_idx)
    
    def _gs_ik(self, action: torch.Tensor) -> torch.Tensor:
        """Genesis inverse kinematics"""
        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]
        
        # Compute target pose
        target_position = delta_position + self._ee_link.get_pos()
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())
        
        # Solve IK
        q_pos = self.arm_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_position,
            quat=target_orientation,
            dofs_idx_local=self._arm_dof_idx,
        )
        return q_pos
    
    def _dls_ik(self, action: torch.Tensor) -> torch.Tensor:
        """Damped least squares inverse kinematics"""
        delta_pose = action[:, :6]
        
        jacobian = self.arm_entity.get_jacobian(link=self._ee_link)
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = (self._dls_lambda ** 2) * torch.eye(n=jacobian.shape[1], device=self._device)
        
        delta_joint_pos = (
            jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)
        
        return self.arm_entity.get_dofs_position(self._arm_dof_idx) + delta_joint_pos

    def command_arm(self, target: Sequence[float] | torch.Tensor) -> None:
        """Command arm joints to target positions using PD control"""
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32, device=self._device)
        if target.ndim == 1:
            target = target.unsqueeze(0).repeat(self._num_envs, 1)
        self.arm_entity.control_dofs_position(target, self._arm_dof_idx)

    def command_gripper(self, target: Sequence[float] | torch.Tensor) -> None:
        """Command gripper joints to target positions using PD control"""
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32, device=self._device)
        if target.ndim == 1:
            target = target.unsqueeze(0).repeat(self._num_envs, 1)
        self.gripper_entity.control_dofs_position(target, self._gripper_dof_idx)

    def command_arm_force(self, force: Sequence[float] | torch.Tensor) -> None:
        """Apply force/torque control to arm joints"""
        if not isinstance(force, torch.Tensor):
            force = torch.tensor(force, dtype=torch.float32, device=self._device)
        if force.ndim == 1:
            force = force.unsqueeze(0).repeat(self._num_envs, 1)
        self.arm_entity.control_dofs_force(force, self._arm_dof_idx)

    def command_gripper_force(self, force: Sequence[float] | torch.Tensor) -> None:
        """Apply force/torque control to gripper joints"""
        if not isinstance(force, torch.Tensor):
            force = torch.tensor(force, dtype=torch.float32, device=self._device)
        if force.ndim == 1:
            force = force.unsqueeze(0).repeat(self._num_envs, 1)
        self.gripper_entity.control_dofs_force(force, self._gripper_dof_idx)

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
        
        self.arm_entity.control_dofs_position_velocity(position, velocity, self._arm_dof_idx)

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
        
        self.gripper_entity.control_dofs_position_velocity(position, velocity, self._gripper_dof_idx)

    def command_full_state(
        self,
        arm_position: torch.Tensor,
        arm_velocity: torch.Tensor,
        gripper_position: torch.Tensor,
        gripper_velocity: torch.Tensor
    ) -> None:
        """Command both arm and gripper with position and velocity targets"""
        self.arm_entity.control_dofs_position_velocity(arm_position, arm_velocity, self._arm_dof_idx)
        self.gripper_entity.control_dofs_position_velocity(gripper_position, gripper_velocity, self._gripper_dof_idx)
    
    def go_to_goal(self, goal_pose: torch.Tensor, gripper_target: torch.Tensor) -> None:
        """
        Go to goal pose using IK.
        
        Parameters
        ----------
        goal_pose : torch.Tensor
            Target end-effector pose [B, 7] (position + quaternion)
        gripper_target : torch.Tensor
            Target gripper joint positions [B, 12]
        """
        q_pos = self.arm_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )
        self.arm_entity.control_dofs_position(position=q_pos, dofs_idx_local=self._arm_dof_idx)
        self.gripper_entity.control_dofs_position(position=gripper_target, dofs_idx_local=self._gripper_dof_idx)

    # ============ Properties ============
    @property
    def base_pos(self) -> torch.Tensor:
        """Robot base position"""
        return self.arm_entity.get_pos().to(self._device)
    
    @property
    def ee_pose(self) -> torch.Tensor:
        """End-effector pose [B, 7] (position + quaternion)"""
        pos = self._ee_link.get_pos()
        quat = self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1).to(self._device)
    
    @property
    def finger_tips_pose(self) -> torch.Tensor:
        """
        All three finger tips poses [B, 3, 7] (position + quaternion for each finger).
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
        return self.arm_entity.get_dofs_position(self._arm_dof_idx).to(self._device)
    
    @property
    def gripper_qpos(self) -> torch.Tensor:
        """Gripper joint positions [B, 12]"""
        return self.gripper_entity.get_dofs_position(self._gripper_dof_idx).to(self._device)
    
    @property
    def arm_qvel(self) -> torch.Tensor:
        """Arm joint velocities [B, 6]"""
        return self.arm_entity.get_dofs_velocity(self._arm_dof_idx).to(self._device)
    
    @property
    def gripper_qvel(self) -> torch.Tensor:
        """Gripper joint velocities [B, 12]"""
        return self.gripper_entity.get_dofs_velocity(self._gripper_dof_idx).to(self._device)
    
    @property
    def palm_pose(self) -> torch.Tensor:
        """Palm link pose [B, 7] (position + quaternion)"""
        pos = self._palm_link.get_pos()
        quat = self._palm_link.get_quat()
        return torch.cat([pos, quat], dim=-1).to(self._device)
