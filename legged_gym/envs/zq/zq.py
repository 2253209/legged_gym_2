# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class ZqRobot(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.target_joint_angles = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            self.target_joint_angles[i] = self.cfg.init_state.target_joint_angles[i]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)

    def compute_observations(self):
        """ Computes observations
        """
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.obs_buf = torch.cat((
            # self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            # self.base_ang_vel  * self.obs_scales.ang_vel,  # 3
            # self.projected_gravity,  # 1
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz,  # 3
            self.commands[:, :3] * self.commands_scale,  # 3
            self.dof_pos * self.obs_scales.dof_pos,  # 10
            self.dof_vel * self.obs_scales.dof_vel,  # 10
            self.actions  # 10
            ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reset_root_states(self, env_ids):
        """ 重置ROOT状态所选环境的身体姿态
            随机给+-17度的 R P Y 角度
        Args:
            env_ids (List[int]): Environemnt ids
        """
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            rpy = torch_rand_float(-0.1, 0.1, (len(env_ids), 3), device=self.device)
            for index, id in enumerate(env_ids):
                self.root_states[id, 3:7] = quat_from_euler_xyz(rpy[index, 0], rpy[index, 1], rpy[index, 2])
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.05, 0.05, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 1.8
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # ------------------------ rewards --------------------

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1. * contacts, dim=1) == 1
        return 1. * single_contact

    def _reward_target_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        # joint_diff = self.dof_pos - self.default_joint_pd_target
        joint_diff = self.dof_pos - self.target_joint_angles
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 5: 7]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        # return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
        return torch.exp(-yaw_roll * 100) - 1.0 * torch.norm(joint_diff, dim=1)

    def _reward_body_feet_dist(self):
        # Penalize body root xy diff feet xy
        self.gym.clear_lines(self.viewer)

        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        center_pos = torch.mean(foot_pos, dim=1)
        body_pos = self.root_states[:, :2]
        pos_dist = torch.norm(body_pos[:, :] - center_pos[:, :], dim=1)

        sphere_geom_1 = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 0, 1))
        sphere_pose_1 = gymapi.Transform(gymapi.Vec3(body_pos[0, 0], body_pos[0, 1], 0.01), r=None)
        sphere_pose_2 = gymapi.Transform(gymapi.Vec3(center_pos[0, 0], center_pos[0, 1], 0.01), r=None)
        gymutil.draw_lines(sphere_geom_1, self.gym, self.viewer, self.envs[0], sphere_pose_1)
        gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[0], sphere_pose_2)

        reward = torch.square(pos_dist * 10)
        # print(f'dist={pos_dist[0]}')
        return reward
