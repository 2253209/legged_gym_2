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
import random
from time import time
import numpy as np
import torch
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from typing import Tuple, Dict
from collections import deque
from legged_gym.envs import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class Zq10Robot(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # self.target_joint_angles = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # for i in range(self.num_dofs):
        #     self.target_joint_angles[i] = self.cfg.init_state.target_joint_angles[i]
        # self.target_joint_angles = self.target_joint_angles.unsqueeze(0)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.init_position = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.body_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.reset_buf2 = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)  # 步态生成器-生成的参考姿势
        self.switch_step_or_stand = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # cmd较小不需要步态的（0,1,1,0...,1,0）
        self.ref_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 步态生成器--计数器
        self.cos_pos = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)  # 每个env当前步态的cos相位。如果步频变化，则可以从这里开始。
        # 观察值的上行delay
        self.obs_history = deque(maxlen=self.cfg.env.queue_len_obs)
        for _ in range(self.cfg.env.queue_len_obs):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_observations, dtype=torch.float, device=self.device))
        # action的下行delay
        self.action_history = deque(maxlen=self.cfg.env.queue_len_act)
        for _ in range(self.cfg.env.queue_len_act):
            self.action_history.append(torch.zeros(
                self.num_envs, self.num_actions, dtype=torch.float, device=self.device))

    #
    def step(self, actions):
        # 从on_policy_runner进来的action，刚从act获取
        # 步态生成
        self.ref_count += 1
        self.compute_reference_states()

        # 下行延迟：延长将action送往扭矩的时间

        # index_act = random.randint(4, 29)
        # if index_act < 10:
        #     action_delayed_0 = self.action_history[0]
        #     action_delayed_1 = self.action_history[1]
        # elif 10 <= index_act < 20:
        #     action_delayed_0 = self.action_history[1]
        #     action_delayed_1 = self.action_history[2]
        # elif index_act < 30:
        #     action_delayed_0 = self.action_history[2]
        #     action_delayed_1 = self.action_history[3]
        #
        #
        # action_delayed = self.action_history[index_act]

        # action_delayed = self.action_history.popleft()
        # self.action_history.append(actions)
        #
        # super().step(action_delayed)
        super().step(actions)
        # 上行延迟：延迟获取obs,但观察到当前帧的action。

        # index_obs = random.randint(4, 29)
        # obs_delayed = self.obs_history[index_obs]
        #
        # obs_delayed = self.obs_history.popleft()
        # self.obs_history.append(torch.clone(self.obs_buf))
        # self.obs_buf[:, 5:self.num_obs-self.num_actions] = obs_delayed[:, 5:self.num_obs-self.num_actions]

        # obs：由当前sin，当前指令，延迟omega、euler、pos、vel，当前action组成。
        # self.obs_buf[:, -self.num_actions:] = actions[:, :]

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_observations(self):
        """ Computes observations
        """
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        # self.dof_vel[:, 4:6] = 0.
        # self.dof_vel[:, 10:12] = 0.
        # print('dof_pos:', list(map(lambda x: "%.4f" % x, self.dof_pos[0])))
        # print('dof_vel:', list(map(lambda x: "%.4f" % x, self.dof_vel[0])))
        # self.actions = torch.cat((
        #     self.cos_pos,
        #     self.cos_pos,
        #     self.cos_pos,
        #     self.cos_pos,
        #     self.cos_pos,
        #     self.cos_pos
        #     ), dim=-1)
        self.obs_buf = torch.cat((
            self.cos_pos,  # 2
            self.commands[:, :3] * self.commands_scale,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz,  # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions #* self.obs_scales.action + self.default_dof_pos#* self.cfg.control.action_scale + self.default_dof_pos,  # 12#改动测试
            ), dim=-1)
        # print(self.base_euler_xyz[0])
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        if torch.isnan(self.obs_buf).any():
            self.obs_buf = torch.nan_to_num(self.obs_buf, nan=0.0001)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        self.ref_count[env_ids] = 0


    def _reset_dofs(self, env_ids):
        if self.cfg.domain_rand.randomize_init_state:
            self.dof_pos[env_ids] = self.default_dof_pos + torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

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
            for index, env_id in enumerate(env_ids):
                if self.cfg.domain_rand.randomize_init_state:
                    self.root_states[env_id, 3:7] = quat_from_euler_xyz(rpy[index, 0], rpy[index, 1], rpy[index, 2])
                self.init_position[env_id, 0:3] = self.root_states[env_id, 0:3]
                self.init_position[env_id, 2] -= 0.1
        #
        # base velocities
        if self.cfg.domain_rand.randomize_init_state:
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.05, 0.05, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        if self.cfg.asset.fix_base_link:
            self.root_states[env_ids, 7:13] = 0
            self.root_states[env_ids, 2] += 1.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:2] = 0.  # cos 2
        noise_vec[2:5] = 0.  # commands 3
        noise_vec[5:8] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel  # 0.2 * 1 * 0.25 = 0.05
        noise_vec[8:11] = noise_scales.gravity * noise_level  # 0.05 * 1. = 0.05

        noise_vec[11:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos  # 0.01 * 1 * 1. = 0.01
        noise_vec[21:31] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel  # 1.5 * 1 * 0.05 = 0.075
        noise_vec[31:41] = 0.  # previous actions
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    def check_termination(self):
        super().check_termination()
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2], dim=1) / 2
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        self.reset_buf2 = base_height < self.cfg.asset.terminate_body_height  # 0.3!!!!!!!!!!!!!!!!!
        self.reset_buf |= self.reset_buf2
        # self.reset_buf2 = self.root_states[:, 2] < self.cfg.asset.terminate_body_height  # 0.3!!!!!!!!!!!!!!!!!
        # self.reset_buf |= self.reset_buf2

    def _resample_commands(self, env_ids):
        # 在reset_index和episode_length_buf==cfg.commands.resampling_time的时候，重新设定指令
        super()._resample_commands(env_ids)
        # set small commands to zero
        stand_ids = (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        # 在这里重置重设指令的步态生成器的初始计数值
        # self.ref_count[env_ids] = 0  # 唐博指出这里有个BUG,在resample command的时候不应该重置步态.

        # 设置所有env的正弦生成标志，如果cmd = 0则不生成正弦步态。
        # self.switch_step_or_stand[:] = 1
        # self.switch_step_or_stand[:] *= torch.norm(self.commands[:, :2], dim=1) > 0.2  # 1=step, 0=stand

    def compute_reference_states(self):
        phase = self.ref_count * self.dt * self.cfg.commands.step_freq * 2.
        # right first
        mask_right = (torch.floor(phase) + 1) % 2
        mask_left = torch.floor(phase) % 2
        # left first
        # mask_right = torch.floor(phase) % 2
        # mask_left = (torch.floor(phase) + 1) % 2
        cos_pos = (1 - torch.cos(2 * torch.pi * phase)) / 2  # 得到一条从0开始增加，频率为step_freq，振幅0～1的曲线，接地比较平滑
        self.cos_pos[:, 0] = cos_pos * mask_right
        self.cos_pos[:, 1] = cos_pos * mask_left

        scale_1 = self.cfg.commands.step_joint_offset
        scale_2 = 2 * scale_1

        self.ref_dof_pos[:, :] = self.default_dof_pos[0, :]
        # right foot stance phase set to default joint pos
        # sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 2] += self.cos_pos[:, 0] * scale_1
        self.ref_dof_pos[:, 3] += -self.cos_pos[:, 0] * scale_2
        self.ref_dof_pos[:, 4] += self.cos_pos[:, 0] * scale_1
        # left foot stance phase set to default joint pos
        # sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 7] += self.cos_pos[:, 1] * scale_1
        self.ref_dof_pos[:, 8] += -self.cos_pos[:, 1] * scale_2
        self.ref_dof_pos[:, 9] += self.cos_pos[:, 1] * scale_1

        # 双足支撑相位
        # self.ref_dof_pos[torch.abs(self.sin_pos[:, 0]) < 0.1, :] = 0. + self.default_dof_pos[0, :]

        # 如果cmd很小，姿态一直为默认姿势，sin相位也为0
        # self.ref_dof_pos[self.switch_step_or_stand == 0, :] = 0. + self.default_dof_pos[0, :]
        # print(self.ref_count[0], self.cos_pos[0, 0], self.cos_pos[0, 1], self.ref_dof_pos[0, [2, 3, 4, 8, 9, 10]])
        # print(self.sin_pos[0, 0], self.ref_dof_pos[0, :])

    # ------------------------ rewards --------------------------------------------------------------------------------
    # ------------------------ rewards --------------------------------------------------------------------------------
    def _reward_dof_acc_ankle(self):
        # Penalize dof accelerations
        # return torch.sum(torch.square((self.last_dof_vel[4, 9] - self.dof_vel[4, 9]) / self.dt), dim=1)
        return torch.square((self.last_dof_vel[:, 4] - self.dof_vel[:, 4]) / self.dt) + torch.square((self.last_dof_vel[:, 9] - self.dof_vel[:, 9]) / self.dt)

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 1.1
        single_contact = torch.sum(1. * contacts, dim=1) == 1
        return 1. * single_contact


    def _reward_target_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        # joint_diff = torch.sum((self.dof_pos - self.target_joint_angles)**2, dim=1)
        joint_diff = torch.sum((self.dof_pos - self.ref_dof_pos) ** 2, dim=1)
        imitate_reward = torch.exp(-7*joint_diff)  # positive reward, not the penalty
        return imitate_reward

    def _reward_tracking_lin_x_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_y_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_body_feet_dist(self):
        # Penalize body root xy diff feet xy
        self.gym.clear_lines(self.viewer)

        # foot_pos = self.rigid_state[:, self.feet_indices, :3]
        # center_pos = torch.mean(foot_pos, dim=1)
        self.body_pos[:, :3] = self.root_states[:, :3]
        # self.body_pos[:, :3] = self.init_position[:, :3]
        # self.body_pos[:, 2] -= 0.75

        pos_dist = torch.norm(self.body_pos[:, :] - self.init_position[:, :], dim=1)

        sphere_geom_1 = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 0, 1))
        sphere_pose_1 = gymapi.Transform(gymapi.Vec3(self.body_pos[0, 0], self.body_pos[0, 1], self.body_pos[0, 2] - 0.7), r=None)
        sphere_pose_2 = gymapi.Transform(gymapi.Vec3(self.init_position[0, 0], self.init_position[0, 1], self.init_position[0, 2] - 0.7), r=None)
        # sphere_pose_2 = gymapi.Transform(gymapi.Vec3(center_pos[0, 0], center_pos[0, 1], center_pos[0, 2]), r=None)

        gymutil.draw_lines(sphere_geom_1, self.gym, self.viewer, self.envs[0], sphere_pose_1)
        gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[0], sphere_pose_2)

        reward = torch.square(pos_dist * 10)
        # print(f'dist={pos_dist[0]}')
        return reward

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penilize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2