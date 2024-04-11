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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class ZqCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 39  # 169
        num_actions = 10
        env_spacing = 1.

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4,
                             0.5]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.850]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'JOINT_Y1': 0.0,
            'JOINT_Y2': 0.0,
            'JOINT_Y3': -0.2,
            'JOINT_Y4': 0.0,
            'JOINT_Y5': 0.1,
            # 'toe_joint_left': -1.57,

            'JOINT_Z1': 0.0,
            'JOINT_Z2': 0.0,
            'JOINT_Z3': -0.2,
            'JOINT_Z4': 0.0,
            'JOINT_Z5': 0.1,
            # 'toe_joint_right': -1.57
        }
        target_joint_angles = [-0.05, -0.1, 0.2, -0.45, 0.25,
                               0.05, 0.1, 0.2, -0.45, 0.25]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'1': 200.0, '2': 200.0, '3': 200.0, '4': 200.0, '5': 50.0
                     }  # [N*m/rad]
        damping = {'1': 5.0, '2': 5.0, '3': 5.0, '4': 5.0, '5': 0.0,
                   }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5

    class sim(LeggedRobotCfg.sim):
        dt = 0.001

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [-3, -3, 3]  # [m]
        lookat = [0., 0, 1.]  # [m]

    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.0, 0.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            heading = [-0.0, 0.0]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/zq01/mjcf/zq_line_foot.xml'
        name = "zq01"
        foot_name = 'foot'
        terminate_after_contacts_on = ['base', '3', '4']
        flip_visual_attachments = False
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter


    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -200.
            tracking_ang_vel = 1.0
            torques = -5.e-5
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            # feet_air_time = 5.
            dof_pos_limits = -1.
            no_fly = 1.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.
            target_joint_pos = 5.0
            tracking_lin_vel = 1.0
            # feet_stumble = -1.0


class ZqCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'zq01'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
