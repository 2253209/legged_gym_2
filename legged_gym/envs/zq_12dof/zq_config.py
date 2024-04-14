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
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Zq12Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 45  # 169
        num_actions = 12
        env_spacing = 1.

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4,
                             0.5]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.855]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'JOINT_Y1': -0.1,
            'JOINT_Y2': 0.0,
            'JOINT_Y3': 0.25,
            'JOINT_Y4': -0.53,
            'JOINT_Y5': 0.3,
            'JOINT_Y6': 0.1,

            'JOINT_Z1': 0.1,
            'JOINT_Z2': 0.0,
            'JOINT_Z3': 0.25,
            'JOINT_Z4': -0.53,
            'JOINT_Z5': 0.3,
            'JOINT_Z6': -0.1,
        }
        target_joint_angles = [0.0, 0.0, 0.25, -0.53, 0.3, 0.0,
                               0.0, 0.0, 0.25, -0.53, 0.3, 0.0]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'JOINT_Y1': 200.0, 'JOINT_Y2': 200.0, 'JOINT_Y3': 200.0, 'JOINT_Y4': 200.0, 'JOINT_Y5': 100.0, 'JOINT_Y6': 100.0,
                     'JOINT_Z1': 200.0, 'JOINT_Z2': 200.0, 'JOINT_Z3': 200.0, 'JOINT_Z4': 200.0, 'JOINT_Z5': 100.0, 'JOINT_Z6': 100.0,
                     }  # [N*m/rad]
        damping = {'JOINT_Y1': 10.0, 'JOINT_Y2': 10.0, 'JOINT_Y3': 10.0, 'JOINT_Y4': 10.0, 'JOINT_Y5': 2.0, 'JOINT_Y6': 2.0,
                   'JOINT_Z1': 10.0, 'JOINT_Z2': 10.0, 'JOINT_Z3': 10.0, 'JOINT_Z4': 10.0, 'JOINT_Z5': 2.0, 'JOINT_Z6': 2.0,
                   }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        gravity = [0., 0., -9.81]  # [m/s^2]

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [-3, -3, 3]  # [m]
        lookat = [0., 0, 1.]  # [m]

    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            # lin_vel_x = [-0.3, 1.0]  # min max [m/s]
            # lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            # ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            # heading = [-0.3, 0.3]
            lin_vel_x = [-0.0, 0.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]  # min max [rad/s]
            heading = [-0.0, 0.0]

    class asset(LeggedRobotCfg.asset):
        # file = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/zq01/mjcf/zq_box_foot.xml'
        file = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/zq01/urdf/zq_box_foot.urdf'
        name = "zq01"
        foot_name = 'foot'
        penalize_contacts_on = ['3', '4']
        terminate_after_contacts_on = []
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        terminate_body_height = 0.4

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.8, 1.2]
        randomize_base_mass = False
        added_mass_range = [-5., 5.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        base_height_target = 0.8
        class scales(LeggedRobotCfg.rewards.scales):
            # termination = -200.
            # tracking_ang_vel = 1.0
            # torques = -5.e-6
            # dof_acc = -2.e-7
            # lin_vel_z = -0.5
            # feet_air_time = 5.
            # dof_pos_limits = -1.
            # no_fly = 0.25
            # dof_vel = -0.0
            # ang_vel_xy = -0.0
            # feet_contact_forces = -0.

            termination = -200.  # 4. 不倒
            tracking_lin_vel = 1.0  # 6. 奖励速度为0
            tracking_ang_vel = 0.0
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            orientation = -1.0  # 5. 重力投影
            #
            torques = -5.e-6
            dof_vel = -0.0
            dof_acc = -2.e-6
            #
            base_height = -2.1  # 1.奖励高度？惩罚高度方差
            feet_air_time = 0.
            collision = -0.1
            dof_pos_limits = -0.
            #
            feet_stumble = -0.0
            feet_contact_forces = -0.
            #
            action_rate = -0.
            stand_still = -1.  # 3. 惩罚：0指令运动。关节角度偏离 初始值
            no_fly = 2.25  # 2. 奖励：两脚都在地上，有一定压力
            # target_joint_pos = 1.1  # 3. 惩罚 身体关节角度 偏离
            # body_feet_dist = -1.0




class Zq12CfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'zq12'
        max_iterations = 3000
        # logging
        save_interval = 100

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
