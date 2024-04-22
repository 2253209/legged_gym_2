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
        num_observations = 47  # 169
        num_actions = 12
        env_spacing = 1.
        queue_len_obs = 3
        queue_len_act = 3

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]  # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.85]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'JOINT_Y1': -0.0,
            'JOINT_Y2': 0.0,
            'JOINT_Y3': 0.21,
            'JOINT_Y4': -0.53,
            'JOINT_Y5': 0.32,
            'JOINT_Y6': 0.0,

            'JOINT_Z1': 0.0,
            'JOINT_Z2': 0.0,
            'JOINT_Z3': 0.21,
            'JOINT_Z4': -0.53,
            'JOINT_Z5': 0.32,
            'JOINT_Z6': -0.0,
        }
        target_joint_angles = [-0., 0.0, 0.21, -0.53, 0.32, 0.,
                               0., 0.0, 0.21, -0.53, 0.32, -0.]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # stiffness = {'JOINT': 100.0}  # [N*m/rad]
        # damping = {'JOINT': 0.0}
        stiffness = {'JOINT_Y1': 200.0, 'JOINT_Y2': 200.0, 'JOINT_Y3': 200.0, 'JOINT_Y4': 200.0, 'JOINT_Y5': 200.0, 'JOINT_Y6': 200.0,
                     'JOINT_Z1': 200.0, 'JOINT_Z2': 200.0, 'JOINT_Z3': 200.0, 'JOINT_Z4': 200.0, 'JOINT_Z5': 200.0, 'JOINT_Z6': 200.0,
                     }  # [N*m/rad]
        damping = {'JOINT_Y1': 5.0, 'JOINT_Y2': 5.0, 'JOINT_Y3': 5.0, 'JOINT_Y4': 5.0, 'JOINT_Y5': 3.0, 'JOINT_Y6': 3.0,
                   'JOINT_Z1': 5.0, 'JOINT_Z2': 5.0, 'JOINT_Z3': 5.0, 'JOINT_Z4': 5.0, 'JOINT_Z5': 3.0, 'JOINT_Z6': 3.0,
                   }  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.05
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 2

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        gravity = [0., 0., -9.81]  # [m/s^2]
        # gravity = [0., 0., 0]  # [m/s^2]

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [-3, -3, 3]  # [m]
        lookat = [0., 0, 1.]  # [m]

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class commands(LeggedRobotCfg.commands):
        step_joint_offset = 0.30  # rad
        step_freq = 1.5  # HZ （e.g. cycle-time=0.66）

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.1, 0.3]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-3.14, 3.14]
            # lin_vel_x = [-0.0, 0.0]  # min max [m/s]
            # lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            # ang_vel_yaw = [-0.0, 0.0]  # min max [rad/s]
            # heading = [-0, 0]

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
        disable_gravity = False
        fix_base_link = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.8, 1.2]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.5
        randomize_init_state = True

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = True
        base_height_target = 0.83
        tracking_sigma = 0.15
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

            termination = -5.  # 4. 不倒
            tracking_lin_vel = 0.
            tracking_lin_x_vel = 1.0  # 6. 奖励速度为0
            tracking_lin_y_vel = 1.0  # 6. 奖励速度为0
            tracking_ang_vel = 1.0
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            orientation = -0.0  # 5. 重力投影
            #
            action_smoothness = -0.002
            torques = -1.e-5
            dof_vel = -0.0
            dof_acc = -1.e-6
            #
            base_height = -0.0  # 1.奖励高度？惩罚高度方差
            feet_air_time = 0.
            collision = -0.1
            dof_pos_limits = -1.  # 让各个关节不要到达最大位置
            #
            feet_stumble = -0.0
            feet_contact_forces = -0.
            #
            action_rate = -0.01
            stand_still = -0.  # 3. 惩罚：0指令运动。关节角度偏离 初始值
            no_fly = 0.0  # 2. 奖励：两脚都在地上，有一定压力
            target_joint_pos = 10.0  # 3. 惩罚 身体关节角度 偏离
            # body_feet_dist = -1.0





class Zq12CfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'zq12'
        max_iterations = 10000
        # logging
        save_interval = 400
        # checkpoint = '8400'
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
