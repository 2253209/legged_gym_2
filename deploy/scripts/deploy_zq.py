# SPDX-License-Identifier: BSD-3-Clause
#
# qin_jian zq tech


import math
import numpy as np
from collections import deque
from isaacgym import torch_utils
import torch
import time
import lcm

from deploy.lcm_types.pd_targets_lcmt import pd_targets_lcmt
from deploy.utils.state_estimator import StateEstimator
from deploy.utils.act_gen import ActionGenerator
from deploy.utils.ankle_joint_converter import decouple, forward_kinematics
from deploy.utils.logger import SimpleLogger
from deploy.utils.key_command import KeyCommand
from legged_gym import LEGGED_GYM_ROOT_DIR


def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    # Returns roll, pitch, yaw in a NumPy array in radians
    eu_ang = np.array([roll_x, pitch_y, yaw_z])
    eu_ang[eu_ang > math.pi] -= 2 * math.pi
    return eu_ang


class Deploy:
    def __init__(self, cfg, path):
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self.cfg = cfg
        self.log_path = path

    def publish_action(self, action, kp, kd):
        command_for_robot = pd_targets_lcmt()

        command_for_robot.q_des = action
        command_for_robot.q_des[10] = -action[10]  # 左脚长电机，在输出给机器人的时候，才将脚部电机的值转换成真实正负号
        command_for_robot.q_des[11] = -action[11]  # 左脚短电机
        command_for_robot.qd_des = np.zeros(self.cfg.env.num_actions)
        command_for_robot.kp = kp
        command_for_robot.kd = kd

        command_for_robot.tau_ff = np.zeros(self.cfg.env.num_actions)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        # 由lcm将神经网络输出的action传入c++ sdk
        self.lc.publish("robot_command", command_for_robot.encode())

    def get_obs(self, es):
        """
        Extracts an observation from the mujoco data structure
        """
        q = es.joint_pos.astype(np.double)
        q[10] = -q[10]  # 左脚长电机，在接收电机数值的时候，真实电机数值转化成模型正负号
        q[11] = -q[11]  # 左脚短电机
        dq = es.joint_vel.astype(np.double)
        dq[10] = -dq[10]
        dq[11] = -dq[11]
        quat = es.quat[[1, 2, 3, 0]].astype(np.double)
        # r = R.from_quat(quat)
        # v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
        omega = es.omegaBody[[0, 1, 2]].astype(np.double)
        # gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        eu_ang = quaternion_to_euler_array(quat)

        return q, dq, eu_ang, omega

    def combine_obs(self, obs, omega, eu_ang, q, dq, action):
        # self.base_ang_vel * self.obs_scales.ang_vel,  # 3
        # self.base_euler_xyz,  # 3
        # self.commands[:, :3] * self.commands_scale,  # 3
        # self.dof_pos * self.obs_scales.dof_pos,  # 10
        # self.dof_vel * self.obs_scales.dof_vel,  # 10
        # self.actions  # 10

        obs[0, 0:3] = omega * self.cfg.normalization.obs_scales.ang_vel  # 3
        obs[0, 3:6] = eu_ang * self.cfg.normalization.obs_scales.quat  # 3
        obs[0, 6] = self.cfg.cmd.vx * self.cfg.normalization.obs_scales.lin_vel  # 1
        obs[0, 7] = self.cfg.cmd.vy * self.cfg.normalization.obs_scales.lin_vel  # 1
        obs[0, 8] = self.cfg.cmd.dyaw * self.cfg.normalization.obs_scales.ang_vel  # 1
        obs[0, 9:19] = q[self.cfg.env.net_index] * self.cfg.normalization.obs_scales.dof_pos  # 10
        obs[0, 19:29] = dq[self.cfg.env.net_index] * self.cfg.normalization.obs_scales.dof_vel  # 10
        obs[0, 29:39] = action[self.cfg.env.net_index]  # 10

        obs = np.clip(obs, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)

    def combine_total_obs(self, total_data, omega, eu_ang, q, dq, action, target_q):
        total_data[0, 0:3] = omega * self.cfg.normalization.obs_scales.ang_vel  # 3
        total_data[0, 3:6] = eu_ang * self.cfg.normalization.obs_scales.quat  # 3
        total_data[0, 6] = self.cfg.cmd.vx * self.cfg.normalization.obs_scales.lin_vel  # 1
        total_data[0, 7] = self.cfg.cmd.vy * self.cfg.normalization.obs_scales.lin_vel  # 1
        total_data[0, 8] = self.cfg.cmd.dyaw * self.cfg.normalization.obs_scales.ang_vel  # 1
        total_data[0, 9:19] = q[self.cfg.env.net_index] * self.cfg.normalization.obs_scales.dof_pos  # 10
        total_data[0, 19:29] = dq[self.cfg.env.net_index] * self.cfg.normalization.obs_scales.dof_vel  # 10
        total_data[0, 29:39] = action[self.cfg.env.net_index]  # 10
        total_data[0, 39:51] = q[:]
        total_data[0, 51:63] = dq[:]
        total_data[0, 63:75] = target_q[:]


    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """
        Calculates torques from position commands
        """

        return (target_q - q) * kp + (target_dq - dq) * kd


    def run_robot(self, policy):

        action = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        action[:] = self.cfg.env.default_dof_pos[:]
        q_last = np.zeros_like(action)
        target_q = np.zeros_like(action)

        kp = np.zeros(self.cfg.env.num_actions)
        kd = np.zeros(self.cfg.env.num_actions)

        obs = np.zeros((1, self.cfg.env.num_single_obs), dtype=np.float32)
        total_data = np.zeros((1, 75), dtype=np.float32)  # 39+36

        current_time = time.time()

        # start thread receiving robot state
        es = StateEstimator(self.lc)
        es.spin()

        key_comm = KeyCommand()
        key_comm.start()
        count_max_merge = 100

        act_gen = ActionGenerator(self.cfg)

        sp_logger = SimpleLogger(f'{LEGGED_GYM_ROOT_DIR}/logs/dep_log')

        try:
            for i in range(10):
                policy(torch.tensor(obs))[0].detach().numpy()

            while key_comm.listening:
                c_delay = time.time() - current_time
                s_delay = self.cfg.env.dt - c_delay
                # print(f'c_delay: {c_delay} s_delay={s_delay} ')
                time.sleep(max(s_delay, 0))
                frq = time.time() - current_time
                if key_comm.timestep % 100 == 0:
                    print(f'frq: {1 / frq} Hz count={key_comm.timestep}')
                current_time = time.time()

                # Obtain an observation
                q, dq, eu_ang, omega = self.get_obs(es)

                self.combine_obs(obs, omega, eu_ang, q, dq, action)
                self.combine_total_obs(total_data, omega, eu_ang, q, dq, action, target_q)

                # 将obs写入文件，在桌面
                sp_logger.save_75(total_data, key_comm.timestep, frq)  # total_obs len 75

                if key_comm.timestep == 0:
                    # action_last[:] = action[:]
                    q_last[:] = q[:]

                if key_comm.stepCalibrate:
                    # 当状态是“静态归零模式”时：将所有电机缓慢置于初始位置。
                    action[:] = self.cfg.env.default_dof_pos[:] * (1.0 // self.cfg.env.action_scale)
                    target_q[:] = self.cfg.env.default_dof_pos[:]
                    kp[:] = self.cfg.robot_config.kps_stand[:]
                    kd[:] = self.cfg.robot_config.kds_stand[:]

                elif key_comm.stepTest:
                    # 当状态是“挂起动腿模式”时：使用动作发生器，生成腿部动作
                    # action[:] = np.array(act_gen.step()) / self.cfg.env.action_scale
                    # kp[:] = self.cfg.robot_config.kps[:]
                    # kd[:] = self.cfg.robot_config.kds[:]
                    pass
                elif key_comm.stepNet:
                    # 当状态是“神经网络模式”时：使用神经网络输出动作
                    a_temp = policy(torch.tensor(obs))[0].detach().numpy()
                    # print(f'net[2]={a_temp[2]} ', end='')
                    action[0:5] = a_temp[0:5]
                    action[5] = -action[4]
                    action[6:11] = a_temp[5:10]
                    action[11] = -a_temp[9]
                    kp[:] = self.cfg.robot_config.kps[:]
                    kd[:] = self.cfg.robot_config.kds[:]
                    target_q = action * self.cfg.env.action_scale
                    target_q = np.clip(target_q, self.cfg.env.joint_limit_min, self.cfg.env.joint_limit_max)

                else:
                    print('退出')

                # 插值
                if key_comm.timestep < count_max_merge:
                    target_q[:] = (q_last[:] / count_max_merge * (count_max_merge - key_comm.timestep - 1)
                                 + target_q[:] / count_max_merge * (key_comm.timestep + 1))

                # print(f'action[2]={action[2]}, action_scaled[2]={action_scaled[2]},')
                # action = np.clip(action,
                #                  self.cfg.env.joint_limit_min,
                #                  self.cfg.env.joint_limit_max)
                # target_q[:] = action_scaled[:]

                # 将神经网络生成的，左右脚的pitch、row位置，映射成关节电机角度
                # my_joint_left, _ = decouple(target_q[5], target_q[4], "left")
                # my_joint_right, _ = decouple(target_q[11], target_q[10], "right")

                # target_q[4] = -my_joint_left[0]
                # target_q[5] = my_joint_left[1]
                # target_q[10] = my_joint_right[0]
                # target_q[11] = -my_joint_right[1]

                # target_dq = np.zeros(self.cfg.env.num_actions, dtype=np.double)
                # Generate PD control
                # tau = self.pd_control(target_q, q, self.cfg.robot_config.kps,
                #                       target_dq, dq, self.cfg.robot_config.kds)  # Calc torques
                # tau = np.clip(tau, -self.cfg.robot_config.tau_limit, self.cfg.robot_config.tau_limit)  # Clamp torques

                # !!!!!!!! send target_q to lcm
                if key_comm.stepCalibrate:
                    self.publish_action(target_q, kp, kd)
                elif key_comm.stepNet:
                    self.publish_action(target_q, kp, kd)
                    # pass
                else:
                    pass
                key_comm.timestep += 1

        except KeyboardInterrupt:
            print(f'用户终止。')
        finally:
            print(f'count={key_comm.timestep}')
            es.close()
            key_comm.stop()
            sp_logger.close()


class DeployCfg:

    class env:
        dt = 0.01
        num_single_obs = 39  # 3+3+3+10+10+10
        action_scale = 0.1
        cycle_time = 1.0
        num_actions = 12
        num_net = 10

        net_index = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
        default_dof_pos = np.array([0., 0., 0.2, -0.53, 0.4, -0.4,
                                    0., 0., 0.2, -0.53, 0.4, -0.4], dtype=np.float32)
        # default_dof_pos = [0., 0., 0., 0., 0., 0.,
        #                    0., 0., 0., 0., 0., 0.]

        joint_limit_min = np.array([-0.5, -0.25, -1.15, -2.2, -0.6, -0.6, -0.5, -0.28, -1.15, -2.2, -0.6, -0.6], dtype=np.float32)
        joint_limit_max = np.array([0.5, 0.25, 1.15, -0.05, 0.6, 0.6, 0.5, 0.28, 1.15, -0.05, 0.6, 0.6], dtype=np.float32)

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0

        clip_observations = 18.
        clip_actions = 18.

    class cmd:
        vx = 0.0  # 0.5
        vy = 0.0  # 0.
        dyaw = 0.0  # 0.05

    class robot_config:
        kps = np.array([200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200], dtype=np.double)
        kds = np.array([10, 10, 10, 10, 4, 4, 10, 10, 10, 10, 4, 4], dtype=np.double)

        kps_stand = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100], dtype=np.double)
        kds_stand = np.array([10, 10, 10, 5, 2, 2, 10, 10, 10, 5, 2, 2], dtype=np.double)

        # tau_limit = 200. * np.ones(10, dtype=np.double)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=False,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    if not args.load_model:
        args.load_model = '/home/qin/Desktop/legged_gym_2/logs/zq01/exported/policies/policy_1.pt'
    policy = torch.jit.load(args.load_model)
    deploy = Deploy(DeployCfg(), args.load_model)
    deploy.run_robot(policy)
