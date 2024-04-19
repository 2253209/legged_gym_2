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
from deploy.utils.ankle_joint_converter import convert_p_joint_2_ori, convert_p_ori_2_joint
from deploy.utils.logger import SimpleLogger, get_title_82
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
        dq = es.joint_vel.astype(np.double)
        quat = es.quat[[1, 2, 3, 0]].astype(np.double)
        # r = R.from_quat(quat)
        # v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
        omega = es.omegaBody[[0, 1, 2]].astype(np.double)
        # gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        eu_ang = quaternion_to_euler_array(quat)
        return q, dq, eu_ang, omega

    def combine_obs(self, obs, omega, eu_ang, q, dq, action, sin_pos):
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
        obs[0, 9:21] = (q - self.cfg.env.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos  # 10
        obs[0, 21:33] = dq * self.cfg.normalization.obs_scales.dof_vel  # 10
        obs[0, 33:45] = action  # 10
        obs[0, 45] = sin_pos[0, 0]


    def combine_total_obs(self, total_data, omega, eu_ang, q, dq, action, target_q, sin_pos):
        total_data[0, 0:3] = omega * self.cfg.normalization.obs_scales.ang_vel  # 3
        total_data[0, 3:6] = eu_ang * self.cfg.normalization.obs_scales.quat  # 3
        total_data[0, 6] = self.cfg.cmd.vx * self.cfg.normalization.obs_scales.lin_vel  # 1
        total_data[0, 7] = self.cfg.cmd.vy * self.cfg.normalization.obs_scales.lin_vel  # 1
        total_data[0, 8] = self.cfg.cmd.dyaw * self.cfg.normalization.obs_scales.ang_vel  # 1
        total_data[0, 9:21] = (q - self.cfg.env.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos  # 10
        total_data[0, 21:33] = dq * self.cfg.normalization.obs_scales.dof_vel  # 10
        total_data[0, 33:45] = action  # 10
        total_data[0, 45] = sin_pos[0, 0]
        total_data[0, 45:57] = q - self.cfg.env.default_dof_pos
        total_data[0, 57:69] = dq
        total_data[0, 69:81] = target_q
        total_data[0, 81] = sin_pos[0, 0]

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """
        Calculates torques from position commands
        """

        return (target_q - q) * kp + (target_dq - dq) * kd


    def run_robot(self, policy):

        action = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        action[:] = self.cfg.env.default_dof_pos[:]
        q_last = np.zeros_like(action)
        q_zero = np.zeros_like(action)
        target_q = np.zeros_like(action)
        target_q2 = np.zeros_like(action)
        phase = torch.tensor([[0.]])

        kp = np.zeros(self.cfg.env.num_actions)
        kd = np.zeros(self.cfg.env.num_actions)

        obs = np.zeros((1, self.cfg.env.num_single_obs), dtype=np.float32)
        total_data = np.zeros((1, 82), dtype=np.float32)  # 39+36

        current_time = time.time()

        # start thread receiving robot state
        es = StateEstimator(self.lc)
        es.spin()

        key_comm = KeyCommand()
        key_comm.start()
        count_total = 0
        count_max_merge = 100

        act_gen = ActionGenerator(self.cfg)

        sp_logger = SimpleLogger(f'{LEGGED_GYM_ROOT_DIR}/logs/dep_log', get_title_82())

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

                phase[0, 0] = (key_comm.timestep * self.cfg.env.dt * self.cfg.env.step_freq) % 1.
                sin_pos = torch.sin(2 * torch.pi * phase)

                # Obtain an observation
                q, dq, eu_ang, omega = self.get_obs(es)
                q = np.clip(q, self.cfg.env.joint_limit_min, self.cfg.env.joint_limit_max)  # 过滤掉比较大的值
                # 将观察得到的脚部电机位置转换成神经网络可以接受的ori位置
                try:
                    # print(q[4], q[5], q[10], q[11] )
                    q[4], q[5], q[10], q[11] = convert_p_joint_2_ori(q[4], q[5], q[10], q[11])
                    q_last[:] = q[:]
                except Exception as e:
                    print(q[4], q[5], q[10], q[11])
                    print(e)
                    q[:] = q_last[:]

                # 脚踝电机观测速度=0
                # dq[4:6] = 0.
                # dq[10:12] = 0.
                self.combine_obs(obs, omega, eu_ang, q, dq, action, sin_pos)
                obs = np.clip(obs, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)

                self.combine_total_obs(total_data, omega, eu_ang, q, dq, action, target_q, sin_pos)

                # 将obs写入文件，在桌面
                sp_logger.save(total_data, count_total, frq)  # total_obs.shape=75

                if key_comm.timestep == 0:
                    # action_last[:] = action[:]
                    q_zero[:] = q[:]

                if key_comm.stepCalibrate:
                    # 当状态是“静态归零模式”时：将所有电机缓慢置于初始位置。
                    action[:] = self.cfg.env.default_dof_pos[:] * (1.0 // self.cfg.env.action_scale)
                    target_q[:] = self.cfg.env.default_dof_pos[:]
                    kp[:] = self.cfg.robot_config.kps_stand[:]
                    kd[:] = self.cfg.robot_config.kds_stand[:]

                elif key_comm.stepTest:
                    # 当状态是“挂起动腿模式”时：使用动作发生器，生成腿部动作
                    action[:] = 0.
                    target_q[:] = 0.
                    kp[:] = self.cfg.robot_config.kps_stand[:]
                    kd[:] = self.cfg.robot_config.kds_stand[:]

                elif key_comm.stepNet:
                    # 当状态是“神经网络模式”时：使用神经网络输出动作
                    action[:] = policy(torch.tensor(obs))[0].detach().numpy()
                    # print(f'net[2]={a_temp[2]} ', end='')

                    kp[:] = self.cfg.robot_config.kps[:]
                    kd[:] = self.cfg.robot_config.kds[:]
                    # target_q = action * self.cfg.env.action_scale
                    target_q[:] = action * self.cfg.env.action_scale + self.cfg.env.default_dof_pos

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
                target_q = np.clip(target_q, self.cfg.env.joint_limit_min, self.cfg.env.joint_limit_max)
                # 将神经网络生成的，左右脚的pitch、row位置，映射成关节电机角度
                target_q2[:] = target_q[:]

                target_q2[4], target_q2[5], target_q2[10], target_q2[11] =\
                    convert_p_ori_2_joint(target_q[4], target_q[5], target_q[10], target_q[11])
                target_q[:] = target_q2[:]
                # target_dq = np.zeros(self.cfg.env.num_actions, dtype=np.double)
                # Generate PD control
                # tau = self.pd_control(target_q, q, self.cfg.robot_config.kps,
                #                       target_dq, dq, self.cfg.robot_config.kds)  # Calc torques
                # tau = np.clip(tau, -self.cfg.robot_config.tau_limit, self.cfg.robot_config.tau_limit)  # Clamp torques

                # !!!!!!!! send target_q to lcm
                if key_comm.stepCalibrate:
                    self.publish_action(target_q2, kp, kd)
                    # pass
                elif key_comm.stepNet:
                    self.publish_action(target_q2, kp, kd)
                    # pass
                else:
                    pass

                key_comm.timestep += 1
                count_total += 1

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
        step_freq = 1.5  # Hz
        num_single_obs = 46  # 3+3+3+10+10+10+1
        action_scale = 0.25
        cycle_time = 1.0
        num_actions = 12


        default_dof_pos = np.array([-0.1, 0.0, 0.21, -0.53, 0.32, 0.1,
                                   0.1, 0.0, 0.21, -0.53, 0.32, -0.1], dtype=np.float32)

        joint_limit_min = np.array([-0.5, -0.25, -1.15, -2.2, -0.5, -0.8, -0.5, -0.28, -1.15, -2.2, -0.8, -0.5], dtype=np.float32)
        joint_limit_max = np.array([0.5, 0.25, 1.15, -0.05, 0.8, 0.5, 0.5, 0.28, 1.15, -0.05, 0.5, 0.8], dtype=np.float32)

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

        kps_stand = np.array([200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200], dtype=np.double)
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
        args.load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/zq12/exported/policies/policy_tabu.pt'
    policy = torch.jit.load(args.load_model)
    deploy = Deploy(DeployCfg(), args.load_model)
    deploy.run_robot(policy)
