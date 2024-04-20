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
from deploy.utils.ankle_joint_converter import convert_pv_joint_2_ori, convert_p_ori_2_joint
from deploy.utils.logger import SimpleLogger, get_title_long
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
        self.obs_net = np.zeros((1, 47), dtype=np.float32)
        self.obs_real = np.zeros((1, 36), dtype=np.float32)

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

    def combine_obs_net(self, cos_pos, omega, eu_ang, pos_net, vel_net, action_net):
        self.obs_net[0, :2] = cos_pos[0, :2]
        self.obs_net[0, 2] = self.cfg.cmd.vx * self.cfg.normalization.obs_scales.lin_vel  # 1
        self.obs_net[0, 3] = self.cfg.cmd.vy * self.cfg.normalization.obs_scales.lin_vel  # 1
        self.obs_net[0, 4] = self.cfg.cmd.dyaw * self.cfg.normalization.obs_scales.ang_vel  # 1
        self.obs_net[0, 5:8] = omega[:3] * self.cfg.normalization.obs_scales.ang_vel  # 3
        self.obs_net[0, 8:11] = eu_ang[:3] * self.cfg.normalization.obs_scales.quat  # 3
        self.obs_net[0, 11:23] = (pos_net - self.cfg.env.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos  # 10
        self.obs_net[0, 23:35] = vel_net[:] * self.cfg.normalization.obs_scales.dof_vel  # 10
        self.obs_net[0, 35:47] = action_net[:]  # 10

    def combine_obs_real(self, pos_real, vel_real, action_real):
        self.obs_real[0, 0:12] = pos_real[:]
        self.obs_real[0, 12:24] = vel_real[:]
        self.obs_real[0, 24:36] = action_real[:]

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """
        Calculates torques from position commands
        """

        return (target_q - q) * kp + (target_dq - dq) * kd


    def run_robot(self, policy):
        obs = np.zeros((1,self.cfg.env.num_single_obs), dtype=np.float32)
        action = np.zeros(self.cfg.env.num_actions, dtype=np.double)
        action[:] = self.cfg.env.default_dof_pos[:]
        q_last = np.zeros_like(action)
        q_zero = np.zeros_like(action)
        target_q = np.zeros_like(action)
        target_q2 = np.zeros_like(action)
        phase = np.zeros((1, 1), dtype=np.float32)
        cos_pos = np.zeros((1, 2), dtype=np.float32)
        kp = np.zeros(self.cfg.env.num_actions)
        kd = np.zeros(self.cfg.env.num_actions)

        current_time = time.time()

        # start thread receiving robot state
        es = StateEstimator(self.lc)
        es.spin()

        key_comm = KeyCommand()
        key_comm.start()
        count_total = 0
        count_max_merge = 100

        act_gen = ActionGenerator(self.cfg)

        sp_logger = SimpleLogger(f'{LEGGED_GYM_ROOT_DIR}/logs/dep_log', get_title_long())

        try:
            for i in range(10):
                policy(torch.tensor(self.obs_net))[0].detach().numpy()

            while key_comm.listening:
                c_delay = time.time() - current_time
                s_delay = self.cfg.env.dt - c_delay
                # print(f'c_delay: {c_delay} s_delay={s_delay} ')
                time.sleep(max(s_delay, 0))
                frq = time.time() - current_time
                if key_comm.timestep % 100 == 0:
                    print(f'frq: {1 / frq} Hz count={key_comm.timestep}')
                current_time = time.time()

                phase[0, 0] = (key_comm.timestep * self.cfg.env.dt * self.cfg.env.step_freq) * 2.
                mask_right = (np.floor(phase) + 1) % 2
                mask_left = np.floor(phase) % 2
                cos_values = (1 - np.cos(2 * np.pi * phase)) / 2  # 得到一条从0开始增加，频率为step_freq，振幅0～1的曲线，接地比较平滑
                cos_pos[0, 0] = cos_values * mask_right
                cos_pos[0, 1] = cos_values * mask_left

                obs[0, :2] = cos_pos[0, :]
                obs[0, 2:35] = 0.
                obs[0, -12:] = action[:]

                # 将obs写入文件，在桌面
                sp_logger.save(np.concatenate((obs, self.obs_real), axis=1), count_total, frq)

                if key_comm.timestep == 0:
                    # action_last[:] = action[:]
                    q_zero[:] = 0.

                if key_comm.stepCalibrate:
                    # 当状态是“静态归零模式”时：将所有电机缓慢置于初始位置。
                    action[:] = self.cfg.env.default_dof_pos[:] * (1.0 // self.cfg.env.action_scale)

                elif key_comm.stepTest:
                    # 当状态是“挂起动腿模式”时：使用动作发生器，生成腿部动作
                    action[:] = 0.

                elif key_comm.stepNet:
                    # 当状态是“神经网络模式”时：使用神经网络输出动作
                    action[:] = policy(torch.tensor(obs))[0].detach().numpy()

                else:
                    print('退出')

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
        step_freq = 2  # Hz
        num_single_obs = 47  # 2+3+3+3+10+10+10+1

        action_scale = 0.25
        cycle_time = 1.0
        num_actions = 12


        default_dof_pos = np.array([-0.0, 0.0, 0.21, -0.53, 0.32, 0.0,
                                   0.0, 0.0, 0.21, -0.53, 0.32, -0.0], dtype=np.float32)

        joint_limit_min = np.array([-0.5, -0.25, -1.15, -2.2, -0.5, -0.8, -0.5, -0.28, -1.15, -2.2, -0.8, -0.5], dtype=np.float32)
        joint_limit_max = np.array([0.5, 0.25, 1.15, -0.05, 0.8, 0.5, 0.5, 0.28, 1.15, -0.05, 0.5, 0.8], dtype=np.float32)

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100.

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
        args.load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/zq12/exported/policies/policy_4-20-5200.pt'
    policy = torch.jit.load(args.load_model)
    deploy = Deploy(DeployCfg(), args.load_model)
    deploy.run_robot(policy)
