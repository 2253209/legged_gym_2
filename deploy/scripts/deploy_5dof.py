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
from deploy.utils.logger import SimpleLogger, get_title_5dof_deploy
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


class DeployCfg:

    class env:
        # dt = 0.01
        dt = 0.01
        # step_freq = 1.  # Hz
        step_freq = 1.5  # Hz

        num_actions = 12
        num_obs_net = 41  # 2+3+3+3+12+12+12
        num_obs_robot = 60  # 12+12+12+12
        action_scale = 0.1
        action_scale_min = action_scale
        action_scale_max = 0.04
        switch_action = 200
        low_pass_rate = 1.0
        # 神经网络默认初始状态
        default_dof_pos = np.array([-0.0, 0.0, 0.21, -0.53, 0.31,
                                    0.0, 0.0, 0.21, -0.53, 0.31], dtype=np.float32)
        # 真机默认初始状态
        default_joint_pos = np.array([-0.0, 0.0, 0.21, -0.53, 0.31, -0.31,
                                      0.0, 0.0, 0.21, -0.53, -0.31, 0.31
                                      ], dtype=np.float32)

        joint_limit_min = np.array([-0.5, -0.25, -1.15, -2.2, -0.7, -0.9,
                                    -0.5, -0.28, -1.15, -2.2, -0.9, -0.7], dtype=np.float32)
        joint_limit_max = np.array([0.5, 0.25, 1.15, -0.05, 0.9, 0.7,
                                    0.5, 0.28, 1.15, -0.05, 0.7, 0.9], dtype=np.float32)

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0

        clip_observations = 100.
        # clip_actions = 100.
        # clip_observations = 50.
        clip_actions = 100.
    class cmd:
        vx = 0.0  # 0.5
        vy = 0.0  # 0.
        dyaw = 0.0  # 0.05

    class robot_config:
        kps = np.array([200, 200, 200, 200, 100, 100, 200, 200, 200, 200, 100, 100], dtype=np.double)
        kds = np.array([10, 10, 10, 10, 2, 2, 10, 10, 10, 10, 2, 2], dtype=np.double)

        kps_stand = np.array([200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200], dtype=np.double)
        kds_stand = np.array([10, 10, 10, 10, 4, 4, 10, 10, 10, 10, 4, 4], dtype=np.double)

        # tau_limit = 200. * np.ones(10, dtype=np.double)

class Deploy:
    def __init__(self, cfg: DeployCfg, path):
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self.cfg = cfg
        self.policy = torch.jit.load(path)
        self.obs_net = np.zeros((1, cfg.env.num_obs_net), dtype=np.float32)
        self.obs_robot = np.zeros((1, cfg.env.num_obs_robot), dtype=np.float32)

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
        omega = es.omegaBody[[0, 1, 2]].astype(np.float32)
        quat = es.quat[[1, 2, 3, 0]].astype(np.float32)
        eu_ang = quaternion_to_euler_array(quat)
        q = es.joint_pos.astype(np.float32)
        dq = es.joint_vel.astype(np.float32)
        tau = es.joint_tau.astype(np.float32)

        return omega, eu_ang, q, dq, tau

    def combine_obs_net(self, phase, omega, eu_ang, pos, vel, action):
        self.obs_net[0, :2] = phase[0, :]
        self.obs_net[0, 2] = self.cfg.cmd.vx * self.cfg.normalization.obs_scales.lin_vel  # 1
        self.obs_net[0, 3] = self.cfg.cmd.vy * self.cfg.normalization.obs_scales.lin_vel  # 1
        self.obs_net[0, 4] = self.cfg.cmd.dyaw * self.cfg.normalization.obs_scales.ang_vel  # 1
        self.obs_net[0, 5:8] = omega[:3] * self.cfg.normalization.obs_scales.ang_vel  # 3
        self.obs_net[0, 8:11] = eu_ang[:3] * self.cfg.normalization.obs_scales.quat  # 3
        self.obs_net[0, 11:21] = (pos - self.cfg.env.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos  # 10
        self.obs_net[0, 21:31] = vel * self.cfg.normalization.obs_scales.dof_vel  # 10
        self.obs_net[0, 31:41] = action  # 10

    def combine_obs_real(self, pos, vel, action, tau_joint_state, tau_joint_cmd):
        self.obs_robot[0, 0:12] = pos
        self.obs_robot[0, 12:24] = vel
        self.obs_robot[0, 24:36] = action
        self.obs_robot[0, 36:48] = tau_joint_state
        self.obs_robot[0, 48:60] = tau_joint_cmd

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """
        Calculates torques from position commands
        """
        return (target_q - q) * kp + (target_dq - dq) * kd

    def run_robot(self):
        # 从真机获取和发送给真机的值
        action_robot = np.zeros(self.cfg.env.num_actions, dtype=np.float32)
        pos_robot = np.copy(action_robot)
        vel_robot = np.copy(action_robot)
        tau_robot = np.copy(action_robot)
        tau_cmd = np.copy(action_robot)

        action_filter = np.copy(action_robot)
        action_filter = self.cfg.env.default_dof_pos[:]

        action_0 = np.copy(action_robot)
        action_1 = np.copy(action_robot)
        # 从神经网络获取和发送给网络的值
        action_net = np.zeros(10, dtype=np.float32)
        pos_net = np.copy(action_net)
        vel_net = np.copy(action_net)
        phase = np.zeros((1, 1), dtype=np.float32)
        cos_pos = np.zeros((1, 2), dtype=np.float32)

        pos_last = np.zeros_like(pos_robot)
        pos_0 = np.zeros_like(pos_robot)

        kp = np.zeros(self.cfg.env.num_actions, dtype=np.float32)
        kd = np.copy(kp)

        current_time = time.time()

        # start thread receiving robot state
        es = StateEstimator(self.lc)
        es.spin()

        key_comm = KeyCommand()
        key_comm.start()
        count_total = 0
        count_max_merge = 30

        sp_logger = SimpleLogger(f'{LEGGED_GYM_ROOT_DIR}/logs/dep_log', get_title_5dof_deploy())

        try:
            for i in range(10):
                self.policy(torch.tensor(self.obs_net))[0].detach().numpy()

            while key_comm.listening:
                c_delay = time.time() - current_time
                s_delay = self.cfg.env.dt - c_delay
                # print(f'c_delay: {c_delay} s_delay={s_delay} ')
                time.sleep(max(s_delay, 0))
                frq = time.time() - current_time
                if key_comm.timestep % 100 == 0:
                    #print(f'frq: {1 / frq} Hz count={key_comm.timestep}')
                    pass
                current_time = time.time()

                # 1. 从真实机器人获取观察值 Obtain an observation from real robot
                omega, eu_ang, pos_robot, vel_robot, tau_robot = self.get_obs(es)
                #pos_robot = np.clip(pos_robot, self.cfg.env.joint_limit_min, self.cfg.env.joint_limit_max)  # 过滤掉超过极限的值

                # 调试,上机时关掉
                # pos_robot[:] = self.cfg.env.default_joint_pos[:]
                # eu_ang[:] = 0.
                # omega[:] = 0.

                # 1.2 当操纵者改变模式时,获取当前关节位置做1秒插值
                # if key_comm.timestep == 0:
                #     pos_0 = pos_robot.copy()
                #     print('Time step = 0, POS COPIED!', pos_0)
                #     key_comm.timestep += 1
                # else:
                #     key_comm.timestep += 1

                # print('All time, POS COPIED!', pos_robot)


                # 2.1 POS和VEL转换: 从真实脚部电机位置 转换成神经网络可以接受的ori位置
                pos_net[0:5] = pos_robot[0:5]
                pos_net[5:9] = pos_robot[6:10]
                pos_net[9] = pos_robot[11]
                vel_net[0:5] = vel_robot[0:5]
                vel_net[5:9] = vel_robot[6:10]
                vel_net[9] = vel_robot[11]

                # 2.2 步态生成
                phase[0, 0] = (key_comm.timestep * self.cfg.env.dt * self.cfg.env.step_freq) * 2.
                # phase[0, 0] += 0.1
                mask_right = (np.floor(phase) + 1) % 2
                mask_left = np.floor(phase) % 2
                cos_values = (1 - np.cos(2 * np.pi * phase)) / 2  # 得到一条从0开始增加，频率为step_freq，振幅0～1的曲线，接地比较平滑
                cos_pos[0, 0] = cos_values * mask_right
                cos_pos[0, 1] = cos_values * mask_left

                # 3.1 组合给神经网络的OBS, 其中action_net是上一帧神经网络生成的动作.其他值都经过缩放,P值还减去了默认姿势
                self.combine_obs_net(cos_pos, omega, eu_ang, pos_net, vel_net, action_net)

                # 3.2 组合从真机得到的OBS, 其中action_real是上一帧的关节目标位置,p和v都是原始的.
                self.combine_obs_real(pos_robot, vel_robot, action_robot, tau_robot, tau_cmd)

                # 3.3 !!!限制OBS可能出现的大数值!!!
                self.obs_net = np.clip(self.obs_net, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)

                if key_comm.keyboardEvent:
                    pos_0 = pos_robot.copy()
                    print('Time step = 0, POS COPIED!', pos_0)
                    key_comm.keyboardEvent = False

                if key_comm.stepCalibrate:
                    self.cfg.env.action_scale = self.cfg.env.action_scale_min
                    action_robot = np.array(self.cfg.env.default_joint_pos)

                    kp[:] = self.cfg.robot_config.kps_stand[:]
                    kd[:] = self.cfg.robot_config.kds_stand[:]
                    if key_comm.timestep < count_max_merge:
                        action_robot[:] = (pos_robot[:] / count_max_merge * (count_max_merge - key_comm.timestep)
                                           + action_robot[:] / count_max_merge * key_comm.timestep)

                    self.publish_action(action_robot, kp, kd)
                    # pass
                elif key_comm.stepTest:
                    pass
                elif key_comm.stepNet:
                    # 5.1 将神经网络输出的踝部关节角度,转换成实际电机指令
                    # 这里可能有问题
                    # 当状态是“神经网络模式”时：使用神经网络输出动作。
                    action_net = self.policy(torch.tensor(self.obs_net))[0].detach().numpy()
                    action_0[0:5] = action_net[0:5]
                    action_0[5] = -action_net[4]
                    action_0[6:10] = action_net[5:9]
                    action_0[11] = action_net[9]
                    action_0[10] = -action_net[9]

                    # 裁剪
                    action_1 = np.clip(action_0, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
                    action_1 = action_1 * self.cfg.env.action_scale + self.cfg.env.default_joint_pos

                    kp[:] = self.cfg.robot_config.kps[:]
                    kd[:] = self.cfg.robot_config.kds[:]

                    action_robot = action_1.copy()
                    action_robot = np.clip(action_robot,
                                           self.cfg.env.joint_limit_min,
                                           self.cfg.env.joint_limit_max)

                    # 插值
                    if key_comm.timestep < count_max_merge:
                        action_robot[:] = (pos_robot[:] / count_max_merge * (count_max_merge - key_comm.timestep)
                                           + action_robot[:] / count_max_merge * key_comm.timestep)

                    tau_cmd = self.pd_control(action_robot, pos_robot, kp, np.zeros(12), vel_robot, kd)
                    # print("Joint torque command is: ", tau_cmd)
                    # print(action_robot)
                    self.publish_action(action_robot, kp, kd)
                    if key_comm.timestep > self.cfg.env.switch_action and self.cfg.env.action_scale < self.cfg.env.action_scale_max:
                        self.cfg.env.action_scale += 0.00001*5
                        print("Interpolate action scale", self.cfg.env.action_scale)
                else:
                    pass

                # 3.4 将obs写入文件，在logs/dep_log/下
                sp_logger.save(np.concatenate((self.obs_net.copy(), self.obs_robot.copy()), axis=1), count_total, frq)

                count_total += 1
                key_comm.timestep += 1

        except KeyboardInterrupt:
            print(f'用户终止。')
        finally:
            # print(f'count={key_comm.timestep}')
            es.close()
            key_comm.stop()
            sp_logger.close()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=False,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    if not args.load_model:
        args.load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/zq10/exported/policies/policy_v2.pt'

    deploy = Deploy(DeployCfg(), args.load_model)
    deploy.run_robot()
