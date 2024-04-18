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
from deploy.utils.ankle_joint_converter import convert_ankle_real_to_net, convert_ankle_net_to_real
from deploy.utils.logger import SimpleLogger, get_title_81
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

    def combine_obs_net(self, obs_net, omega, eu_ang, pos_net, vel_net, action_net):
        obs_net[0, 0:3] = omega * self.cfg.normalization.obs_scales.ang_vel  # 3
        obs_net[0, 3:6] = eu_ang * self.cfg.normalization.obs_scales.quat  # 3
        obs_net[0, 6] = self.cfg.cmd.vx * self.cfg.normalization.obs_scales.lin_vel  # 1
        obs_net[0, 7] = self.cfg.cmd.vy * self.cfg.normalization.obs_scales.lin_vel  # 1
        obs_net[0, 8] = self.cfg.cmd.dyaw * self.cfg.normalization.obs_scales.ang_vel  # 1
        obs_net[0, 9:21] = (pos_net - self.cfg.env.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos  # 10
        obs_net[0, 21:33] = vel_net * self.cfg.normalization.obs_scales.dof_vel  # 10
        obs_net[0, 33:45] = action_net  # 10

    def combine_obs_real(self, obs_real, pos_real, vel_real, action_real):
        obs_real[0, 0:12] = pos_real
        obs_real[0, 12:24] = vel_real
        obs_real[0, 24:36] = action_real

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """
        Calculates torques from position commands
        """

        return (target_q - q) * kp + (target_dq - dq) * kd


    def run_robot(self, policy):
        # 从真机获取和发送给真机的值
        action_real = np.zeros(self.cfg.env.num_actions, dtype=np.float32)
        pos_real = np.zeros_like(action_real)
        vel_real = np.zeros_like(action_real)
        obs_real = np.zeros((1, self.cfg.env.num_obs_real), dtype=np.float32)

        # 从神经网络获取和发送给网络的值
        action_net = np.zeros(self.cfg.env.num_actions, dtype=np.float32)
        pos_net = np.zeros_like(action_net)
        vel_net = np.zeros_like(action_net)
        obs_net = np.zeros((1, self.cfg.env.num_obs_net), dtype=np.float32)

        pos_net_last = np.zeros_like(pos_real)
        pos_real_0 = np.zeros_like(pos_real)

        kp = np.zeros(self.cfg.env.num_actions)
        kd = np.zeros(self.cfg.env.num_actions)

        current_time = time.time()

        # start thread receiving robot state
        es = StateEstimator(self.lc)
        es.spin()

        key_comm = KeyCommand()
        key_comm.start()
        count_max_merge = 100

        sp_logger = SimpleLogger(f'{LEGGED_GYM_ROOT_DIR}/logs/dep_log', get_title_81())

        try:
            for i in range(10):
                policy(torch.tensor(obs_net))[0].detach().numpy()

            while key_comm.listening:
                c_delay = time.time() - current_time
                s_delay = self.cfg.env.dt - c_delay
                # print(f'c_delay: {c_delay} s_delay={s_delay} ')
                time.sleep(max(s_delay, 0))
                frq = time.time() - current_time
                if key_comm.timestep % 100 == 0:
                    print(f'frq: {1 / frq} Hz count={key_comm.timestep}')
                current_time = time.time()

                # 1. 从真实机器人获取观察值 Obtain an observation from real robot
                pos_real, vel_real, eu_ang, omega = self.get_obs(es)

                # 2.1 POS转换: 从真实脚部电机位置 转换成神经网络可以接受的ori位置
                try:
                    # print(q[4], q[5], q[10], q[11] )
                    pos_net[:] = pos_real[:]
                    pos_net[4], pos_net[5], pos_net[10], pos_net[11] = (
                        convert_ankle_real_to_net(pos_real[4], pos_real[5], pos_real[10], pos_real[11]))
                    pos_net = np.clip(pos_net, self.cfg.env.joint_limit_min, self.cfg.env.joint_limit_max)  # 过滤掉超过极限的值
                    pos_net_last[:] = pos_net[:]
                except Exception as e:
                    print(pos_real[4], pos_real[5], pos_real[10], pos_real[11])
                    print(e)
                    pos_net[:] = pos_net_last[:]

                # 2.2 VEL转换: 将脚踝电机观测速度=0
                vel_net[:] = vel_real[:]
                vel_net[4:6] = 0.
                vel_net[10:12] = 0.

                # 3.1 组合真机的OBS, 其中action_real是上一帧的关节目标位置,p和v都是未经缩放的.
                self.combine_obs_real(obs_real, pos_real, vel_real, action_real)

                # 3.2 组合给神经网络的OBS, 其中action_net是上一帧神经网络生成的动作.其他值都经过缩放
                self.combine_obs_net(obs_net, omega, eu_ang, pos_net, vel_net, action_net)
                # 3.3 !!!限制OBS可能出现的大数值!!!
                obs = np.clip(obs_net, -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)

                # 3.4 将obs写入文件，在logs/dep_log/下
                sp_logger.save(np.concatenate((obs_net, obs_real), axis=1), key_comm.timestep, frq)  # total_obs.shape=75

                # 4.1 当操纵者改变模式时,获取当前关节位置做1秒插值
                if key_comm.timestep == 0:
                    pos_real_0[:] = pos_real[:]

                # 4.2 操纵者改变模式
                if key_comm.stepCalibrate:
                    # 当状态是“静态归零模式”时：将所有电机设置初始姿态。
                    action_net[:] = self.cfg.env.default_dof_pos[:] * (1.0 // self.cfg.env.action_scale)
                    action_real[:] = self.cfg.env.default_dof_pos[:]
                    kp[:] = self.cfg.robot_config.kps_stand[:]
                    kd[:] = self.cfg.robot_config.kds_stand[:]

                elif key_comm.stepTest:
                    # 当状态是“挂起动腿模式”时：使用动作发生器，生成腿部动作
                    action_net[:] = 0.
                    action_real[:] = 0.
                    kp[:] = self.cfg.robot_config.kps_stand[:]
                    kd[:] = self.cfg.robot_config.kds_stand[:]

                elif key_comm.stepNet:
                    # 当状态是“神经网络模式”时：使用神经网络输出动作。
                    action_net = policy(torch.tensor(obs_net))[0].detach().numpy()
                    kp[:] = self.cfg.robot_config.kps[:]
                    kd[:] = self.cfg.robot_config.kds[:]

                    # 关键一步:将神经网络生成的值*action_scale +默认关节位置 !!!!!!
                    action_real = action_net * self.cfg.env.action_scale + self.cfg.env.default_dof_pos
                    print(action_real)
                else:
                    print('退出')

                # 5.1 插值平滑输出
                if key_comm.timestep < count_max_merge:
                    action_real[:] = (pos_real_0[:] / count_max_merge * (count_max_merge - key_comm.timestep - 1)
                                      + action_real[:] / count_max_merge * (key_comm.timestep + 1))

                # 5.2 将神经网络输出的踝部关节角度,转换成实际电机指令
                action_real[4], action_real[5], action_real[10], action_real[11] = (
                    convert_ankle_net_to_real(action_real[4], action_real[5], action_real[10], action_real[11]))

                action_real = np.clip(action_real,
                                      self.cfg.env.joint_limit_min,
                                      self.cfg.env.joint_limit_max)

                # 5.3 将计算出来的真实电机值, 通过LCM发送给机器人
                if key_comm.stepCalibrate:
                    self.publish_action(action_real, kp, kd)
                elif key_comm.stepTest:
                    pass
                elif key_comm.stepNet:
                    self.publish_action(action_real, kp, kd)
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
        num_obs_real = 36  # 12+12+12
        num_obs_net = 45  # 3+3+3+12+12+12
        action_scale = 0.05
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
        args.load_model = '/home/qin/Desktop/legged_gym_2/logs/zq12/exported/policies/policy_yq2.pt'
    policy = torch.jit.load(args.load_model)
    deploy = Deploy(DeployCfg(), args.load_model)
    deploy.run_robot(policy)
