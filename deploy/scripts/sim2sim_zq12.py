
import math
import time

import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import Zq12Cfg
import torch

from deploy.utils.logger import SimpleLogger, get_title_81


class Sim2simCfg(Zq12Cfg):

    class cmd:
        vx = 0.0  # 0.5
        vy = 0.  # 0.
        dyaw = 0.0  # 0.05

    class sim(Zq12Cfg.sim):
        sim_duration = 60.0
        file = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/zq01/mjcf/zq_box_foot.xml'
        # file = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/zq01/urdf/zq_box_foot.urdf'


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


def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return q, dq, quat, v, omega, gvec


def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg: Sim2simCfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim.file)
    model.opt.timestep = cfg.sim.dt
    data = mujoco.MjData(model)
    action_startup = np.zeros(cfg.env.num_actions, dtype=np.float32)
    default_joint_pos = np.zeros(cfg.env.num_actions, dtype=np.float32)
    for index, value in enumerate(cfg.init_state.default_joint_angles.values()):
        action_startup[index] = value * (1 // cfg.control.action_scale)
        default_joint_pos[index] = value
    data.qpos[7:] = default_joint_pos[:]

    kps = np.zeros(cfg.env.num_actions, dtype=np.float32)
    kds = np.zeros(cfg.env.num_actions, dtype=np.float32)
    for i, v in enumerate(cfg.control.stiffness.values()):
        kps[i] = np.float32(v)
    for i, v in enumerate(cfg.control.damping.values()):
        kds[i] = np.float32(v)

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros(cfg.env.num_actions, dtype=np.float32)
    action = np.zeros(cfg.env.num_actions, dtype=np.float32)

    count_lowlevel = 0
    count_max_merge = 50
    sp_logger = SimpleLogger(f'{LEGGED_GYM_ROOT_DIR}/logs/sim_log', get_title_81())

    # for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
    action[:] = action_startup[:]

    try:
        obs = np.zeros([1, cfg.env.num_observations], dtype=np.float32)  # 45
        total_data = np.zeros((1, 81), dtype=np.float32)  # 45+36
        for i in range(10):
            policy(torch.tensor(obs))[0].detach().numpy()

        last_time = time.time()
        for _ in range(int(cfg.sim.sim_duration / cfg.sim.dt)):
            # Obtain an observation
            q, dq, quat, v, omega, gvec = get_obs(data)
            q = q[-cfg.env.num_actions:]
            dq = dq[-cfg.env.num_actions:]

            if count_lowlevel % cfg.control.decimation == 0:

                eu_ang = quaternion_to_euler_array(quat)

                obs[0, 0:3] = omega * cfg.normalization.obs_scales.ang_vel
                obs[0, 3:6] = eu_ang
                obs[0, 6] = cfg.cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 7] = cfg.cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 8] = cfg.cmd.dyaw * cfg.normalization.obs_scales.ang_vel

                obs[0, 9:21] = (q - default_joint_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = action

                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                # hist_obs.append(obs)
                # hist_obs.popleft()
                total_data[0, 0:45] = obs[0, :]
                total_data[0, 45:57] = q[:]
                total_data[0, 57:69] = dq[:]
                total_data[0, 69:81] = target_q[:]
                curr_time = time.time()
                sp_logger.save(total_data, count_lowlevel, curr_time - last_time)
                last_time = curr_time

                # policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                # for i in range(cfg.env.frame_stack):
                #     policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

                action[:] = policy(torch.tensor(obs))[0].detach().numpy()
                # action[:] = action_startup[:]

                if count_lowlevel < count_max_merge:
                    # print(f'{count_lowlevel} action[2]={action[2]}', end='')

                    action[:] = (action_startup[:] / count_max_merge * (count_max_merge - count_lowlevel)
                                 + action[:] / count_max_merge * count_lowlevel)
                    # print(f' merged[2]={action[2]}')
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

                target_q = action * cfg.control.action_scale

            target_dq = np.zeros(cfg.env.num_actions, dtype=np.double)
            # Generate PD control
            tau = pd_control(target_q, q, kps,
                             target_dq, dq, kds)  # Calc torques
            print('target_q=%.4f, q=%.4f, tau==%.4f,' % (target_q[0], q[0], tau[0]))
            # tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau
            # mujoco.mj_resetData(model, data)
            mujoco.mj_step(model, data)
            viewer.render()
            count_lowlevel += 1
    except KeyboardInterrupt:
        pass
    finally:
        sp_logger.close()
        viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    if not args.load_model:
        args.load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/zq12/exported/policies/policy_1.pt'
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
