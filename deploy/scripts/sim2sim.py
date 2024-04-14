import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR

import torch


class cmd:
    vx = 0.4
    vy = 0.0
    dyaw = 0.0


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
    return np.array([roll_x, pitch_y, yaw_z])


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
    return (q, dq, quat, v, omega, gvec)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)

    # 'JOINT_Y1': -0.1,
    # 'JOINT_Y2': 0.0,
    # 'JOINT_Y3': 0.25,
    # 'JOINT_Y4': -0.53,
    # 'JOINT_Y5': 0.3,
    # 'JOINT_Y6': 0.1,
    #
    # 'JOINT_Z1': 0.1,
    # 'JOINT_Z2': 0.0,
    # 'JOINT_Z3': 0.25,
    # 'JOINT_Z4': -0.53,
    # 'JOINT_Z5': 0.3,
    # 'JOINT_Z6': -0.1,
    action_startup = np.array([-0.1, 0., 0.25, -0.53, 0.3, 0.1,
                               0.1, 0., 0.25, -0.53, 0.3, -0.1])
    data.qpos[7:] = action_startup[:]

    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)

    count_lowlevel = 0

    action_startup[:] = action_startup[:] * 10.
    action[:] = action_startup[:]
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        # 1000hz -> 100hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi

            obs[0, 0:3] = omega * cfg.normalization.obs_scales.ang_vel
            obs[0, 3:6] = eu_ang
            obs[0, 6] = cmd.vx * cfg.normalization.obs_scales.lin_vel
            obs[0, 7] = cmd.vy * cfg.normalization.obs_scales.lin_vel
            obs[0, 8] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
            obs[0, 9:21] = q * cfg.normalization.obs_scales.dof_pos
            obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 33:45] = action

            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            # hist_obs.append(obs)
            # hist_obs.popleft()
            #
            # policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            # for i in range(cfg.env.frame_stack):
            #     policy_input[0, i * cfg.env.num_single_obs: (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]
            action[:] = policy(torch.tensor(obs))[0].detach().numpy()
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

            if count_lowlevel < 100:
                # print(f'{count_lowlevel} action[2]={action[2]}', end='')

                action[:] = (action_startup[:] / 100 * (100 - count_lowlevel)
                             + action[:] / 100 * count_lowlevel)

            target_q = action * cfg.control.action_scale

        target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                         target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()


    class Sim2simCfg():
        class env:
            num_actions = 12
            num_single_obs = 45

        class control:
            action_scale = 0.1

        class normalization:
            class obs_scales:
                lin_vel = 2.
                ang_vel = 0.25
                dof_pos = 1.
                dof_vel = 0.05
                quat = 1.
                height_measurements = 5.0

            clip_observations = 18.
            clip_actions = 18.

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/XBot/mjcf/XBot-L-terrain.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/zq01/mjcf/zq_box_foot.xml'
            sim_duration = 60.0
            dt = 0.005
            decimation = 2

        class robot_config:
            kps = np.array([200, 200, 200, 200, 100, 100, 200, 200, 200, 200, 100, 100], dtype=np.double)
            kds = np.array([10, 10, 10, 10, 2, 2, 10, 10, 10, 10, 2, 2], dtype=np.double)
            tau_limit = 200. * np.ones(12, dtype=np.double)

    if not args.load_model:
        args.load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/zq12/exported/policies/policy_1.pt'
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())