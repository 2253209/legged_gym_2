
import torch
import numpy as np


class ActionGenerator:

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = 1
        self.num_dof = 12
        self.device = "cpu"
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.dt = 0.005
        self.target_joint_pos_scale = 0.2
        # default_dof_pos： 所有电机初始位置都是0
        # self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = np.array(self.cfg.env.default_dof_pos)
        self.dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        self.actions = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)


    def _get_phase(self):
        cycle_time = self.cfg.env.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        stance_mask[torch.abs(sin_pos) < 0.1] = 1  # 双脚同时接地的时刻
        # print('phase={:.3f} sin={:.3f} step= {:.1f} left= {:.1f}  right= {:.1f}'.format(phase[0], sin_pos[0], self.episode_length_buf[0], stance_mask[0, 0], stance_mask[0, 1]))
        return stance_mask

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.target_joint_pos_scale
        scale_2 = 2 * scale_1
        self.ref_dof_pos[0] = self.default_dof_pos[0]
        self.ref_dof_pos[1] = self.default_dof_pos[1]
        self.ref_dof_pos[6] = self.default_dof_pos[6]
        self.ref_dof_pos[7] = self.default_dof_pos[7]
        # left swing
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[2] = - sin_pos_l * scale_1 + self.default_dof_pos[2]
        self.ref_dof_pos[3] = sin_pos_l * scale_2 + self.default_dof_pos[3]
        self.ref_dof_pos[4] = -sin_pos_l * scale_1 + self.default_dof_pos[4]
        self.ref_dof_pos[5] = 0
        # right
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[8] = sin_pos_r * scale_1 + self.default_dof_pos[8]
        self.ref_dof_pos[9] = - sin_pos_r * scale_2 + self.default_dof_pos[9]
        self.ref_dof_pos[10] = sin_pos_r * scale_1 + self.default_dof_pos[10]
        self.ref_dof_pos[11] = 0

        # self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0.

    def calibrate(self, joint_pos):
        # 将所有电机缓缓重置到初始状态
        target = joint_pos - self.default_dof_pos
        # joint_pos[target > 0.03] -= 0.03
        # joint_pos[target < -0.03] += 0.03
        joint_pos[target > 0.01] -= 0.01
        joint_pos[target < -0.01] += 0.01
        return 4 * joint_pos

    def step(self):
        self.compute_ref_state()
        self.episode_length_buf += 1
        actions = 4 * self.ref_dof_pos
        # delay = torch.rand((self.num_envs, 1), device=self.device)
        # actions = (1 - delay) * actions + delay * self.actions
        return actions
