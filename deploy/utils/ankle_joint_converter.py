
import numpy as np


# 逆运动学
def inverse_kinematics(l_bar, l_rod, l_spacing, q_roll, q_pitch, leg='right' or 'left'):
    """
    逆运动学
    :param l_bar: 凸轮长
    :param l_rod: 连杆长
    :param l_spacing: 两连杆距离
    :param q_roll: 脚踝横滚角（绕x）
    :param q_pitch: 脚踝俯仰角（绕y）
    """
    l_rod1 = l_rod[0]
    l_rod2 = l_rod[1]

    l_spacing1 = l_spacing[0]
    l_spacing2 = l_spacing[1]

    short_link_angle_0 = 14.2 * np.pi / 180
    long_link_angle_0 = 15.5 * np.pi / 180

    # 点位图对应的点坐标
    # r_A1_0 = np.array([0, l_spacing1, l_rod1])
    # r_B1_0 = np.array([-l_bar, l_spacing1, l_rod1])
    # r_C1_0 = np.array([-l_bar, l_spacing1, 0])

    # r_A2_0 = np.array([0, -l_spacing2, l_rod2])
    # r_B2_0 = np.array([-l_bar, -l_spacing2, l_rod2])
    # r_C2_0 = np.array([-l_bar, -l_spacing2, 0])

    if leg == 'right':
        r_A1_0 = np.array([-20.7, 44.5, 215.01])
        r_B1_0 = np.array([-20.7 - l_bar * np.cos(long_link_angle_0), 44.5, 215.01 - l_bar * np.sin(long_link_angle_0)])
        r_C1_0 = np.array([-43.4, 44.5, -12])

        r_A2_0 = np.array([-6.3, -40, 119.84])
        r_B2_0 = np.array([-6.3 - l_bar * np.cos(short_link_angle_0), -40, 119.84 - l_bar * np.sin(short_link_angle_0)])
        r_C2_0 = np.array([-43.4, -40, -12])
    else:
        r_A1_0 = np.array([-6.3, 40, 119.84])
        r_B1_0 = np.array([-6.3 - l_bar * np.cos(short_link_angle_0), 40, 119.84 - l_bar * np.sin(short_link_angle_0)])
        r_C1_0 = np.array([-43.4, 40, -12])

        r_A2_0 = np.array([-20.7, -44.5, 215.01])
        r_B2_0 = np.array([-20.7 - l_bar * np.cos(long_link_angle_0), -44.5, 215.01 - l_bar * np.sin(long_link_angle_0)])
        r_C2_0 = np.array([-43.4, -44.5, -12])

    r_A_0 = [r_A1_0, r_A2_0]
    r_B_0 = [r_B1_0, r_B2_0]
    r_C_0 = [r_C1_0, r_C2_0]

    # print("r_A_0: ",r_A_0)
    # print("r_B_0: ",r_B_0)
    # print("r_C_0: ",r_C_0)

    # 横滚俯仰角对应的旋转矩阵
    R_y = np.array([[np.cos(q_pitch), 0, np.sin(q_pitch)],
                    [0, 1, 0],
                    [-np.sin(q_pitch), 0, np.cos(q_pitch)]])

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(q_roll), -np.sin(q_roll)],
                    [0, np.sin(q_roll), np.cos(q_roll)]])

    x_rot = R_y @ R_x

    THETA = []
    r_A = []
    r_B = []
    r_C = []
    r_bar = []
    r_rod = []
    for i in range(2):
        r_A_i = r_A_0[i]
        r_AB_0 = r_B_0[i] - r_A_0[i]
        # print("i: ",i," | r_AB_0: ",r_AB_0)

        r_C_i = x_rot @ r_C_0[i]
        r_CA = r_A_i - r_C_i

        M = l_rod[i] ** 2 - np.linalg.norm(r_CA) ** 2 - l_bar ** 2
        N = r_CA[0] * r_AB_0[0] + r_CA[2] * r_AB_0[2]
        K = r_CA[0] * r_AB_0[2] - r_CA[2] * r_AB_0[0]

        a = 4 * K ** 2 + 4 * N ** 2
        b = -4 * M * K
        c = M ** 2 - 4 * N ** 2

        if b ** 2 - 4 * a * c < 0.0:
            print('input roll or pitch of ankle beyond the range!!')
            print('leg, roll, pitch: ', leg, q_roll, q_pitch)
        theta_i = np.arcsin((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))  # 电机指令

        R_y_theta = np.array([[np.cos(theta_i), 0, np.sin(theta_i)],
                              [0, 1, 0],
                              [-np.sin(theta_i), 0, np.cos(theta_i)]])
        r_B_i = r_A_i + R_y_theta @ (r_B_0[i] - r_A_0[i])

        r_bar_i = r_B_i - r_A_i
        r_rod_i = r_C_i - r_B_i

        r_A.append(r_A_i)
        r_B.append(r_B_i)
        r_C.append(r_C_i)
        r_bar.append(r_bar_i)
        r_rod.append(r_rod_i)
        THETA.append(theta_i)

    return r_A, r_B, r_C, r_bar, r_rod, THETA


# 计算Jacobian
def jacobian(r_C, r_bar, r_rod, q_pitch, s_11, s_21):
    """
    计算雅可比矩阵
    :param r_C: C点向量
    :param r_bar: 凸轮向量
    :param r_rod: 连杆向量
    :param q_pitch: 脚踝俯仰角（绕y）
    :param s_11: A1点单位方向向量
    :param s_21: A2点单位方向向量
    """
    # J_x * d_x = J_theta * d_theta

    J_x = np.block([[r_rod[0].T, np.cross(r_C[0], r_rod[0]).T],
                    [r_rod[1].T, np.cross(r_C[1], r_rod[1]).T]])

    J_theta = np.block([[s_11 @ np.cross(r_bar[0], r_rod[0]), 0],
                        [0, s_21 @ np.cross(r_bar[1], r_rod[1])]])

    J = np.linalg.pinv(J_theta) @ J_x

    G = np.array([[0, 0, 0, np.cos(q_pitch), 0, -np.sin(q_pitch)],
                  [0, 0, 0, 0, 1, 0]]).T
    J_c = J @ G

    return J_c


# 正运动学
def forward_kinematics(theta_ref, leg='right' or 'left'):
    """
    正向运动学
    :param theta_ref: 参考电机角度指令
    :param l_bar: 凸轮长
    :param l_rod: 连杆长
    :param l_spacing: 两连杆距离
    :param s_11: A1点单位方向向量
    :param s_21: A2点单位方向向量
    """
    s_11 = np.array([0, 1, 0])  # A1点单位方向向量
    s_21 = np.array([0, 1, 0])  # A2点单位方向向量
    l_bar = 45  # A1B1
    if leg == 'right':
        l_rod1 = 215.975  # B1C1
        l_rod2 = 121  # B2C2
        l_spacing1 = 44.5  # 47.9
        l_spacing2 = 40  # 38.6
    else:
        l_rod1 = 121  # B1C1
        l_rod2 = 215.975  # B2C2
        l_spacing1 = 40  # 38.6
        l_spacing2 = 44.5  # 47.9

    l_rod = [l_rod1, l_rod2]
    l_spacing = [l_spacing1, l_spacing2]
    # angle2rad = 1 / 180 * np.pi
    f_error = np.array([10, 10])
    x_c_k = np.array([0, 0])
    j_c_k = np.zeros((2, 2), dtype=np.float32)
    i = 0
    while np.linalg.norm(f_error) > 1e-6:
        _, _, r_C, r_bar, r_rod, THETA_k = inverse_kinematics(l_bar, l_rod, l_spacing, x_c_k[0], x_c_k[1], leg)
        J_c_k = jacobian(r_C, r_bar, r_rod, x_c_k[1], s_11, s_21)
        f_ik = np.block([THETA_k[0], THETA_k[1]])
        x_c_k_pre = x_c_k
        j_c_k = np.linalg.pinv(J_c_k)
        x_c_k = x_c_k - j_c_k @ (f_ik - theta_ref)
        f_error = f_ik - theta_ref
        i += 1
        if i > 10:
            raise ValueError("！！！forward_kinematics Excessive loop iterations(10).！！！")
        # print("theta_ref: ",theta_ref,"x_c_k_pre: ", x_c_k_pre, "f_ik: ",f_ik,"new_x_c_k: ",x_c_k)
        # print("J_c_k: ")
        # print(J_c_k)
        # print("inv_J: ")
        # print(np.linalg.pinv(J_c_k))
    return x_c_k, J_c_k


# 解耦
def decouple(roll, pitch, leg='right' or 'left'):
    my_l_bar = 45  # A1B1
    if leg == 'right':
        my_l_rod1 = 215.975  # B1C1
        my_l_rod2 = 121  # B2C2
        my_l_spacing1 = 44.5  # 47.9
        my_l_spacing2 = 40  # 38.6
    else:
        my_l_rod1 = 121  # B1C1
        my_l_rod2 = 215.975  # B2C2
        my_l_spacing1 = 40  # 38.6
        my_l_spacing2 = 44.5  # 47.9

    my_l_rod = [my_l_rod1, my_l_rod2]
    my_l_spacing = [my_l_spacing1, my_l_spacing2]

    angle2rad = 1 / 180 * np.pi
    my_s_11 = np.array([0, 1, 0])  # A1点单位方向向量
    my_s_21 = np.array([0, 1, 0])  # A2点单位方向向量
    my_r_A, my_r_B, my_r_C, my_r_bar, my_r_rod, my_THETA = inverse_kinematics(my_l_bar, my_l_rod,
                                                                              my_l_spacing, roll, pitch, leg)
    Jac = jacobian(my_r_C, my_r_bar, my_r_rod, pitch, my_s_11, my_s_21)

    return my_THETA, Jac


def tau_mapping_universal2motor(motor_joint_pos, tau_universal_joint, leg='right' or 'left'):
    _, jac = forward_kinematics(motor_joint_pos, leg)  # jac: from universal_joint_vel to motor_joint_vel

    tau_motor_joint = np.linalg.pinv(np.transpose(jac)) @ tau_universal_joint

    return tau_motor_joint


def tau_mapping_motor2universal(motor_joint_pos, tau_motor_joint, leg='right' or 'left'):
    _, jac = forward_kinematics(motor_joint_pos, leg)  # jac: from universal_joint_vel to motor_joint_vel

    tau_universal_joint = np.transpose(jac) @ tau_motor_joint

    return tau_universal_joint

def convert_ankle_tau_2_motor(motor_joint_pos, tau_universal_joint):  # 4\5\10\11
    tau_motor_right_leg = tau_mapping_universal2motor(np.array([-motor_joint_pos[0], motor_joint_pos[1]]), np.array([tau_universal_joint[1], -tau_universal_joint[0]]), leg='right')
    tau_motor_left_leg = tau_mapping_universal2motor(np.array([-motor_joint_pos[3], motor_joint_pos[2]]), np.array([tau_universal_joint[3], -tau_universal_joint[2]]), leg='left')

    return np.array([-tau_motor_right_leg[0], tau_motor_right_leg[1],
            tau_motor_left_leg[1], -tau_motor_left_leg[0]])

def convert_motor_tau_2_ankle(motor_joint_pos, tau_motor_joint):  # 4\5\10\11
    tau_ankle_right_leg = tau_mapping_motor2universal(np.array([-motor_joint_pos[0], motor_joint_pos[1]]), np.array([-tau_motor_joint[0], tau_motor_joint[1]]), leg='right')
    tau_ankle_left_leg = tau_mapping_motor2universal(np.array([-motor_joint_pos[3], motor_joint_pos[2]]), np.array([-tau_motor_joint[3], tau_motor_joint[2]]), leg='left')

    return np.array([-tau_ankle_right_leg[1], tau_ankle_right_leg[0],
            -tau_ankle_left_leg[1], tau_ankle_left_leg[0]])

def convert_p_ori_2_joint(p1, p2, p3, p4):  # 4/5/10/11
    joint_p_right, jac_right = decouple(p2, p1, "right")
    joint_p_left, jac_left = decouple(p4, p3, "left")
    return (joint_p_right[0], -joint_p_right[1],
            -joint_p_left[1], joint_p_left[0])  # 4\5\10\11

def convert_p_joint_2_ori(p1, p2, p3, p4):  # 4\5\10\11
    ori_right, jac_right = forward_kinematics(np.array([p1, -p2]), leg='right')
    ori_left, jac_left = forward_kinematics(np.array([-p4, p3]), leg='left')

    return (ori_right[1], ori_right[0],
            -ori_left[1], -ori_left[0])

def convert_pv_ori_2_joint(p1, p2, p3, p4, v1, v2, v3, v4):  # 4/5/10/11
    joint_p_right, jac_right = decouple(p2, p1, "right")
    joint_p_left, jac_left = decouple(p4, p3, "left")
    joint_v_right = jac_right @ np.array([v2, v1])  # 2*2
    joint_v_left = jac_left @ np.array([v4, v3])
    return (joint_p_right[0], -joint_p_right[1], -joint_p_left[1], joint_p_left[0],
            joint_v_right[0], -joint_v_right[1], -joint_v_left[1], joint_v_left[0],)  # 4\5\10\11

def convert_pv_joint_2_ori(p1, p2, p3, p4, v1, v2, v3, v4):  # 4\5\10\11
    ori_right, jac_right = forward_kinematics(np.array([p1, -p2]), leg='right')
    ori_left, jac_left = forward_kinematics(np.array([-p4, p3]), leg='left')
    ori_v_right = np.linalg.pinv(jac_right) @ np.array([v1, -v2])  # 2*2
    ori_v_left = np.linalg.pinv(jac_left) @ np.array([-v4, v3])

    return (ori_right[1], ori_right[0], -ori_left[1], -ori_left[0],
            ori_v_right[1], ori_v_right[0], -ori_v_left[1], -ori_v_left[0])

if __name__ == '__main__':
    angle2rad = 1 / 180 * np.pi
    red2angle = 180 / np.pi
    #
    # # my_theta_ref = np.array([-46.38490723, -53.91584432]) * angle2rad  # 参考电机角度指令
    # my_theta_ref = np.array([6, 6]) * angle2rad
    # print('my_theta_ref:', my_theta_ref)
    # my_joint_angles = forward_kinematics(my_theta_ref,leg='left')
    #
    # print('my_joint_angles:', my_joint_angles)
    #
    # my_compute_angles,_ = decouple(my_joint_angles[0]*angle2rad,my_joint_angles[1]*angle2rad, leg='left')
    #
    # print('my_compute_angles:', np.array(my_compute_angles) * red2angle)
    #
    # my_joint_right, _ = decouple(0.0,0.3,"right")
    # my_joint_left, _ = decouple(0.0,0.3,"left")
    #
    # print(my_joint_right, my_joint_left)

    # ！！！！！------------------------------------------------
    # arr = np.array([
    #     [1.06, -0.95, -0.95, 1.06],
    #     [-0.68, 0.69, 0.69, -0.68],
    #     [0.6, 0.3, 0.4, 0.40],
    #     [-0.4, -0.4, -0.6, -0.3],
    #     [0.41338, -0.2383, -0.41338, 0.2383],
    # ])
    # for x in arr:
    #     o1, o2, o3, o4 = convert_p_joint_2_ori(x[0], x[1], x[2], x[3])  # 电机角度——脚板姿态 4\5\10\11
    #     j1, j2, j3, j4 = convert_p_ori_2_joint(o1, o2, o3, o4)          # 脚板姿态——电机角度 4\5\10\11
    #     print('原始电机角度: %.4f, %.4f, %.4f, %.4f' % (x[0], x[1], x[2], x[3]))
    #     print('电机转关节: %.4f, %.4f, %.4f, %.4f' % (o1, o2, o3, o4))
    #     print('关节转电机: %.4f, %.4f, %.4f, %.4f' % (j1, j2, j3, j4))
    #     print('----------------')

    # ！！！！！------------------------------------------------
    # arr = np.array([
    #     [0, 0, 0, 0, 20., -20., -20., 20.],
    #     [0, 0, 0, 0, 20., 20., -20., -20.],
    #     # [-0.68, 0.69, 0.69, -0.68],
    #     # [0.6, 0.3, 0.4, 0.40],
    #     # [-0.4, -0.4, -0.6, -0.3],
    #     # [0.41338, -0.2383, -0.41338, 0.2383],
    # ])
    # for x in arr:
    #     o1, o2, o3, o4, v1, v2, v3, v4 = (
    #         convert_pv_joint_2_ori(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], ))  # 电机角度——脚板姿态 4\5\10\11
    #     j1, j2, j3, j4 = convert_p_ori_2_joint(o1, o2, o3, o4)          # 脚板姿态——电机角度 4\5\10\11
    #     print('原始电机角度: %.4f, %.4f, %.4f, %.4f ' % (x[0], x[1], x[2], x[3]), end='')
    #     print('原始电机速度: %.4f, %.4f, %.4f, %.4f ' % (x[4], x[5], x[6], x[7]))
    #     print('转关节角度: %.4f, %.4f, %.4f, %.4f ' % (o1, o2, o3, o4), end='')
    #     print('转关节速度: %.4f, %.4f, %.4f, %.4f ' % (v1, v2, v3, v4))
    #     print('关节转电机: %.4f, %.4f, %.4f, %.4f' % (j1, j2, j3, j4))
    #     print('----------------')

    # ！！！！！------------------------------------------------

    motor_joint_pos_input = np.array([0.0, 0.0, 0.0, 0.0])
    tau_motor_input = np.array([-48, 48, 48, -48])

    # tau_ankle_output = convert_motor_tau_2_ankle(motor_joint_pos_input, tau_motor_input)

    tau_ankle_output = np.array([0.1,0.0,0.1,0.0])
    tau_motor_check = convert_ankle_tau_2_motor(motor_joint_pos_input, tau_ankle_output)

    print('motor_joint_pos_input', motor_joint_pos_input)
    print('tau_motor_input', tau_motor_input)
    print('----------')
    print('tau_ankle_output', tau_ankle_output)
    print('tau_motor_check', tau_motor_check)
    #
    arr = np.array([
        [0.0, 0.0, -0.3279, 0.2753, ],
        [0.0, 0.0, -0.3279, 0.2753, ],
        [0.0, 0.0, -0.3491, 0.2956, ],
        [0.0, 0.0, -0.3455, 0.2935, ],
        [0.0, 0.0, -0.3415, 0.2923, ],
        [0.0, 0.0, -0.3357, 0.2927, ],
        [0.0, 0.0, -0.3287, 0.2949, ],
        [0.0, 0.0, -0.3235, 0.2979, ],
        [0.0, 0.0, -0.3207, 0.3017, ],
        [0.0, 0.0, -0.3209, 0.3071, ],
        [0.0, 0.0, -0.3222, 0.3143, ],
        [0.0, 0.0, -0.3236, 0.3229, ],
        [0.0, 0.0, -0.3252, 0.3325, ],
        [0.0, 0.0, -0.3299, 0.3445, ],
        [0.0, 0.0, -0.3349, 0.3549, ],
        [0.0, 0.0, -0.3383, 0.3639, ],
        [0.0, 0.0, -0.3387, 0.3697, ],
        [0.0, 0.0, -0.3334, 0.3719, ],
        [0.0, 0.0, -0.3298, 0.3733, ],
        [0.0, 0.0, -0.3197, 0.3695, ],
        [0.0, 0.0, -0.3072, 0.361, ],
        [0.0, 0.0, -0.2933, 0.3502, ],
        [0.0, 0.0, -0.2721, 0.3327, ],
        [0.0, 0.0, -0.2488, 0.3085, ],
        [0.0, 0.0, -0.2327, 0.2825, ],
        [0.0, 0.0, -0.2177, 0.2614, ],
        [0.0, 0.0, -0.2151, 0.2539, ],
        [0.0, 0.0, -0.2143, 0.2534, ],
        [0.0, 0.0, -0.2033, 0.2423, ],
        [0.0, 0.0, -0.2041, 0.2245, ],
        [0.0, 0.0, -0.2043, 0.186, ],
        [0.0, 0.0, -0.2346, 0.1795, ],
        [0.0, 0.0, -0.2521, 0.1965, ],
        [0.0, 0.0, -0.2842, 0.251, ],
        [0.0, 0.0, -0.2565, 0.2416, ],
        [0.0, 0.0, -0.2281, 0.2193, ],
        [0.0, 0.0, -0.2185, 0.2137, ],
        [0.0, 0.0, -0.2531, 0.224, ],
        [0.0, 0.0, -0.3117, 0.2676, ],
        [0.0, 0.0, -0.3854, 0.3435, ],
        [0.0, 0.0, -0.4073, 0.3736, ],
        [0.0, 0.0, -0.4127, 0.3943, ],
        [0.0, 0.0, -0.4625, 0.4337, ],
        [0.0, 0.0, -0.5027, 0.4643, ],
        [0.0, 0.0, -0.5227, 0.477, ],
        [0.0, 0.0, -0.5591, 0.5157, ],
        [0.0, 0.0, -0.5859, 0.5479, ],
        [0.0, 0.0, -0.5812, 0.5603, ],
        [0.0, 0.0, -0.5987, 0.5852, ],
        [0.0, 0.0, -0.6163, 0.6076, ],
        [0.0, 0.0, -0.6438, 0.6281, ],
        [0.0, 0.0, -0.6438, 0.6367, ],

    ])
    for x in arr:
        # j1, j2, j3, j4 = convert_p_ori_2_joint(x[0], x[1], x[2], x[3])  # 脚板姿态——电机角度 4\5\10\11 p/r/p/r
        init1, init2, init3, init4 = convert_p_joint_2_ori(0.0, 0.0, -0.3279, 0.2753)
        o1, o2, o3, o4 = convert_p_joint_2_ori(x[0], x[1], x[2], x[3])  # 电机角度——脚板姿态 4\5\10\11 p/r/p/r

        # print('关节转电机: %.4f, %.4f, %.4f, %.4f' % (j1, j2, j3, j4))
        print('%.4f, %.4f, %.4f, %.4f' % (o1 - init1, o2 - init2, o3 - init3, o4 - init4))

        # o1, o2, o3, o4 = convert_p_joint_2_ori(x[0], x[1], x[2] - 0.31, x[3] + 0.03)  # 电机角度——脚板姿态 4\5\10\11 p/r/p/r
        # print('%.4f, %.4f, %.4f, %.4f' % (o1, o2, o3, o4))
        #
        # print('---')

