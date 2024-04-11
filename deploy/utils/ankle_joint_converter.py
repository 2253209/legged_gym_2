import numpy as np

# 逆运动学
def inverse_kinematics(l_bar, l_rod, l_spacing, q_roll, q_pitch):
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

    # 点位图对应的点坐标
    r_A1_0 = np.array([0, l_spacing1 / 2, l_rod1])
    r_B1_0 = np.array([-l_bar, l_spacing1 / 2, l_rod1])
    r_C1_0 = np.array([-l_bar, l_spacing1 / 2, 0])

    r_A2_0 = np.array([0, -l_spacing2 / 2, l_rod2])
    r_B2_0 = np.array([-l_bar, -l_spacing2 / 2, l_rod2])
    r_C2_0 = np.array([-l_bar, -l_spacing2 / 2, 0])

    r_A_0 = [r_A1_0, r_A2_0]
    r_B_0 = [r_B1_0, r_B2_0]
    r_C_0 = [r_C1_0, r_C2_0]

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
        r_C_i = x_rot @ r_C_0[i]

        a = r_C_i - r_A_i
        a = a[0]
        b = r_A_i - r_C_i
        b = b[2]
        c = (l_rod[i] ** 2 - l_bar ** 2 - np.linalg.norm(r_C_i - r_A_i) ** 2) / (2 * l_bar)

        theta_i = np.arcsin(
            (b * c + np.sqrt(b ** 2 * c ** 2 - (a ** 2 + b ** 2) * (c ** 2 - a ** 2))) / (
                    a ** 2 + b ** 2))  # 电机指令

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

    J = np.linalg.inv(J_theta) @ J_x

    G = np.array([[0, 0, 0, np.cos(q_pitch), 0, -np.sin(q_pitch)],
                  [0, 0, 0, 0, 1, 0]]).T
    J_c = J @ G

    return J_c

# 正运动学
def forward_kinematics(theta_ref,leg='right'or'left'):
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
        l_rod1 = 216  # B1C1
        l_rod2 = 120  # B2C2
        l_spacing1 = 47.9
        l_spacing2 = 38.6
    else:
        l_rod1 = 120  # B1C1
        l_rod2 = 216  # B2C2
        l_spacing1 = 38.6
        l_spacing2 = 47.9

    l_rod = [l_rod1, l_rod2]
    l_spacing = [l_spacing1, l_spacing2]
    angle2rad = 1 / 180 * np.pi
    f_error = np.array([10, 10])
    x_c_k = np.array([0, 0])
    while np.linalg.norm(f_error) > 1e-6:
        _, _, r_C, r_bar, r_rod, THETA_k = inverse_kinematics(l_bar, l_rod, l_spacing, x_c_k[0], x_c_k[1])
        J_c_k = jacobian(r_C, r_bar, r_rod, x_c_k[1], s_11, s_21)
        f_ik = np.block([THETA_k[0], THETA_k[1]])
        x_c_k = x_c_k - np.linalg.pinv(J_c_k) @ (f_ik - theta_ref)
        f_error = f_ik - theta_ref
    return x_c_k/ angle2rad

# 解耦
def decouple(roll,pitch,leg='right'or'left'):
    my_l_bar = 45  # A1B1
    if leg == 'right':
        my_l_rod1 = 216  # B1C1
        my_l_rod2 = 120  # B2C2
        my_l_spacing1 = 47.9
        my_l_spacing2 = 38.6
    else:
        my_l_rod1 = 120  # B1C1
        my_l_rod2 = 216  # B2C2
        my_l_spacing1 = 38.6
        my_l_spacing2 = 47.9

    my_l_rod = [my_l_rod1, my_l_rod2]
    my_l_spacing = [my_l_spacing1, my_l_spacing2]

    angle2rad = 1 / 180 * np.pi
    my_s_11 = np.array([0, 1, 0])  # A1点单位方向向量
    my_s_21 = np.array([0, 1, 0])  # A2点单位方向向量
    my_r_A, my_r_B, my_r_C, my_r_bar, my_r_rod, my_THETA = inverse_kinematics(my_l_bar, my_l_rod,
                                                                                  my_l_spacing, roll, pitch)
    Jac = jacobian(my_r_C, my_r_bar, my_r_rod, pitch, my_s_11, my_s_21)
    
    return my_THETA, Jac

if __name__ == '__main__':
    angle2rad = 1 / 180 * np.pi
    red2angle = 180 / np.pi

    # my_theta_ref = np.array([-46.38490723, -53.91584432]) * angle2rad  # 参考电机角度指令
    my_theta_ref = np.array([6, 6]) * angle2rad
    print('my_theta_ref:', my_theta_ref)
    my_joint_angles = forward_kinematics(my_theta_ref,leg='left')

    print('my_joint_angles:', my_joint_angles)

    my_compute_angles,_ = decouple(my_joint_angles[0]*angle2rad,my_joint_angles[1]*angle2rad,leg='left')

    print('my_compute_angles:', np.array(my_compute_angles) * red2angle)

    my_joint_right, _ = decouple(0.0,0.3,"right")
    my_joint_left, _ = decouple(0.0,0.3,"left")

    print(my_joint_right, my_joint_left)
