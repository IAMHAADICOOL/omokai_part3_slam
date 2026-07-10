import numpy as np
from math import cos, sin, atan2


def quaternion_from_euler(roll, pitch, yaw):
    # ZYX Tait-Bryan convention: q = q_z(yaw) * q_y(pitch) * q_x(roll)
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)

    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy + sr * sp * cy,
        cr * cp * cy - sr * sp * sy,
    ]


def euler_from_quaternion(x, y, z, w):
    # Inverse of ZYX Tait-Bryan; pitch clamped to [-1, 1] to guard against numerical drift past ±π/2
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def quaternion_ned_to_enu(q):
    # NED→ENU axis permutation: x_enu=y_ned, y_enu=x_ned, z_enu=-z_ned
    # Corresponds to a 180° rotation about the NED x-axis followed by a 90° rotation about z
    q_enu = np.array([q[1], q[0], -q[2], q[3]], dtype=float)
    norm = np.linalg.norm(q_enu)

    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    return q_enu / norm


def warp_angle(angle):
    # Maps any angle to the principal interval (-π, π] via modular arithmetic
    return (angle + np.pi) % (2.0 * np.pi) - np.pi