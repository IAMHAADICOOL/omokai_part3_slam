import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from irobot_create_msgs.msg import WheelVels

import gtsam
import numpy as np
from math import sin, cos

from .utils import (
    quaternion_from_euler,
    euler_from_quaternion,
    quaternion_ned_to_enu,
    warp_angle,
)
from .line_extractor import RansacLineExtractor
from .data_association import DataAssociation


def normalize_polar_line(rho, alpha):
    # Enforce the canonical form ρ ≥ 0 so each physical line has a unique (ρ, α) representation.
    # Negating ρ and shifting α by π maps (-ρ, α) to the equivalent positive-ρ form.
    rho = float(rho)
    alpha = float(alpha)

    if rho < 0.0:
        rho = -rho
        alpha = warp_angle(alpha + np.pi)
    else:
        alpha = warp_angle(alpha)

    return float(rho), float(alpha)


def rho_key(index):
    return gtsam.symbol("R", index)


def alpha_key(index):
    return gtsam.symbol("A", index)


def make_polar_line_factor(
    pose_key,
    rho_landmark_key,
    alpha_landmark_key,
    measurement,
    noise_model,
):
    # GTSAM CustomFactor encoding the polar line measurement residual:
    #   e = [ρ_pred − ρ_obs,  wrap(α_pred − α_obs)]
    # where the predicted observation from robot pose Xk = (x, y, θ) and world landmark (ρ_w, α_w) is:
    #   ρ_pred = ρ_w − x cos α_w − y sin α_w
    #   α_pred = wrap(α_w − θ)
    rho_obs = float(measurement[0])
    alpha_obs = float(measurement[1])

    def error_func(this, values, jacobians):
        pose = values.atPose2(this.keys()[0])
        rho_w = float(values.atDouble(this.keys()[1]))
        alpha_w = float(values.atDouble(this.keys()[2]))

        x_b = float(pose.x())
        y_b = float(pose.y())
        theta_b = float(pose.theta())

        rho_pred = rho_w - x_b * cos(alpha_w) - y_b * sin(alpha_w)
        alpha_pred = alpha_w - theta_b

        flipped = False

        # Re-normalise to canonical ρ ≥ 0; Jacobian signs are negated accordingly
        if rho_pred < 0.0:
            rho_pred = -rho_pred
            alpha_pred = warp_angle(alpha_pred + np.pi)
            flipped = True
        else:
            alpha_pred = warp_angle(alpha_pred)

        residual = np.array(
            [
                rho_pred - rho_obs,
                warp_angle(alpha_pred - alpha_obs),
            ],
            dtype=float,
        )

        if jacobians is not None:
            alpha_body = warp_angle(alpha_w - theta_b)

            # H_X = ∂e/∂X_k: rows are [∂e_ρ/∂x, ∂e_ρ/∂y, ∂e_ρ/∂θ; ∂e_α/∂x, ∂e_α/∂y, ∂e_α/∂θ]
            H_pose = np.array(
                [
                    [-cos(alpha_body), -sin(alpha_body), 0.0],
                    [0.0, 0.0, -1.0],
                ],
                dtype=float,
                order="F",
            )

            # H_ρ = ∂e/∂ρ_w = [1; 0]  (only the ρ residual depends on ρ_w)
            H_rho = np.array(
                [
                    [1.0],
                    [0.0],
                ],
                dtype=float,
                order="F",
            )

            # H_α = ∂e/∂α_w = [x sin α_w − y cos α_w; 1]
            H_alpha = np.array(
                [
                    [x_b * sin(alpha_w) - y_b * cos(alpha_w)],
                    [1.0],
                ],
                dtype=float,
                order="F",
            )

            if flipped:
                H_pose[0, :] *= -1.0
                H_rho[0, :] *= -1.0
                H_alpha[0, :] *= -1.0

            jacobians[0] = np.asfortranarray(H_pose)
            jacobians[1] = np.asfortranarray(H_rho)
            jacobians[2] = np.asfortranarray(H_alpha)

        return residual

    return gtsam.CustomFactor(
        noise_model,
        [pose_key, rho_landmark_key, alpha_landmark_key],
        error_func,
    )


def _hsv_to_rgb(h, s, v):
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


class GraphSlam(Node):
    def __init__(self):
        super().__init__("graph_slam_localization")

        self.declare_parameter("odom_frame", "world_enu")
        self.declare_parameter("base_frame", "turtlebot/base_footprint")

        self.declare_parameter("scan_topic", "/turtlebot/scan")
        self.declare_parameter("joint_states_topic", "/turtlebot/joint_states")
        self.declare_parameter("imu_topic", "/turtlebot/sensors/imu_data")
        self.declare_parameter("ground_truth_topic", "/turtlebot/odom_ground_truth")
        self.declare_parameter("odom_topic", "/odom_graphslam")
        self.declare_parameter(
            "ground_truth_enu_topic",
            "/turtlebot/odom_ground_truth_enu",
        )
        self.declare_parameter("landmark_topic", "/slam_landmarks")

        self.declare_parameter("publish_tf", False)
        self.declare_parameter("graphslam_tf_child_frame", "graphslam/base_footprint")

        self.declare_parameter(
            "wheel_left_joint_name",
            "turtlebot/wheel_left_joint",
        )
        self.declare_parameter(
            "wheel_right_joint_name",
            "turtlebot/wheel_right_joint",
        )

        self.declare_parameter("wheel_radius", 0.035)
        self.declare_parameter("wheel_base_distance", 0.230)
        self.declare_parameter("wheel_vel_noise_left", 0.01)
        self.declare_parameter("wheel_vel_noise_right", 0.01)

        self.declare_parameter("min_points_per_line", 10)
        self.declare_parameter("max_line_distance", 0.05)
        self.declare_parameter("max_gap_distance", 0.25)
        self.declare_parameter("max_iterations", 120)
        self.declare_parameter("min_line_length", 0.30)
        self.declare_parameter("max_range", 8.0)
        self.declare_parameter("min_inlier_ratio", 0.03)
        self.declare_parameter("duplicate_rho_threshold", 0.15)
        self.declare_parameter("duplicate_alpha_threshold", 0.15)
        self.declare_parameter("segment_merge_gap_distance", 0.35)

        self.declare_parameter("confidence_level", 0.99)
        self.declare_parameter("confirmed_min_observations", 2)
        self.declare_parameter("new_landmark_min_observations", 2)
        self.declare_parameter("allow_landmark_reuse", True)

        self.declare_parameter("measurement_sigma_rho_floor", 0.04)
        self.declare_parameter("measurement_sigma_alpha_floor", 0.06)

        self.declare_parameter("max_landmark_segment_length", 2.50)

        self.declare_parameter("visualization_segment_merge_gap", 0.25)

        self.declare_parameter("key_frame_update", 20)
        self.declare_parameter("num_min_key", 10)
        self.declare_parameter("map_publish_period", 0.50)
        self.declare_parameter("real_robot", False)

        self.world_frame = self.get_parameter("odom_frame").value
        self.base_footprint_frame = self.get_parameter("base_frame").value

        self.scan_topic = self.get_parameter("scan_topic").value
        self.joint_states_topic = self.get_parameter("joint_states_topic").value
        self.imu_topic = self.get_parameter("imu_topic").value
        self.ground_truth_topic = self.get_parameter("ground_truth_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.ground_truth_enu_topic = self.get_parameter("ground_truth_enu_topic").value
        self.landmark_topic = self.get_parameter("landmark_topic").value

        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.graphslam_tf_child_frame = self.get_parameter(
            "graphslam_tf_child_frame"
        ).value

        self.base_length = float(self.get_parameter("wheel_base_distance").value)
        self.radius_wheel = float(self.get_parameter("wheel_radius").value)

        self.left_wheel_joint_name = self.get_parameter(
            "wheel_left_joint_name"
        ).value
        self.right_wheel_joint_name = self.get_parameter(
            "wheel_right_joint_name"
        ).value

        wheel_vel_noise_left = float(
            self.get_parameter("wheel_vel_noise_left").value
        )
        wheel_vel_noise_right = float(
            self.get_parameter("wheel_vel_noise_right").value
        )

        # Q = diag(σ_L², σ_R²) — process noise covariance for the unicycle kinematic model
        self.covariance_wheel_encoder = np.diag(
            np.array(
                [
                    wheel_vel_noise_left**2,
                    wheel_vel_noise_right**2,
                ],
                dtype=float,
            )
        )

        self.confidence_level = float(self.get_parameter("confidence_level").value)

        self.confirmed_min_observations = int(
            self.get_parameter("confirmed_min_observations").value
        )

        self.new_landmark_min_observations = int(
            self.get_parameter("new_landmark_min_observations").value
        )

        self.allow_landmark_reuse = bool(
            self.get_parameter("allow_landmark_reuse").value
        )

        self.measurement_sigma_rho_floor = float(
            self.get_parameter("measurement_sigma_rho_floor").value
        )

        self.measurement_sigma_alpha_floor = float(
            self.get_parameter("measurement_sigma_alpha_floor").value
        )

        self.duplicate_rho_threshold = float(
            self.get_parameter("duplicate_rho_threshold").value
        )

        self.duplicate_alpha_threshold = float(
            self.get_parameter("duplicate_alpha_threshold").value
        )

        self.max_landmark_segment_length = float(
            self.get_parameter("max_landmark_segment_length").value
        )

        self.visualization_segment_merge_gap = float(
            self.get_parameter("visualization_segment_merge_gap").value
        )

        self.key_frame_update = int(self.get_parameter("key_frame_update").value)
        self.num_min_key = int(self.get_parameter("num_min_key").value)

        self.map_publish_period = float(
            self.get_parameter("map_publish_period").value
        )
        self.real_robot = bool(self.get_parameter("real_robot").value)
        self.get_logger().info(f"[GraphSLAM] real_robot={self.real_robot}")

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.Pk = np.diag([0.1, 0.1, 0.1])
        self.intialize_theta = False

        self._use_wheel_vels = False

        self.wheel_vels_sub = self.create_subscription(
            WheelVels,
            "/wheel_vels",
            self.wheel_vels_callback,
            20,
        )

        self.joint_states_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self.joint_state_callback,
            20,
        )

        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.recieve_imu,
            50,
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.receive_lidar,
            20,
        )

        self.odom_ground_truth = self.create_subscription(
            Odometry,
            self.ground_truth_topic,
            self.recieve_odom_ground_truth,
            20,
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            self.odom_topic,
            20,
        )

        self.odom_ground_truth_enu_pub = self.create_publisher(
            Odometry,
            self.ground_truth_enu_topic,
            20,
        )

        self.landmark_pub = self.create_publisher(
            MarkerArray,
            self.landmark_topic,
            20,
        )

        self.tf_br = None
        if self.publish_tf:
            self.tf_br = TransformBroadcaster(self)

        self.first_time = True
        self.last_time = self.get_clock().now()

        self._gt_origin_x = None
        self._gt_origin_y = None
        self.imu_update_flag = False
        self.imu_orientation = 0.0
        self._last_imu_time = 0.0
        self.imu_covariance = np.array([[0.01**2]], dtype=float)
        self.imu_buffer = []
        self.imu_time_buffer = []

        self.lidar_update_flag = False
        self.line_buffer = []
        self.line_time_buffer = []

        self.polar_coordinates = []
        self.polar_covariances = []
        self.line_segments_local = []

        self.rel_disp = np.zeros((3, 1), dtype=float)
        self.rel_cov = np.zeros((3, 3), dtype=float)

        params = gtsam.ISAM2Params()
        self.isam2 = gtsam.ISAM2(params)
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()

        self.i = 0
        self.k = 0
        self.initialize = True

        self.line_extractor = RansacLineExtractor(
            min_points_per_line=self.get_parameter("min_points_per_line").value,
            max_line_distance=self.get_parameter("max_line_distance").value,
            max_gap_distance=self.get_parameter("max_gap_distance").value,
            max_iterations=self.get_parameter("max_iterations").value,
            min_line_length=self.get_parameter("min_line_length").value,
            max_range=self.get_parameter("max_range").value,
            min_inlier_ratio=self.get_parameter("min_inlier_ratio").value,
            duplicate_rho_threshold=self.get_parameter(
                "duplicate_rho_threshold"
            ).value,
            duplicate_alpha_threshold=self.get_parameter(
                "duplicate_alpha_threshold"
            ).value,
            segment_merge_gap_distance=self.get_parameter(
                "segment_merge_gap_distance"
            ).value,
            measurement_sigma_rho_floor=self.measurement_sigma_rho_floor,
            measurement_sigma_alpha_floor=self.measurement_sigma_alpha_floor,
            lidar_angle_offset=0.0 if self.real_robot else np.pi,
        )

        self.line_feature_map = []
        self.line_feature_cov_map = []

        self.landmark_segments_world = []

        self.landmark_observation_count = []
        self.landmark_first_seen_pose = {}
        self.landmark_last_seen_pose = {}

        self.pending_line_features = []
        self.pending_line_covariances = []

        self.pending_line_segments_world = []

        self.pending_line_observation_count = []
        self.pending_line_first_seen_pose = []
        self.pending_line_last_seen_pose = []

        self.H = []
        self._map_dirty = True
        self.data_association = None

        self.last_landmark_marker_count = 0
        self.last_uncertainty_marker_count = 0
        self.last_label_marker_count = 0
        self.last_map_log_time = 0.0

        self.create_timer(self.map_publish_period, self.publish_landmarks)

        self.get_logger().info("[GraphSLAM] Started successfully.")
        self.get_logger().info(f"[GraphSLAM] scan_topic={self.scan_topic}")
        self.get_logger().info(
            f"[GraphSLAM] joint_states_topic={self.joint_states_topic}"
        )
        self.get_logger().info(f"[GraphSLAM] imu_topic={self.imu_topic}")
        self.get_logger().info(f"[GraphSLAM] landmark_topic={self.landmark_topic}")
        self.get_logger().info(f"[GraphSLAM] world_frame={self.world_frame}")
        self.get_logger().info(f"[GraphSLAM] base_frame={self.base_footprint_frame}")
        self.get_logger().info(f"[GraphSLAM] publish_tf={self.publish_tf}")
        self.get_logger().info(
            f"[GraphSLAM] graphslam_tf_child_frame={self.graphslam_tf_child_frame}"
        )
        self.get_logger().info(
            f"[GraphSLAM] visualization_segment_merge_gap="
            f"{self.visualization_segment_merge_gap:.3f}"
        )

    def _safe_covariance(self, cov, dim):
        cov = np.asarray(cov, dtype=float)

        if cov.shape != (dim, dim):
            cov = np.eye(dim) * 1e-3

        cov = 0.5 * (cov + cov.T)
        cov += np.eye(dim) * 1e-9

        return cov

    def _floor_polar_covariance(self, cov):
        cov = self._safe_covariance(cov, 2)

        cov[0, 0] = max(cov[0, 0], self.measurement_sigma_rho_floor**2)
        cov[1, 1] = max(cov[1, 1], self.measurement_sigma_alpha_floor**2)

        return cov

    def _safe_noise_model(self, cov, dim):
        cov = self._safe_covariance(cov, dim)
        return gtsam.noiseModel.Gaussian.Covariance(cov)

    def recieve_imu(self, msg):
        cov_orientation = msg.orientation_covariance

        if self.real_robot:
            # Create3 orientation quaternion is frozen; integrate gyroscope rate directly
            omega_z = msg.angular_velocity.z
            # Deadband below sensor noise floor (~0.01 rad/s)
            if abs(omega_z) < 0.01:
                omega_z = 0.0
            now = self.get_clock().now().nanoseconds * 1e-9
            if self._last_imu_time > 0:
                dt = now - self._last_imu_time
                if 0.0 < dt < 0.5:
                    self.imu_orientation = warp_angle(
                        self.imu_orientation + omega_z * dt
                    )
            self._last_imu_time = now
        else:
            # Stonefish publishes IMU orientation in NED; convert to ENU with +π/2 heading correction
            orientation_ned = np.array(
                [msg.orientation.x, msg.orientation.y,
                 msg.orientation.z, msg.orientation.w],
                dtype=float,
            )
            orientation_enu = quaternion_ned_to_enu(orientation_ned)
            self.imu_orientation = euler_from_quaternion(
                orientation_enu[0], orientation_enu[1],
                orientation_enu[2], orientation_enu[3],
            )[2]
            self.imu_orientation = warp_angle(self.imu_orientation + np.pi / 2.0)

        self.imu_covariance = np.array(
            [[max(float(cov_orientation[8]), 1e-6)]],
            dtype=float,
        )

        self.imu_update_flag = True

        self.imu_buffer.append([self.imu_orientation, self.imu_covariance])
        self.imu_time_buffer.append(self.get_clock().now().nanoseconds / 1e9)

        if len(self.imu_buffer) > 20:
            self.imu_buffer.pop(0)
            self.imu_time_buffer.pop(0)

        if not self.intialize_theta:
            self.theta = self.imu_orientation
            self.intialize_theta = True

    def receive_lidar(self, msg):
        lidar_points = self.line_extractor.transform_lidar_to_cartesian(msg)
        line_segments = self.line_extractor.extract_line_segments(lidar_points)

        polar_coordinates = self.line_extractor.calculate_polar_coordinates(
            line_segments
        )

        polar_covariances = []

        for line in line_segments:
            cov_line = self.line_extractor.calculate_covariance_matrix(line)
            cov_line = self._floor_polar_covariance(cov_line)
            polar_covariances.append(cov_line)

        self.lidar_update_flag = True

        self.line_buffer.append(
            [
                polar_coordinates,
                polar_covariances,
                line_segments,
            ]
        )

        self.line_time_buffer.append(self.get_clock().now().nanoseconds / 1e9)

        if len(self.line_buffer) > 20:
            self.line_buffer.pop(0)
            self.line_time_buffer.pop(0)

    def f(self, wheel_velocity, dt):
        # Unicycle kinematic prediction: x̄_k = f(x_{k-1}, u_k, dt)
        # v = r/2 (ω_L + ω_R),  ω = r/L (ω_R − ω_L)
        linear_velocity = (
            0.5
            * self.radius_wheel
            * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        )

        angular_velocity = (
            self.radius_wheel
            / self.base_length
            * (-wheel_velocity[0, 0] + wheel_velocity[1, 0])
        )

        x_k = self.x + cos(self.theta) * linear_velocity * dt
        y_k = self.y + sin(self.theta) * linear_velocity * dt
        theta_k = warp_angle(self.theta + angular_velocity * dt)

        return x_k, y_k, theta_k

    def Jfx(self, wheel_velocity, dt):
        # J_fx = ∂f/∂x — state transition Jacobian for covariance propagation P̄_k = J_fx P_{k-1} J_fx^T + J_fw Q J_fw^T
        linear_velocity = (
            0.5
            * self.radius_wheel
            * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        )

        return np.array(
            [
                [1.0, 0.0, -sin(self.theta) * linear_velocity * dt],
                [0.0, 1.0, cos(self.theta) * linear_velocity * dt],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def Jfw(self, wheel_velocity, dt):
        # J_fw = ∂f/∂w — noise input Jacobian mapping wheel velocity noise w~Q into state space
        return np.array(
            [
                [
                    0.5 * self.radius_wheel * cos(self.theta) * dt,
                    0.5 * self.radius_wheel * cos(self.theta) * dt,
                ],
                [
                    0.5 * self.radius_wheel * sin(self.theta) * dt,
                    0.5 * self.radius_wheel * sin(self.theta) * dt,
                ],
                [
                    -self.radius_wheel / self.base_length * dt,
                    self.radius_wheel / self.base_length * dt,
                ],
            ],
            dtype=float,
        )

    def Jfx_rel(self, wheel_velocity, dt):
        # Analogous to Jfx but evaluated at the accumulated relative displacement θ_rel for BetweenFactor covariance
        linear_velocity = (
            0.5
            * self.radius_wheel
            * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        )

        theta_rel = self.rel_disp[2, 0]

        return np.array(
            [
                [1.0, 0.0, -sin(theta_rel) * linear_velocity * dt],
                [0.0, 1.0, cos(theta_rel) * linear_velocity * dt],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def Jfw_rel(self, dt):
        # Noise input Jacobian for the relative displacement accumulator (used to build the BetweenFactorPose2 noise model)
        theta_rel = self.rel_disp[2, 0]

        return np.array(
            [
                [
                    0.5 * self.radius_wheel * cos(theta_rel) * dt,
                    0.5 * self.radius_wheel * cos(theta_rel) * dt,
                ],
                [
                    0.5 * self.radius_wheel * sin(theta_rel) * dt,
                    0.5 * self.radius_wheel * sin(theta_rel) * dt,
                ],
                [
                    -self.radius_wheel / self.base_length * dt,
                    self.radius_wheel / self.base_length * dt,
                ],
            ],
            dtype=float,
        )

    def _insert_landmark_initial(self, landmark_id, polar_feature):
        rho_w, alpha_w = normalize_polar_line(
            polar_feature[0],
            polar_feature[1],
        )

        if not self.initial.exists(rho_key(landmark_id)):
            self.initial.insert(rho_key(landmark_id), rho_w)

        if not self.initial.exists(alpha_key(landmark_id)):
            self.initial.insert(alpha_key(landmark_id), alpha_w)

    def _robot_to_world_point(self, point_robot, pose_vec):
        x = float(pose_vec[0])
        y = float(pose_vec[1])
        theta = float(pose_vec[2])

        px = float(point_robot[0])
        py = float(point_robot[1])

        point_world = np.array(
            [
                x + cos(theta) * px - sin(theta) * py,
                y + sin(theta) * px + cos(theta) * py,
            ],
            dtype=float,
        )

        return point_world

    def _transform_segment_to_world(self, local_segment, pose_vec):
        start_world = self._robot_to_world_point(local_segment[0], pose_vec)
        end_world = self._robot_to_world_point(local_segment[1], pose_vec)

        return [start_world, end_world]

    def _segment_length(self, segment):
        if segment is None:
            return 0.0

        return float(np.linalg.norm(segment[1] - segment[0]))

    def _line_geometry(self, rho_w, alpha_w):
        rho_w, alpha_w = normalize_polar_line(rho_w, alpha_w)

        normal = np.array(
            [cos(alpha_w), sin(alpha_w)],
            dtype=float,
        )

        direction = np.array(
            [-sin(alpha_w), cos(alpha_w)],
            dtype=float,
        )

        center_on_line = rho_w * normal

        return rho_w, alpha_w, normal, direction, center_on_line

    def _segment_interval_on_polar_line(self, segment_world, rho_w, alpha_w):
        rho_w, alpha_w, _, direction, center_on_line = self._line_geometry(
            rho_w,
            alpha_w,
        )

        p1 = np.asarray(segment_world[0], dtype=float)
        p2 = np.asarray(segment_world[1], dtype=float)

        t1 = float((p1 - center_on_line) @ direction)
        t2 = float((p2 - center_on_line) @ direction)

        return min(t1, t2), max(t1, t2)

    def _interval_to_segment(self, t_min, t_max, rho_w, alpha_w):
        rho_w, alpha_w, _, direction, center_on_line = self._line_geometry(
            rho_w,
            alpha_w,
        )

        if t_max < t_min:
            t_min, t_max = t_max, t_min

        length = t_max - t_min

        if length > self.max_landmark_segment_length:
            t_mid = 0.5 * (t_min + t_max)
            half = 0.5 * self.max_landmark_segment_length
            t_min = t_mid - half
            t_max = t_mid + half

        start = center_on_line + t_min * direction
        end = center_on_line + t_max * direction

        return [start, end]

    def _align_segment_to_polar_line(self, segment_world, rho_w, alpha_w):
        if segment_world is None:
            return None

        t_min, t_max = self._segment_interval_on_polar_line(
            segment_world,
            rho_w,
            alpha_w,
        )

        return self._interval_to_segment(t_min, t_max, rho_w, alpha_w)

    def _merge_segment_list_on_polar_line(
        self,
        old_segment_list,
        new_segment,
        rho_w,
        alpha_w,
    ):
        if old_segment_list is None:
            old_segment_list = []

        old_segment_list = [
            segment for segment in old_segment_list if segment is not None
        ]

        if new_segment is None:
            return old_segment_list

        aligned_new_segment = self._align_segment_to_polar_line(
            new_segment,
            rho_w,
            alpha_w,
        )

        if aligned_new_segment is None:
            return old_segment_list

        new_min, new_max = self._segment_interval_on_polar_line(
            aligned_new_segment,
            rho_w,
            alpha_w,
        )

        intervals = []

        for old_segment in old_segment_list:
            old_min, old_max = self._segment_interval_on_polar_line(
                old_segment,
                rho_w,
                alpha_w,
            )

            intervals.append([old_min, old_max])

        intervals.append([new_min, new_max])
        intervals.sort(key=lambda item: item[0])

        merged_intervals = []

        for interval in intervals:
            if len(merged_intervals) == 0:
                merged_intervals.append(interval)
                continue

            prev = merged_intervals[-1]

            gap = interval[0] - prev[1]

            if gap <= self.visualization_segment_merge_gap:
                prev[1] = max(prev[1], interval[1])
            else:
                merged_intervals.append(interval)

        merged_segments = []

        for t_min, t_max in merged_intervals:
            segment = self._interval_to_segment(t_min, t_max, rho_w, alpha_w)

            if self._segment_length(segment) >= 0.05:
                merged_segments.append(segment)

        return merged_segments

    def _ensure_landmark_storage(self, landmark_id):
        while len(self.landmark_segments_world) <= landmark_id:
            self.landmark_segments_world.append([])

        while len(self.landmark_observation_count) <= landmark_id:
            self.landmark_observation_count.append(0)

    def _initialize_landmark_segment(self, landmark_id, obs_idx, pose_vec):
        self._ensure_landmark_storage(landmark_id)

        if obs_idx >= len(self.line_segments_local):
            self.landmark_segments_world[landmark_id] = []
            return

        rho_w, alpha_w = self.line_feature_map[landmark_id]

        segment_world = self._transform_segment_to_world(
            self.line_segments_local[obs_idx],
            pose_vec,
        )

        aligned_segment = self._align_segment_to_polar_line(
            segment_world,
            rho_w,
            alpha_w,
        )

        if aligned_segment is None:
            self.landmark_segments_world[landmark_id] = []
        else:
            self.landmark_segments_world[landmark_id] = [aligned_segment]

    def _update_landmark_segment(self, landmark_id, obs_idx, pose_vec):
        self._ensure_landmark_storage(landmark_id)

        if obs_idx >= len(self.line_segments_local):
            return

        rho_w, alpha_w = self.line_feature_map[landmark_id]

        new_segment_world = self._transform_segment_to_world(
            self.line_segments_local[obs_idx],
            pose_vec,
        )

        self.landmark_segments_world[landmark_id] = (
            self._merge_segment_list_on_polar_line(
                self.landmark_segments_world[landmark_id],
                new_segment_world,
                rho_w,
                alpha_w,
            )
        )

    def _realign_landmark_segment_after_optimization(
        self,
        landmark_id,
        rho_w,
        alpha_w,
    ):
        if landmark_id >= len(self.landmark_segments_world):
            return

        segment_list = self.landmark_segments_world[landmark_id]

        if segment_list is None:
            self.landmark_segments_world[landmark_id] = []
            return

        realigned_segments = []

        for segment in segment_list:
            aligned_segment = self._align_segment_to_polar_line(
                segment,
                rho_w,
                alpha_w,
            )

            if aligned_segment is not None and self._segment_length(aligned_segment) >= 0.05:
                realigned_segments.append(aligned_segment)

        self.landmark_segments_world[landmark_id] = realigned_segments

    def _feature_geometry_score(self, candidate, feature):
        rho_c = float(candidate[0])
        alpha_c = float(candidate[1])

        rho_f = float(feature[0])
        alpha_f = float(feature[1])

        rho_error = abs(rho_c - rho_f)
        alpha_error = abs(warp_angle(alpha_c - alpha_f))

        if rho_error > self.duplicate_rho_threshold:
            return None

        if alpha_error > self.duplicate_alpha_threshold:
            return None

        score = (
            rho_error / max(self.duplicate_rho_threshold, 1e-9)
            + alpha_error / max(self.duplicate_alpha_threshold, 1e-9)
        )

        return float(score)

    def _find_duplicate_feature_in_list(self, candidate, feature_list):
        best_idx = None
        best_score = np.inf

        for idx, feature in enumerate(feature_list):
            score = self._feature_geometry_score(candidate, feature)

            if score is None:
                continue

            if score < best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _blend_polar_features(self, old_feature, new_feature, old_count):
        old_rho, old_alpha = normalize_polar_line(
            old_feature[0],
            old_feature[1],
        )

        new_rho, new_alpha = normalize_polar_line(
            new_feature[0],
            new_feature[1],
        )

        count = max(int(old_count), 1)

        rho = (count * old_rho + new_rho) / float(count + 1)

        x = count * cos(old_alpha) + cos(new_alpha)
        y = count * sin(old_alpha) + sin(new_alpha)

        alpha = warp_angle(np.arctan2(y, x))

        return [float(rho), float(alpha)]

    def _candidate_segment_world(self, obs_idx, pose_vec, feature):
        if obs_idx >= len(self.line_segments_local):
            return None

        segment_world = self._transform_segment_to_world(
            self.line_segments_local[obs_idx],
            pose_vec,
        )

        rho_w, alpha_w = feature

        return self._align_segment_to_polar_line(
            segment_world,
            rho_w,
            alpha_w,
        )

    def _remove_pending_landmark(self, pending_idx):
        self.pending_line_features.pop(pending_idx)
        self.pending_line_covariances.pop(pending_idx)
        self.pending_line_segments_world.pop(pending_idx)
        self.pending_line_observation_count.pop(pending_idx)
        self.pending_line_first_seen_pose.pop(pending_idx)
        self.pending_line_last_seen_pose.pop(pending_idx)

    def _promote_pending_landmark(self, pending_idx):
        feature = self.pending_line_features[pending_idx]
        cov = self.pending_line_covariances[pending_idx]
        segment_list = self.pending_line_segments_world[pending_idx]
        obs_count = self.pending_line_observation_count[pending_idx]
        first_seen = self.pending_line_first_seen_pose[pending_idx]
        last_seen = self.pending_line_last_seen_pose[pending_idx]

        rho_w, alpha_w = normalize_polar_line(feature[0], feature[1])
        feature = [rho_w, alpha_w]

        landmark_id = len(self.line_feature_map)

        self.line_feature_map.append(feature)
        self.line_feature_cov_map.append(self._floor_polar_covariance(cov))
        self._map_dirty = True

        self._ensure_landmark_storage(landmark_id)

        self.landmark_observation_count[landmark_id] = int(obs_count)
        self.landmark_first_seen_pose[landmark_id] = int(first_seen)
        self.landmark_last_seen_pose[landmark_id] = int(last_seen)

        final_segments = []

        if segment_list is not None:
            for segment in segment_list:
                aligned_segment = self._align_segment_to_polar_line(
                    segment,
                    rho_w,
                    alpha_w,
                )

                final_segments = self._merge_segment_list_on_polar_line(
                    final_segments,
                    aligned_segment,
                    rho_w,
                    alpha_w,
                )

        self.landmark_segments_world[landmark_id] = final_segments

        self._insert_landmark_initial(landmark_id, feature)

        self._remove_pending_landmark(pending_idx)

        self.get_logger().info(
            f"[DA] promoted pending line -> L({landmark_id}), "
            f"rho={rho_w:.3f}, alpha={alpha_w:.3f}, obs={obs_count}, "
            f"segments={len(final_segments)}"
        )

        return landmark_id

    def _handle_unmatched_observation(self, xk_bar, Pk_bar, obs_idx, pose_vec):
        if self.data_association is None:
            return None

        candidate, candidate_cov = self.data_association.observation_to_world_feature(
            xk_bar,
            Pk_bar,
            self.polar_coordinates[obs_idx],
            self.polar_covariances[obs_idx],
        )

        candidate_cov = self._floor_polar_covariance(candidate_cov)

        existing_id = self._find_duplicate_feature_in_list(
            candidate,
            self.line_feature_map,
        )

        if existing_id is not None:
            self._ensure_landmark_storage(existing_id)

            self.landmark_observation_count[existing_id] += 1
            self.landmark_last_seen_pose[existing_id] = self.i + 1
            self._map_dirty = True  # invalidate DA cache; map feature count changed

            new_segment = self._candidate_segment_world(
                obs_idx,
                pose_vec,
                self.line_feature_map[existing_id],
            )

            self.landmark_segments_world[existing_id] = (
                self._merge_segment_list_on_polar_line(
                    self.landmark_segments_world[existing_id],
                    new_segment,
                    self.line_feature_map[existing_id][0],
                    self.line_feature_map[existing_id][1],
                )
            )

            self.get_logger().info(
                f"[DA] obs#{obs_idx} rescued by geometry -> MATCH L({existing_id})"
            )

            return existing_id

        pending_idx = self._find_duplicate_feature_in_list(
            candidate,
            self.pending_line_features,
        )

        new_segment = self._candidate_segment_world(
            obs_idx,
            pose_vec,
            candidate,
        )

        if pending_idx is None:
            pending_segment_list = []

            if new_segment is not None:
                pending_segment_list = [new_segment]

            self.pending_line_features.append(candidate)
            self.pending_line_covariances.append(candidate_cov)
            self.pending_line_segments_world.append(pending_segment_list)
            self.pending_line_observation_count.append(1)
            self.pending_line_first_seen_pose.append(self.i + 1)
            self.pending_line_last_seen_pose.append(self.i + 1)

            pending_idx = len(self.pending_line_features) - 1

            self.get_logger().info(
                f"[DA] obs#{obs_idx} -> PENDING P({pending_idx}), "
                f"rho={candidate[0]:.3f}, alpha={candidate[1]:.3f}, "
                f"obs=1/{self.new_landmark_min_observations}"
            )

        else:
            old_count = self.pending_line_observation_count[pending_idx]

            blended_feature = self._blend_polar_features(
                self.pending_line_features[pending_idx],
                candidate,
                old_count,
            )

            old_cov = self.pending_line_covariances[pending_idx]
            self.pending_line_covariances[pending_idx] = 0.5 * old_cov + 0.5 * candidate_cov
            self.pending_line_features[pending_idx] = blended_feature
            self.pending_line_observation_count[pending_idx] += 1
            self.pending_line_last_seen_pose[pending_idx] = self.i + 1

            self.pending_line_segments_world[pending_idx] = (
                self._merge_segment_list_on_polar_line(
                    self.pending_line_segments_world[pending_idx],
                    new_segment,
                    blended_feature[0],
                    blended_feature[1],
                )
            )

            self.get_logger().info(
                f"[DA] obs#{obs_idx} -> UPDATE PENDING P({pending_idx}), "
                f"obs={self.pending_line_observation_count[pending_idx]}/"
                f"{self.new_landmark_min_observations}, "
                f"segments={len(self.pending_line_segments_world[pending_idx])}"
            )

        if self.pending_line_observation_count[pending_idx] >= self.new_landmark_min_observations:
            return self._promote_pending_landmark(pending_idx)

        return None

    def _log_data_association(self, xk_bar, Pk_bar, H_raw, map_features_before):
        observations = len(self.polar_coordinates)
        matches = sum(h is not None for h in H_raw)
        new_or_unmatched = sum(h is None for h in H_raw)

        recovered_duplicates = 0

        if self.data_association is not None:
            recovered_duplicates = int(
                getattr(self.data_association, "recovered_duplicates", 0)
            )

        self.get_logger().info(
            f"[DA] observations={observations}, "
            f"map_features={map_features_before}, "
            f"matches={matches}, "
            f"new_or_unmatched={new_or_unmatched}, "
            f"recovered_duplicates={recovered_duplicates}, "
            f"pending={len(self.pending_line_features)}, "
            f"H={H_raw}"
        )

        if self.data_association is None:
            return

        for obs_idx, association in enumerate(H_raw):
            rho_obs = float(self.polar_coordinates[obs_idx][0])
            alpha_obs = float(self.polar_coordinates[obs_idx][1])

            if association is None:
                self.get_logger().info(
                    f"[DA] obs#{obs_idx}=({rho_obs:.3f},{alpha_obs:.3f}) "
                    f"-> UNMATCHED/PENDING"
                )
                continue

            hfj = self.data_association.hfj(xk_bar, association)
            phfj = self.data_association.expected_observation_covariance(
                xk_bar,
                Pk_bar,
                association,
            )

            chi2_value = self.data_association.SquaredMahalanobisDistance(
                hfj,
                phfj,
                self.polar_coordinates[obs_idx],
                self.polar_covariances[obs_idx],
            )

            seen_count = 0

            if association < len(self.landmark_observation_count):
                seen_count = self.landmark_observation_count[association]

            segment_count = 0

            if association < len(self.landmark_segments_world):
                segment_count = len(self.landmark_segments_world[association])

            self.get_logger().info(
                f"[DA] obs#{obs_idx}=({rho_obs:.3f},{alpha_obs:.3f}) "
                f"-> MATCH L({association}) chi2={chi2_value:.2f}, "
                f"seen_count={seen_count}, segments={segment_count}"
            )

            if association in self.landmark_first_seen_pose:
                first_seen = self.landmark_first_seen_pose[association]

                if self.i - first_seen > self.key_frame_update:
                    self.get_logger().info(
                        f"[DA] LOOP_CLOSURE_ACTIVE: "
                        f"X({self.i}) re-observing L({association}) "
                        f"first seen at X({first_seen}), "
                        f"gap={self.i - first_seen} poses — "
                        f"ISAM2 will correct drift on next optimization"
                    )

    def _apply_data_association_results(self, xk_bar, Pk_bar, H_raw):
        pose_vec = np.array(
            [
                float(xk_bar[0, 0]),
                float(xk_bar[1, 0]),
                float(xk_bar[2, 0]),
            ],
            dtype=float,
        )

        self.H = list(H_raw)

        if self.data_association is None:
            return

        for obs_idx, association in enumerate(H_raw):
            if association is None:
                landmark_id = self._handle_unmatched_observation(
                    xk_bar,
                    Pk_bar,
                    obs_idx,
                    pose_vec,
                )

                if landmark_id is not None:
                    self.H[obs_idx] = landmark_id

                continue

            landmark_id = int(association)

            self._ensure_landmark_storage(landmark_id)

            self.landmark_observation_count[landmark_id] += 1
            self.landmark_last_seen_pose[landmark_id] = self.i + 1

            self._update_landmark_segment(landmark_id, obs_idx, pose_vec)

    def Prediction(self, wheel_velocity, dt):
        x_bar, y_bar, theta_bar = self.f(wheel_velocity, dt)

        linear_velocity = (
            0.5
            * self.radius_wheel
            * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        )

        angular_velocity = (
            self.radius_wheel
            / self.base_length
            * (-wheel_velocity[0, 0] + wheel_velocity[1, 0])
        )

        dx_rel = cos(self.rel_disp[2, 0]) * linear_velocity * dt
        dy_rel = sin(self.rel_disp[2, 0]) * linear_velocity * dt
        dtheta_rel = angular_velocity * dt

        self.rel_disp += np.array(
            [
                [dx_rel],
                [dy_rel],
                [dtheta_rel],
            ],
            dtype=float,
        )

        self.rel_disp[2, 0] = warp_angle(self.rel_disp[2, 0])

        Jfx = self.Jfx(wheel_velocity, dt)
        Jfw = self.Jfw(wheel_velocity, dt)

        Pk_bar = (
            Jfx @ self.Pk @ Jfx.T
            + Jfw @ self.covariance_wheel_encoder @ Jfw.T
        )

        Pk_bar = self._safe_covariance(Pk_bar, 3)

        xk_bar = np.array(
            [
                [x_bar],
                [y_bar],
                [theta_bar],
            ],
            dtype=float,
        )

        Jfx_rel = self.Jfx_rel(wheel_velocity, dt)
        Jfw_rel = self.Jfw_rel(dt)

        self.rel_cov = (
            Jfx_rel @ self.rel_cov @ Jfx_rel.T
            + Jfw_rel @ self.covariance_wheel_encoder @ Jfw_rel.T
        )

        self.rel_cov = self._safe_covariance(self.rel_cov, 3)

        return xk_bar, Pk_bar

    def Update(self, xk_bar, Pk_bar):
        zk = np.array([[self.imu_orientation]], dtype=float)
        Rk = self._safe_covariance(self.imu_covariance, 1)

        self.i += 1

        pose_key_prev = gtsam.symbol("X", self.i - 1)
        pose_key_curr = gtsam.symbol("X", self.i)

        # BetweenFactorPose2 encodes the relative displacement Δx_rel accumulated since the last keyframe
        rel_pos = gtsam.Pose2(
            float(self.rel_disp[0, 0]),
            float(self.rel_disp[1, 0]),
            float(self.rel_disp[2, 0]),
        )

        odometry_noise = self._safe_noise_model(self.rel_cov, 3)

        self.graph.add(
            gtsam.BetweenFactorPose2(
                pose_key_prev,
                pose_key_curr,
                rel_pos,
                odometry_noise,
            )
        )

        if self.imu_update_flag:
            # PoseRotationPrior2D constrains the heading component of Xk to the IMU-derived yaw estimate
            sigma_heading = float(np.sqrt(Rk[0, 0] + 1e-7))

            heading_prior_pose = gtsam.Pose2(
                0.0,
                0.0,
                float(zk[0, 0]),
            )

            heading_noise = gtsam.noiseModel.Isotropic.Sigma(
                1,
                sigma_heading,
            )

            self.graph.add(
                gtsam.PoseRotationPrior2D(
                    pose_key_curr,
                    heading_prior_pose,
                    heading_noise,
                )
            )

            self.k += 1

        if not self.initial.exists(pose_key_curr):
            self.initial.insert(
                pose_key_curr,
                gtsam.Pose2(
                    float(xk_bar[0, 0]),
                    float(xk_bar[1, 0]),
                    float(xk_bar[2, 0]),
                ),
            )

        if self.lidar_update_flag:
            for j in range(len(self.H)):
                if self.H[j] is None:
                    continue

                if j >= len(self.polar_covariances):
                    continue

                landmark_id = int(self.H[j])

                noise_model = self._safe_noise_model(
                    self._floor_polar_covariance(self.polar_covariances[j]),
                    2,
                )

                self.graph.add(
                    make_polar_line_factor(
                        pose_key_curr,
                        rho_key(landmark_id),
                        alpha_key(landmark_id),
                        self.polar_coordinates[j],
                        noise_model,
                    )
                )

            # Avoid double-counting keyframe ticks when IMU and LiDAR both arrive in the same cycle
            if len(self.H) > 0 and not self.imu_update_flag:
                self.k += 1

        # Trigger full graph optimisation every key_frame_update sensor-fused nodes,
        # but only after num_min_key poses have accumulated to ensure a well-constrained graph
        should_optimize = (
            self.i > self.num_min_key
            and self.k >= self.key_frame_update
        )

        if should_optimize:
            self.k = 0

            try:
                if self.initialize:
                    # First optimisation uses Levenberg-Marquardt for global batch initialisation;
                    # subsequent updates use iSAM2 incremental re-linearisation
                    optimizer = gtsam.LevenbergMarquardtOptimizer(
                        self.graph,
                        self.initial,
                    )

                    self.initial = optimizer.optimize()
                    self.initialize = False

                self.isam2.update(self.graph, self.initial)

                self.graph = gtsam.NonlinearFactorGraph()
                self.initial = gtsam.Values()

                full_graph = self.isam2.getFactorsUnsafe()
                results = self.isam2.calculateEstimate()

                pose = results.atPose2(pose_key_curr)

                xk = np.array(
                    [
                        [pose.x()],
                        [pose.y()],
                        [pose.theta()],
                    ],
                    dtype=float,
                )

                marginals = gtsam.Marginals(full_graph, results)

                try:
                    Pk = marginals.marginalCovariance(pose_key_curr)
                    Pk = self._safe_covariance(Pk, 3)
                except Exception as exc:
                    self.get_logger().warning(
                        f"[GraphSLAM] Could not compute pose marginal: {exc}"
                    )
                    Pk = Pk_bar

                for j in range(len(self.line_feature_map)):
                    try:
                        rho_w, alpha_w = normalize_polar_line(
                            results.atDouble(rho_key(j)),
                            results.atDouble(alpha_key(j)),
                        )

                        self.line_feature_map[j] = [rho_w, alpha_w]

                        landmark_keys = gtsam.KeyVector()
                        landmark_keys.append(rho_key(j))
                        landmark_keys.append(alpha_key(j))

                        self.line_feature_cov_map[j] = self._floor_polar_covariance(
                            marginals.jointMarginalCovariance(
                                landmark_keys
                            ).fullMatrix()
                        )

                        self._realign_landmark_segment_after_optimization(
                            j,
                            rho_w,
                            alpha_w,
                        )

                        if j < len(self.landmark_observation_count):
                            obs_count = self.landmark_observation_count[j]
                        else:
                            obs_count = 0

                        segment_count = 0

                        if j < len(self.landmark_segments_world):
                            segment_count = len(self.landmark_segments_world[j])

                        self.get_logger().info(
                            f"[OPT] L({j}) rho={rho_w:.3f}, "
                            f"alpha={alpha_w:.3f}, obs={obs_count}, "
                            f"segments={segment_count}"
                        )

                    except Exception as exc:
                        self.get_logger().warning(
                            f"[GraphSLAM] Could not update landmark {j}: {exc}"
                        )

                self.publish_landmarks()

            except Exception as exc:
                self.get_logger().error(
                    f"[GraphSLAM] Optimization failed: {exc}"
                )

                xk = xk_bar
                Pk = Pk_bar

        else:
            xk = xk_bar
            Pk = Pk_bar

        self.rel_disp = np.zeros((3, 1), dtype=float)
        self.rel_cov = np.zeros((3, 3), dtype=float)

        return xk, Pk

    def _wheel_velocity_from_joint_state(self, msg):
        if (
            self.left_wheel_joint_name in msg.name
            and self.right_wheel_joint_name in msg.name
        ):
            left_idx = msg.name.index(self.left_wheel_joint_name)
            right_idx = msg.name.index(self.right_wheel_joint_name)

            return msg.velocity[left_idx], msg.velocity[right_idx]

        if len(msg.velocity) < 2:
            return None

        return msg.velocity[0], msg.velocity[1]

    def _curve_boundary_points(self, segment_world, landmark_id):
        if segment_world is None:
            return [], []

        if landmark_id >= len(self.line_feature_map):
            return [], []

        if landmark_id >= len(self.line_feature_cov_map):
            return [], []

        start_point = np.asarray(segment_world[0], dtype=float)
        end_point = np.asarray(segment_world[1], dtype=float)

        line_vec = end_point - start_point
        line_len = float(np.linalg.norm(line_vec))

        if line_len <= 1e-9:
            return [], []

        _, alpha_w = self.line_feature_map[landmark_id]
        alpha_w = warp_angle(alpha_w)

        tangent = line_vec / line_len

        normal = np.array(
            [cos(alpha_w), sin(alpha_w)],
            dtype=float,
        )

        cov_line = self._floor_polar_covariance(
            self.line_feature_cov_map[landmark_id]
        )

        samples_per_meter = 80.0
        min_samples = 40
        max_samples = 400

        num_samples = int(
            np.clip(
                np.ceil(samples_per_meter * line_len),
                min_samples,
                max_samples,
            )
        )

        chi2_95_2d = 5.991464547107979

        boundary_plus = []
        boundary_minus = []

        for s_value in np.linspace(0.0, line_len, num_samples):
            point_nominal = start_point + float(s_value) * tangent

            jacobian = np.array(
                [
                    1.0,
                    point_nominal[0] * sin(alpha_w)
                    - point_nominal[1] * cos(alpha_w),
                ],
                dtype=float,
            )

            variance_dist = float(jacobian @ cov_line @ jacobian.T)
            variance_dist = max(variance_dist, 0.0)

            confidence_dist = float(np.sqrt(chi2_95_2d * variance_dist))
            confidence_dist = min(confidence_dist, 0.20)

            boundary_plus.append(point_nominal + confidence_dist * normal)
            boundary_minus.append(point_nominal - confidence_dist * normal)

        return boundary_plus, boundary_minus

    def _append_uncertainty_curve_marker(
        self,
        marker_array,
        marker_id,
        boundary_points,
        now,
    ):
        marker = Marker()

        marker.header.frame_id = self.world_frame
        marker.header.stamp = now
        marker.ns = "stable_slam_landmark_uncertainty_world"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.025

        marker.color.a = 0.90
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        marker.pose.orientation.w = 1.0

        for point in boundary_points:
            marker.points.append(
                Point(
                    x=float(point[0]),
                    y=float(point[1]),
                    z=0.075,
                )
            )

        marker_array.markers.append(marker)

    def _landmark_color(self, landmark_id):
        """Golden-ratio hue cycle for perceptually distinct landmark colours."""
        golden_ratio = 0.618033988749895
        h = (landmark_id * golden_ratio) % 1.0
        r, g, b = _hsv_to_rgb(h, 0.85, 0.95)
        return float(r), float(g), float(b)

    def publish_landmarks(self):
        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        marker_id = 0
        uncertainty_marker_id = 0
        published_count = 0
        published_uncertainty_curves = 0

        for landmark_id, segment_list in enumerate(self.landmark_segments_world):
            if segment_list is None:
                continue

            if landmark_id >= len(self.landmark_observation_count):
                continue

            if (
                self.landmark_observation_count[landmark_id]
                < self.confirmed_min_observations
            ):
                continue

            for segment_idx, segment in enumerate(segment_list):
                if segment is None:
                    continue

                length = self._segment_length(segment)

                if length < 0.05:
                    continue

                marker = Marker()

                marker.header.frame_id = self.world_frame
                marker.header.stamp = now
                marker.ns = "stable_slam_landmarks_world"
                marker.id = marker_id
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD

                marker.points.append(
                    Point(
                        x=float(segment[0][0]),
                        y=float(segment[0][1]),
                        z=0.05,
                    )
                )

                marker.points.append(
                    Point(
                        x=float(segment[1][0]),
                        y=float(segment[1][1]),
                        z=0.05,
                    )
                )

                marker.scale.x = 0.07

                r, g, b = self._landmark_color(landmark_id)
                marker.color.a = 1.0
                marker.color.r = r
                marker.color.g = g
                marker.color.b = b

                marker.pose.orientation.w = 1.0

                marker_array.markers.append(marker)

                midpoint_x = float(segment[0][0] + segment[1][0]) / 2.0
                midpoint_y = float(segment[0][1] + segment[1][1]) / 2.0

                label_marker = Marker()
                label_marker.header.frame_id = self.world_frame
                label_marker.header.stamp = now
                label_marker.ns = "stable_slam_landmark_labels"
                label_marker.id = landmark_id
                label_marker.type = Marker.TEXT_VIEW_FACING
                label_marker.action = Marker.ADD
                label_marker.pose.position.x = midpoint_x
                label_marker.pose.position.y = midpoint_y
                label_marker.pose.position.z = 0.25
                label_marker.pose.orientation.w = 1.0
                label_marker.scale.z = 0.18
                label_marker.color.a = 1.0
                label_marker.color.r = 1.0
                label_marker.color.g = 1.0
                label_marker.color.b = 1.0
                label_marker.text = f"L({landmark_id})"
                marker_array.markers.append(label_marker)

                marker_id += 1
                published_count += 1

                boundary_plus, boundary_minus = self._curve_boundary_points(
                    segment,
                    landmark_id,
                )

                if len(boundary_plus) > 1 and len(boundary_minus) > 1:
                    self._append_uncertainty_curve_marker(
                        marker_array,
                        uncertainty_marker_id,
                        boundary_plus,
                        now,
                    )

                    uncertainty_marker_id += 1
                    published_uncertainty_curves += 1

                    self._append_uncertainty_curve_marker(
                        marker_array,
                        uncertainty_marker_id,
                        boundary_minus,
                        now,
                    )

                    uncertainty_marker_id += 1
                    published_uncertainty_curves += 1

        for old_marker_id in range(marker_id, self.last_landmark_marker_count):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = now
            marker.ns = "stable_slam_landmarks_world"
            marker.id = old_marker_id
            marker.action = Marker.DELETE
            marker.pose.orientation.w = 1.0

            marker_array.markers.append(marker)

        for old_marker_id in range(
            uncertainty_marker_id,
            self.last_uncertainty_marker_count,
        ):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = now
            marker.ns = "stable_slam_landmark_uncertainty_world"
            marker.id = old_marker_id
            marker.action = Marker.DELETE
            marker.pose.orientation.w = 1.0

            marker_array.markers.append(marker)

        self.last_landmark_marker_count = marker_id
        self.last_uncertainty_marker_count = uncertainty_marker_id

        current_label_count = len(self.line_feature_map)
        for old_id in range(current_label_count, self.last_label_marker_count):
            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = now
            marker.ns = "stable_slam_landmark_labels"
            marker.id = old_id
            marker.action = Marker.DELETE
            marker.pose.orientation.w = 1.0
            marker_array.markers.append(marker)
        self.last_label_marker_count = current_label_count

        self.landmark_pub.publish(marker_array)

        current_time = self.get_clock().now().nanoseconds / 1e9

        if current_time - self.last_map_log_time > 1.0:
            self.last_map_log_time = current_time

            total_visual_segments = 0

            for segment_list in self.landmark_segments_world:
                if segment_list is not None:
                    total_visual_segments += len(segment_list)

            self.get_logger().info(
                f"[MAP] topic={self.landmark_topic}, "
                f"published_segments={published_count}, "
                f"published_uncertainty_curves={published_uncertainty_curves}, "
                f"total_landmarks={len(self.line_feature_map)}, "
                f"total_visual_segments={total_visual_segments}, "
                f"pending={len(self.pending_line_features)}, "
                f"confirmed_min_obs={self.confirmed_min_observations}, "
                f"visual_merge_gap={self.visualization_segment_merge_gap:.3f}"
            )

    def joint_state_callback(self, msg):
        if self._use_wheel_vels:
            return  # wheel_vels topic takes precedence on real hardware

        if not self.intialize_theta:
            return

        wheel_velocities = self._wheel_velocity_from_joint_state(msg)

        if wheel_velocities is None:
            self.get_logger().warning(
                "JointState has no wheel velocities yet"
            )
            return

        left_wheel_velocity, right_wheel_velocity = wheel_velocities
        self._process_wheel_odometry(left_wheel_velocity, right_wheel_velocity)

    def wheel_vels_callback(self, msg):
        self._use_wheel_vels = True
        self._process_wheel_odometry(msg.velocity_left, msg.velocity_right)

    def _process_wheel_odometry(self, left_wheel_velocity, right_wheel_velocity):
        if not self.intialize_theta:
            return

        current_time = self.get_clock().now()

        if self.first_time:
            self.first_time = False
            self.last_time = current_time

            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([1e-7, 1e-7, 1e-7], dtype=float)
            )

            pose_key = gtsam.symbol("X", self.i)

            self.graph.add(
                gtsam.PriorFactorPose2(
                    pose_key,
                    gtsam.Pose2(0.0, 0.0, float(self.theta)),
                    prior_noise,
                )
            )

            self.initial = gtsam.Values()
            self.initial.insert(
                pose_key,
                gtsam.Pose2(0.0, 0.0, float(self.theta)),
            )

            self.get_logger().info(
                f"[GraphSLAM] Initial pose X(0) inserted with "
                f"theta={self.theta:.3f}"
            )

            return

        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if dt <= 0.0:
            return

        wheel_velocity = np.array(
            [
                [left_wheel_velocity],
                [right_wheel_velocity],
            ],
            dtype=float,
        )

        if self.imu_update_flag and len(self.imu_buffer) > 0:
            self.imu_orientation = self.imu_buffer[-1][0]
            self.imu_covariance = self.imu_buffer[-1][1]

            self.imu_buffer.clear()
            self.imu_time_buffer.clear()

        if self.lidar_update_flag and len(self.line_buffer) > 0:
            latest_line_data = self.line_buffer[-1]

            self.polar_coordinates = latest_line_data[0]
            self.polar_covariances = latest_line_data[1]
            self.line_segments_local = latest_line_data[2]

            self.line_buffer.clear()
            self.line_time_buffer.clear()

        xk_bar, Pk_bar = self.Prediction(wheel_velocity, dt)

        if self.lidar_update_flag:
            map_features_before = len(self.line_feature_map)

            if self.data_association is None or self._map_dirty:
                self.data_association = DataAssociation(
                    self.confidence_level,
                    self.line_feature_map,
                    self.line_feature_cov_map,
                    allow_landmark_reuse=self.allow_landmark_reuse,
                    duplicate_rho_threshold=self.duplicate_rho_threshold,
                    duplicate_alpha_threshold=self.duplicate_alpha_threshold,
                    measurement_sigma_rho_floor=self.measurement_sigma_rho_floor,
                    measurement_sigma_alpha_floor=self.measurement_sigma_alpha_floor,
                )
                self._map_dirty = False
            else:
                # Refresh map references without reconstructing the DA object
                self.data_association.map_feature = self.line_feature_map
                self.data_association.map_feature_cov = self.line_feature_cov_map
                self.data_association.nf = len(self.line_feature_map)

            H_raw = self.data_association.DataAssociation(
                xk_bar,
                Pk_bar,
                self.polar_coordinates,
                self.polar_covariances,
            )

            self._log_data_association(
                xk_bar,
                Pk_bar,
                H_raw,
                map_features_before,
            )

            self._apply_data_association_results(
                xk_bar,
                Pk_bar,
                H_raw,
            )

        if self.imu_update_flag or self.lidar_update_flag:
            xk, Pk = self.Update(xk_bar, Pk_bar)

            self.x = float(xk[0, 0])
            self.y = float(xk[1, 0])
            self.theta = warp_angle(float(xk[2, 0]))
            self.Pk = self._safe_covariance(Pk, 3)

            self.imu_update_flag = False

        else:
            self.x = float(xk_bar[0, 0])
            self.y = float(xk_bar[1, 0])
            self.theta = warp_angle(float(xk_bar[2, 0]))
            self.Pk = self._safe_covariance(Pk_bar, 3)

        if self.lidar_update_flag:
            self.publish_landmarks()
            self.lidar_update_flag = False

        linear_velocity = (
            0.5
            * self.radius_wheel
            * (wheel_velocity[0, 0] + wheel_velocity[1, 0])
        )

        angular_velocity = (
            self.radius_wheel
            / self.base_length
            * (-wheel_velocity[0, 0] + wheel_velocity[1, 0])
        )

        odom_msg = Odometry()

        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = self.world_frame
        odom_msg.child_frame_id = self.graphslam_tf_child_frame

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        q = quaternion_from_euler(0.0, 0.0, self.theta)

        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        odom_msg.twist.twist.linear.x = float(linear_velocity)
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0

        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = float(angular_velocity)

        cov = np.zeros((6, 6), dtype=float)

        cov[0:2, 0:2] = self.Pk[0:2, 0:2]
        cov[0:2, 5] = self.Pk[0:2, 2]
        cov[5, 0:2] = self.Pk[2, 0:2]
        cov[5, 5] = self.Pk[2, 2]

        odom_msg.pose.covariance = cov.flatten().tolist()

        self.odom_pub.publish(odom_msg)

        if self.publish_tf and self.tf_br is not None:
            t = TransformStamped()

            t.header.stamp = current_time.to_msg()
            t.header.frame_id = self.world_frame
            t.child_frame_id = self.graphslam_tf_child_frame

            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0

            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            self.tf_br.sendTransform(t)

    def recieve_odom_ground_truth(self, msg):
        odom_ground_truth = msg

        x = odom_ground_truth.pose.pose.position.x
        y = odom_ground_truth.pose.pose.position.y

        # Fix the ENU origin to the first received ground-truth pose
        if self._gt_origin_x is None:
            self._gt_origin_x = y
            self._gt_origin_y = x
            self.get_logger().info(
                f"[GT] Origin set: raw=({x:.3f},{y:.3f}) "
                f"stored offset=({self._gt_origin_x:.3f},{self._gt_origin_y:.3f})"
            )

        linear_velocity = odom_ground_truth.twist.twist.linear.x
        angular_velocity = odom_ground_truth.twist.twist.angular.z

        q_x = odom_ground_truth.pose.pose.orientation.x
        q_y = odom_ground_truth.pose.pose.orientation.y
        q_z = odom_ground_truth.pose.pose.orientation.z
        q_w = odom_ground_truth.pose.pose.orientation.w

        _, _, theta = euler_from_quaternion(q_x, q_y, q_z, q_w)

        theta = warp_angle(np.pi / 2.0 - theta)

        q = quaternion_from_euler(0.0, 0.0, theta)

        current_time = self.get_clock().now()

        odom_ground_truth_enu_msg = Odometry()

        odom_ground_truth_enu_msg.header.stamp = current_time.to_msg()
        odom_ground_truth_enu_msg.header.frame_id = self.world_frame
        odom_ground_truth_enu_msg.child_frame_id = self.base_footprint_frame

        odom_ground_truth_enu_msg.pose.pose.position.x = -(y - self._gt_origin_x)
        odom_ground_truth_enu_msg.pose.pose.position.y = -(x - self._gt_origin_y)
        odom_ground_truth_enu_msg.pose.pose.position.z = 0.0

        odom_ground_truth_enu_msg.pose.pose.orientation.x = q[0]
        odom_ground_truth_enu_msg.pose.pose.orientation.y = q[1]
        odom_ground_truth_enu_msg.pose.pose.orientation.z = q[2]
        odom_ground_truth_enu_msg.pose.pose.orientation.w = q[3]

        odom_ground_truth_enu_msg.twist.twist.linear.x = linear_velocity
        odom_ground_truth_enu_msg.twist.twist.linear.y = 0.0
        odom_ground_truth_enu_msg.twist.twist.linear.z = 0.0

        odom_ground_truth_enu_msg.twist.twist.angular.x = 0.0
        odom_ground_truth_enu_msg.twist.twist.angular.y = 0.0
        odom_ground_truth_enu_msg.twist.twist.angular.z = angular_velocity

        cov = np.zeros((6, 6), dtype=float)
        odom_ground_truth_enu_msg.pose.covariance = cov.flatten().tolist()

        self.odom_ground_truth_enu_pub.publish(odom_ground_truth_enu_msg)

        self.get_logger().debug(
            f"[GT] ned=({x:.2f},{y:.2f}) -> enu=({y:.2f},{x:.2f}) theta_enu={theta:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)

    graph_slam = GraphSlam()

    rclpy.spin(graph_slam)

    graph_slam.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()