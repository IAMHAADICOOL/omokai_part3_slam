import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from scipy.linalg import block_diag
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from .utils import warp_angle


class RansacLineExtractor:
    def __init__(
        self,
        min_points_per_line=10,
        max_line_distance=0.05,
        max_gap_distance=0.25,
        max_iterations=120,
        min_line_length=0.30,
        max_range=8.0,
        min_inlier_ratio=0.03,
        duplicate_rho_threshold=0.15,
        duplicate_alpha_threshold=0.15,
        segment_merge_gap_distance=0.35,
        measurement_sigma_rho_floor=0.04,
        measurement_sigma_alpha_floor=0.06,
        lidar_angle_offset=math.pi,
        real_robot=False,
    ):
        self.min_points_per_line = int(min_points_per_line)
        self.max_line_distance = float(max_line_distance)
        self.max_gap_distance = float(max_gap_distance)
        self.max_iterations = int(max_iterations)
        self.min_line_length = float(min_line_length)
        self.max_range = float(max_range)
        self.min_inlier_ratio = float(min_inlier_ratio)

        self.duplicate_rho_threshold = float(duplicate_rho_threshold)
        self.duplicate_alpha_threshold = float(duplicate_alpha_threshold)
        self.segment_merge_gap_distance = float(segment_merge_gap_distance)

        self.measurement_sigma_rho_floor = float(measurement_sigma_rho_floor)
        self.measurement_sigma_alpha_floor = float(measurement_sigma_alpha_floor)
        self.lidar_angle_offset = float(lidar_angle_offset)
        self.real_robot = bool(real_robot)

        self.rng = np.random.default_rng()

    def transform_lidar_to_cartesian(self, lidar_msg):
        lidar_points = []

        angle_increment = lidar_msg.angle_increment

        if self.real_robot:
            # Real RPLidar: angles are already in robot ENU frame, scan direction is CCW
            current_angle = lidar_msg.angle_min
            for range_value in lidar_msg.ranges:
                if (
                    math.isfinite(range_value)
                    and range_value >= lidar_msg.range_min
                    and range_value <= lidar_msg.range_max
                    and range_value <= self.max_range
                ):
                    x_point = math.cos(current_angle) * range_value
                    y_point = math.sin(current_angle) * range_value
                    lidar_points.append(np.array([x_point, y_point], dtype=np.float64))
                current_angle += angle_increment
        else:
            # Stonefish LiDAR publishes in NED frame with CW scan order; apply π offset and reverse iteration to convert to ENU
            current_angle = lidar_msg.angle_min + math.pi
            for range_value in lidar_msg.ranges:
                if (
                    math.isfinite(range_value)
                    and range_value >= lidar_msg.range_min
                    and range_value <= lidar_msg.range_max
                    and range_value <= self.max_range
                ):
                    x_point = math.cos(current_angle) * range_value
                    y_point = math.sin(current_angle) * range_value
                    lidar_points.append(np.array([x_point, y_point], dtype=np.float64))
                current_angle -= angle_increment

        return lidar_points

    def extract_line_segments(self, lidar_points):
        # Iterative RANSAC loop: at each iteration the best-fitting line is extracted,
        # its inliers are refined via SVD, split at physical gaps, and consumed from the remaining point set
        if len(lidar_points) < self.min_points_per_line:
            return []

        remaining = np.asarray(lidar_points, dtype=np.float64)
        original_count = remaining.shape[0]

        line_segments = []

        while remaining.shape[0] >= self.min_points_per_line:
            best = self._ransac_best_line(remaining)

            if best is None:
                break

            best_inlier_mask, best_count = best

            if best_count < self.min_points_per_line:
                break

            # Global inlier ratio guard rejects sparse hypotheses that arise from noise clusters
            if self.min_inlier_ratio > 0.0:
                inlier_ratio = best_count / max(1, original_count)

                if inlier_ratio < self.min_inlier_ratio:
                    break

            inliers = remaining[best_inlier_mask]
            outliers = remaining[~best_inlier_mask]

            # First SVD refit on raw RANSAC inliers to obtain a minimum-variance direction estimate
            refit = self._refit_line_svd(inliers)

            if refit is None:
                remaining = outliers
                continue

            _, direction, point_on_line = refit

            # Gap filter splits the inlier set at discontinuities exceeding max_gap_distance;
            # the largest contiguous group is retained and remainder returned to the pool
            kept_inliers, returned_inliers = self._gap_filter(
                inliers,
                direction,
                point_on_line,
            )

            if kept_inliers.shape[0] < self.min_points_per_line:
                remaining = self._stack_points(outliers, returned_inliers)
                continue

            # Second SVD refit on gap-filtered inliers for a stable final line estimate
            refit = self._refit_line_svd(kept_inliers)

            if refit is None:
                remaining = self._stack_points(outliers, returned_inliers)
                continue

            _, direction, point_on_line = refit

            start_point, end_point, line_length = self._line_segment_from_inliers(
                kept_inliers,
                direction,
                point_on_line,
            )

            if line_length >= self.min_line_length:
                line_segments.append([start_point, end_point])

            remaining = self._stack_points(outliers, returned_inliers)

        # Merge collinear segments that RANSAC split across adjacent iterations
        line_segments = self.merge_duplicate_segments(line_segments)

        return line_segments

    def fit_line(self, point1, point2):
        a = point2[1] - point1[1]
        b = point1[0] - point2[0]
        c = point1[1] * point2[0] - point1[0] * point2[1]

        return a, b, c

    def distance_from_line(self, point, line_params):
        a, b, c = line_params

        denominator = math.hypot(a, b)

        if denominator < 1e-12:
            return math.inf

        return abs(a * point[0] + b * point[1] + c) / denominator

    def calculate_polar_coordinates(self, line_segments):
        # Converts each Cartesian line segment (endpoints) to Hough normal form (ρ, α):
        #   ρ = −c / √(a²+b²),  α = atan2(b, a)
        # where ax + by + c = 0 is the endpoint-fitted line equation.
        # Negative ρ values are normalised to ρ > 0 by reflecting across the origin.
        polar_coordinates = []

        for line in line_segments:
            a, b, c = self.fit_line(line[0], line[1])

            normalizer = math.hypot(a, b)

            if normalizer < 1e-12:
                continue

            rho = -c / normalizer
            alpha = math.atan2(b, a)

            if rho < 0.0:
                rho = -rho
                alpha = warp_angle(alpha + math.pi)
            else:
                alpha = warp_angle(alpha)

            polar_coordinates.append([float(rho), float(alpha)])

        return polar_coordinates

    def calculate_covariance_matrix(self, line):
        # Propagates endpoint position uncertainty (σ_r = 0.02 m isotropic) to polar parameter uncertainty via Jacobian:
        #   R_i = J · diag(cov_p1, cov_p2) · J^T
        # where J = ∂(ρ, α)/∂(x1, y1, x2, y2) is derived analytically from the line equation ax + by + c = 0
        sigma_r = 0.02

        cov_max_2d_point = np.array(
            [[sigma_r**2, 0.0], [0.0, sigma_r**2]],
            dtype=float,
        )

        a, b, c = self.fit_line(line[0], line[1])

        norm_sq = a**2 + b**2
        norm = math.sqrt(norm_sq)

        if norm < 1e-12:
            return np.diag(
                [
                    self.measurement_sigma_rho_floor**2,
                    self.measurement_sigma_alpha_floor**2,
                ]
            )

        signed_c = c if abs(c) > 1e-12 else 1e-12

        # Partial derivatives of ρ = |c|/‖(a,b)‖ with respect to line coefficients
        dr_da = -abs(signed_c) * a / math.sqrt(norm_sq**3)
        dr_db = -abs(signed_c) * b / math.sqrt(norm_sq**3)
        dr_dc = math.copysign(1.0, signed_c) / norm

        # Partial derivatives of α = atan2(b, a)
        dtheta_da = -b / norm_sq
        dtheta_db = a / norm_sq

        # Chain rule: ∂(a, b, c)/∂(x1, y1, x2, y2) from the two-point line formula
        da_dy1 = -1.0
        da_dy2 = 1.0

        db_dx1 = 1.0
        db_dx2 = -1.0

        dc_dx1 = -line[1][1]
        dc_dy1 = line[1][0]
        dc_dx2 = line[0][1]
        dc_dy2 = -line[0][0]

        jacobian = np.array(
            [
                [
                    dr_dc * dc_dx1 + dr_db * db_dx1,
                    dr_dc * dc_dy1 + dr_da * da_dy1,
                    dr_dc * dc_dx2 + dr_db * db_dx2,
                    dr_dc * dc_dy2 + dr_da * da_dy2,
                ],
                [
                    dtheta_db * db_dx1,
                    dtheta_da * da_dy1,
                    dtheta_db * db_dx2,
                    dtheta_da * da_dy2,
                ],
            ],
            dtype=float,
        )

        cov_line = (
            jacobian
            @ block_diag(cov_max_2d_point, cov_max_2d_point)
            @ jacobian.T
        )

        cov_line += np.diag([1e-6, 1e-6])
        cov_line = 0.5 * (cov_line + cov_line.T)

        # Enforce measurement uncertainty floors to prevent overconfident associations
        cov_line[0, 0] = max(cov_line[0, 0], self.measurement_sigma_rho_floor**2)
        cov_line[1, 1] = max(cov_line[1, 1], self.measurement_sigma_alpha_floor**2)

        return cov_line

    def merge_duplicate_segments(self, line_segments):
        if len(line_segments) <= 1:
            return line_segments

        segments = [
            [np.asarray(line[0], dtype=np.float64), np.asarray(line[1], dtype=np.float64)]
            for line in line_segments
        ]

        changed = True

        while changed:
            changed = False
            used = [False for _ in segments]
            merged_segments = []

            for i in range(len(segments)):
                if used[i]:
                    continue

                current = segments[i]
                used[i] = True

                for j in range(i + 1, len(segments)):
                    if used[j]:
                        continue

                    if self._can_merge_segments(current, segments[j]):
                        current = self._merge_two_segments(current, segments[j])
                        used[j] = True
                        changed = True

                merged_segments.append(current)

            segments = merged_segments

        return segments

    def _segment_polar(self, segment):
        polar = self.calculate_polar_coordinates([segment])

        if len(polar) == 0:
            return None

        return polar[0]

    def _segment_interval_on_line(self, segment, rho, alpha):
        normal = np.array([math.cos(alpha), math.sin(alpha)], dtype=np.float64)
        direction = np.array([-math.sin(alpha), math.cos(alpha)], dtype=np.float64)

        center = rho * normal

        p1 = np.asarray(segment[0], dtype=np.float64)
        p2 = np.asarray(segment[1], dtype=np.float64)

        t1 = float((p1 - center) @ direction)
        t2 = float((p2 - center) @ direction)

        return min(t1, t2), max(t1, t2), direction, center

    def _can_merge_segments(self, segment_a, segment_b):
        polar_a = self._segment_polar(segment_a)
        polar_b = self._segment_polar(segment_b)

        if polar_a is None or polar_b is None:
            return False

        rho_a, alpha_a = polar_a
        rho_b, alpha_b = polar_b

        rho_close = abs(rho_a - rho_b) <= self.duplicate_rho_threshold
        alpha_close = abs(warp_angle(alpha_a - alpha_b)) <= self.duplicate_alpha_threshold

        if not rho_close or not alpha_close:
            return False

        a_min, a_max, _, _ = self._segment_interval_on_line(
            segment_a,
            rho_a,
            alpha_a,
        )
        b_min, b_max, _, _ = self._segment_interval_on_line(
            segment_b,
            rho_a,
            alpha_a,
        )

        gap = max(0.0, max(a_min, b_min) - min(a_max, b_max))

        return gap <= self.segment_merge_gap_distance

    def _merge_two_segments(self, segment_a, segment_b):
        polar_a = self._segment_polar(segment_a)

        if polar_a is None:
            return segment_a

        rho_a, alpha_a = polar_a

        intervals = []

        for segment in [segment_a, segment_b]:
            t_min, t_max, direction, center = self._segment_interval_on_line(
                segment,
                rho_a,
                alpha_a,
            )
            intervals.append((t_min, t_max))

        t_min = min(interval[0] for interval in intervals)
        t_max = max(interval[1] for interval in intervals)

        start = center + t_min * direction
        end = center + t_max * direction

        return [start, end]

    def _ransac_best_line(self, points):
        # Standard RANSAC: sample 2 points, compute perpendicular distances to the candidate line,
        # classify inliers at threshold d_max, retain the hypothesis with maximum inlier count
        point_count = points.shape[0]

        if point_count < 2:
            return None

        best_count = 0
        best_inlier_mask = None

        for _ in range(self.max_iterations):
            idx = self.rng.choice(point_count, size=2, replace=False)

            point1 = points[idx[0]]
            point2 = points[idx[1]]

            if np.linalg.norm(point2 - point1) < 1e-9:
                continue

            distances = self._point_to_line_distances(points, point1, point2)
            inlier_mask = distances <= self.max_line_distance
            count = int(np.count_nonzero(inlier_mask))

            if count > best_count:
                best_count = count
                best_inlier_mask = inlier_mask

        if best_inlier_mask is None:
            return None

        return best_inlier_mask, best_count

    @staticmethod
    def _point_to_line_distances(points, point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        denominator = math.hypot(y2 - y1, x2 - x1)

        if denominator < 1e-12:
            return np.full(points.shape[0], np.inf)

        numerator = np.abs(
            (y2 - y1) * points[:, 0]
            - (x2 - x1) * points[:, 1]
            + x2 * y1
            - y2 * x1
        )

        return numerator / denominator

    @staticmethod
    def _refit_line_svd(inliers):
        # Minimum-variance line estimate via SVD of the centred inlier matrix.
        # The dominant right singular vector (first row of V^T) is the principal axis of inlier scatter,
        # giving a more stable direction than the two-point RANSAC hypothesis
        if inliers.shape[0] < 2:
            return None

        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid

        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        direction = vh[0]
        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1e-12:
            return None

        direction = direction / direction_norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)

        return normal, direction, centroid

    def _gap_filter(self, inliers, direction, point_on_line):
        rel = inliers - point_on_line
        t_values = rel @ direction

        sort_idx = np.argsort(t_values)

        t_sorted = t_values[sort_idx]
        points_sorted = inliers[sort_idx]

        gaps = np.diff(t_sorted)
        split_positions = np.where(gaps > self.max_gap_distance)[0]

        if len(split_positions) == 0:
            return inliers, np.empty((0, 2), dtype=np.float64)

        starts = np.concatenate([[0], split_positions + 1])
        ends = np.concatenate([split_positions + 1, [len(t_sorted)]])
        sizes = ends - starts

        best_segment = int(np.argmax(sizes))

        kept = points_sorted[starts[best_segment] : ends[best_segment]]

        returned_groups = [
            points_sorted[start:end]
            for idx, (start, end) in enumerate(zip(starts, ends))
            if idx != best_segment
        ]

        returned = (
            np.vstack(returned_groups)
            if returned_groups
            else np.empty((0, 2), dtype=np.float64)
        )

        return kept, returned

    @staticmethod
    def _line_segment_from_inliers(inliers, direction, point_on_line):
        t_values = (inliers - point_on_line) @ direction

        t_min = float(np.min(t_values))
        t_max = float(np.max(t_values))

        start_point = point_on_line + t_min * direction
        end_point = point_on_line + t_max * direction

        line_length = float(np.linalg.norm(end_point - start_point))

        return start_point, end_point, line_length

    @staticmethod
    def _stack_points(first, second):
        if first.size == 0:
            return second.reshape((-1, 2))

        if second.size == 0:
            return first.reshape((-1, 2))

        return np.vstack((first, second))


class LineExtractionNode(Node):
    def __init__(self):
        super().__init__("polar_line_extractor_node")

        self.declare_parameter("scan_topic", "/turtlebot/scan")
        self.declare_parameter("scan_points_topic", "/turtlebot/scan_points")
        self.declare_parameter("line_segments_topic", "/turtlebot/line_segments")

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
        self.declare_parameter("measurement_sigma_rho_floor", 0.04)
        self.declare_parameter("measurement_sigma_alpha_floor", 0.06)
        self.declare_parameter("scan_frame", "turtlebot/base_footprint")
        self.declare_parameter("real_robot", False)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.scan_points_topic = self.get_parameter("scan_points_topic").value
        self.line_segments_topic = self.get_parameter("line_segments_topic").value

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
            measurement_sigma_rho_floor=self.get_parameter(
                "measurement_sigma_rho_floor"
            ).value,
            measurement_sigma_alpha_floor=self.get_parameter(
                "measurement_sigma_alpha_floor"
            ).value,
            real_robot=self.get_parameter("real_robot").value,
        )

        self.scan_frame = self.get_parameter("scan_frame").value

        self.laser_sub_ = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.receive_lidar_scan,
            20,
        )

        self.marker_array_pub_ = self.create_publisher(
            MarkerArray,
            self.scan_points_topic,
            20,
        )

        self.line_array_pub_ = self.create_publisher(
            MarkerArray,
            self.line_segments_topic,
            20,
        )

        self.create_timer(0.5, self.visualize_lidar_points)
        self.create_timer(0.5, self.visualize_line_segments)

        self.lidar_points = None
        self.last_point_marker_count = 0
        self.last_line_marker_count = 0

        self.get_logger().info("[RANSAC] Polar line extractor started.")
        self.get_logger().info(f"[RANSAC] scan_topic={self.scan_topic}")
        self.get_logger().info(f"[RANSAC] line_segments_topic={self.line_segments_topic}")
        self.get_logger().info(f"[RANSAC] scan_frame={self.scan_frame}")

    def receive_lidar_scan(self, msg):
        self.lidar_points = self.line_extractor.transform_lidar_to_cartesian(msg)

    def visualize_lidar_points(self):
        if self.lidar_points is None:
            return

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, point in enumerate(self.lidar_points):
            marker = Marker()

            marker.header.frame_id = self.scan_frame
            marker.header.stamp = now
            marker.ns = "lidar_points"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(point[0])
            marker.pose.position.y = float(point[1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.035
            marker.scale.y = 0.035
            marker.scale.z = 0.035

            marker.color.a = 0.8
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)

        for marker_id in range(len(self.lidar_points), self.last_point_marker_count):
            marker = Marker()
            marker.header.frame_id = self.scan_frame
            marker.header.stamp = now
            marker.ns = "lidar_points"
            marker.id = marker_id
            marker.action = Marker.DELETE
            marker.pose.orientation.w = 1.0
            marker_array.markers.append(marker)

        self.last_point_marker_count = len(self.lidar_points)

        self.marker_array_pub_.publish(marker_array)

    def visualize_line_segments(self):
        if self.lidar_points is None:
            return

        line_segments = self.line_extractor.extract_line_segments(self.lidar_points)
        polar_coordinates = self.line_extractor.calculate_polar_coordinates(
            line_segments
        )

        polar_for_log = [
            (round(float(p[0]), 3), round(float(p[1]), 3))
            for p in polar_coordinates
        ]

        self.get_logger().info(
            f"[RANSAC] lines={len(line_segments)}, polar={polar_for_log}"
        )

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for i, line in enumerate(line_segments):
            start_point = line[0]
            end_point = line[1]

            marker = Marker()

            marker.header.frame_id = self.scan_frame
            marker.header.stamp = now
            marker.ns = "live_ransac_lines_robot_frame"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.points.append(
                Point(
                    x=float(start_point[0]),
                    y=float(start_point[1]),
                    z=0.03,
                )
            )

            marker.points.append(
                Point(
                    x=float(end_point[0]),
                    y=float(end_point[1]),
                    z=0.03,
                )
            )

            marker.scale.x = 0.04

            marker.color.a = 0.95
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0

            marker.pose.orientation.w = 1.0

            marker_array.markers.append(marker)

        for marker_id in range(len(line_segments), self.last_line_marker_count):
            marker = Marker()
            marker.header.frame_id = self.scan_frame
            marker.header.stamp = now
            marker.ns = "live_ransac_lines_robot_frame"
            marker.id = marker_id
            marker.action = Marker.DELETE
            marker.pose.orientation.w = 1.0
            marker_array.markers.append(marker)

        self.last_line_marker_count = len(line_segments)

        self.line_array_pub_.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)

    line_extraction_node = LineExtractionNode()

    rclpy.spin(line_extraction_node)

    line_extraction_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()