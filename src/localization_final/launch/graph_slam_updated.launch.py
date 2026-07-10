# Simulation launch: Stonefish bringup + Graph SLAM stack for the turtlebot_hoi_circuit1 scenario.
# LiDAR is published in NED frame; the line extractor applies a π offset to convert to ENU.
# publish_tf is disabled to prevent a TF conflict with the Stonefish diff_drive_odometry broadcaster.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def robot_frame(frame_suffix):
    return PythonExpression(
        [
            "'",
            LaunchConfiguration("robot_name"),
            frame_suffix,
            "'",
        ]
    )


def robot_topic(topic_suffix):
    return PythonExpression(
        [
            "'/",
            LaunchConfiguration("robot_name"),
            topic_suffix,
            "'",
        ]
    )


def generate_launch_description():
    pkg_sim = FindPackageShare("turtlebot_simulation")
    pkg_self = FindPackageShare("localization_final")

    args = [
        DeclareLaunchArgument("robot_name", default_value="turtlebot"),

        DeclareLaunchArgument(
            "simulation_data",
            default_value=PathJoinSubstitution([pkg_sim, "resources"]),
        ),

        DeclareLaunchArgument(
            "scenario_description",
            default_value=PathJoinSubstitution(
                [
                    pkg_sim,
                    "scenarios",
                    "turtlebot_hoi_circuit1.scn",
                ]
            ),
        ),

        DeclareLaunchArgument("simulation_rate", default_value="1000.0"),
        DeclareLaunchArgument("window_resolution_x", default_value="1200"),
        DeclareLaunchArgument("window_resolution_y", default_value="800"),
        DeclareLaunchArgument("rendering_quality", default_value="high"),

        # Process noise std-dev [rad/s] — Q = diag(σ_L², σ_R²) in the unicycle kinematic model
        DeclareLaunchArgument("wheel_vel_noise_left", default_value="0.10"),
        DeclareLaunchArgument("wheel_vel_noise_right", default_value="0.10"),

        # RANSAC line extraction parameters
        DeclareLaunchArgument("min_points_per_line", default_value="8"),
        DeclareLaunchArgument("max_line_distance", default_value="0.03"),   # d_max inlier threshold [m]
        DeclareLaunchArgument("max_gap_distance", default_value="0.25"),
        DeclareLaunchArgument("max_iterations", default_value="120"),
        DeclareLaunchArgument("min_line_length", default_value="0.20"),
        DeclareLaunchArgument("max_range", default_value="8.0"),

        DeclareLaunchArgument("min_inlier_ratio", default_value="0.03"),
        # Geometry duplicate gate thresholds for pending-to-confirmed rescue and segment merging
        DeclareLaunchArgument("duplicate_rho_threshold", default_value="0.15"),
        DeclareLaunchArgument("duplicate_alpha_threshold", default_value="0.15"),

        # ICNN data association: χ²(confidence_level, dof=2) gate; pending buffer promotion threshold
        DeclareLaunchArgument("confidence_level", default_value="0.97"),
        DeclareLaunchArgument("confirmed_min_observations", default_value="2"),
        DeclareLaunchArgument("new_landmark_min_observations", default_value="2"),
        DeclareLaunchArgument("max_landmark_segment_length", default_value="2.50"),
        # iSAM2 optimisation trigger: run every key_frame_update sensor-fused nodes after num_min_key poses
        DeclareLaunchArgument("key_frame_update", default_value="20"),
        DeclareLaunchArgument("num_min_key", default_value="10"),
        DeclareLaunchArgument("map_publish_period", default_value="0.50"),
        DeclareLaunchArgument("allow_landmark_reuse", default_value="true"),
        DeclareLaunchArgument("segment_merge_gap_distance", default_value="0.35"),
        # Measurement uncertainty floors — prevent overconfident polar line associations on sensor-noise-limited scans
        DeclareLaunchArgument("measurement_sigma_rho_floor", default_value="0.04"),
        DeclareLaunchArgument("measurement_sigma_alpha_floor", default_value="0.06"),
        DeclareLaunchArgument("visualization_segment_merge_gap", default_value="0.40"),

        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution(
                [
                    pkg_self,
                    "rviz",
                    "task2_graphslam_minimal.rviz",
                ]
            ),
        ),
    ]

    turtlebot_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    pkg_sim,
                    "launch",
                    "turtlebot_basic.launch.py",
                ]
            )
        ),
        launch_arguments={
            "robot_name": LaunchConfiguration("robot_name"),
            "simulation_data": LaunchConfiguration("simulation_data"),
            "scenario_description": LaunchConfiguration("scenario_description"),
            "simulation_rate": LaunchConfiguration("simulation_rate"),
            "window_resolution_x": LaunchConfiguration("window_resolution_x"),
            "window_resolution_y": LaunchConfiguration("window_resolution_y"),
            "rendering_quality": LaunchConfiguration("rendering_quality"),
            "rviz_config": LaunchConfiguration("rviz_config"),
        }.items(),
    )

    graph_slam = Node(
        package="localization_final",
        executable="task2_data_association_localization",
        name="task2_data_association_localization",
        output="screen",
        emulate_tty=True,
        respawn=False,
        parameters=[
            {
                "odom_frame": ParameterValue("world_enu", value_type=str),

                "base_frame": ParameterValue(
                    robot_frame("/base_footprint"),
                    value_type=str,
                ),

                "wheel_left_joint_name": ParameterValue(
                    robot_frame("/wheel_left_joint"),
                    value_type=str,
                ),

                "wheel_right_joint_name": ParameterValue(
                    robot_frame("/wheel_right_joint"),
                    value_type=str,
                ),

                "scan_topic": ParameterValue(
                    robot_topic("/scan"),
                    value_type=str,
                ),

                "joint_states_topic": ParameterValue(
                    robot_topic("/joint_states"),
                    value_type=str,
                ),

                "imu_topic": ParameterValue(
                    robot_topic("/sensors/imu_data"),
                    value_type=str,
                ),

                "ground_truth_topic": ParameterValue(
                    robot_topic("/odom_ground_truth"),
                    value_type=str,
                ),

                "odom_topic": ParameterValue(
                    "/odom_graphslam",
                    value_type=str,
                ),

                "ground_truth_enu_topic": ParameterValue(
                    "/turtlebot/odom_ground_truth_enu",
                    value_type=str,
                ),

                "landmark_topic": ParameterValue(
                    "/slam_landmarks",
                    value_type=str,
                ),

                # Disabled to prevent TF conflict with diff_drive_odometry and stonefish_simulator
                "publish_tf": ParameterValue(
                    False,
                    value_type=bool,
                ),

                "graphslam_tf_child_frame": ParameterValue(
                    "graphslam/base_footprint",
                    value_type=str,
                ),

                "wheel_radius": 0.035,
                "wheel_base_distance": 0.230,

                "wheel_vel_noise_left": ParameterValue(
                    LaunchConfiguration("wheel_vel_noise_left"),
                    value_type=float,
                ),

                "wheel_vel_noise_right": ParameterValue(
                    LaunchConfiguration("wheel_vel_noise_right"),
                    value_type=float,
                ),

                "min_points_per_line": ParameterValue(
                    LaunchConfiguration("min_points_per_line"),
                    value_type=int,
                ),

                "max_line_distance": ParameterValue(
                    LaunchConfiguration("max_line_distance"),
                    value_type=float,
                ),

                "max_gap_distance": ParameterValue(
                    LaunchConfiguration("max_gap_distance"),
                    value_type=float,
                ),

                "max_iterations": ParameterValue(
                    LaunchConfiguration("max_iterations"),
                    value_type=int,
                ),

                "min_line_length": ParameterValue(
                    LaunchConfiguration("min_line_length"),
                    value_type=float,
                ),

                "max_range": ParameterValue(
                    LaunchConfiguration("max_range"),
                    value_type=float,
                ),

                "min_inlier_ratio": ParameterValue(
                    LaunchConfiguration("min_inlier_ratio"),
                    value_type=float,
                ),

                "duplicate_rho_threshold": ParameterValue(
                    LaunchConfiguration("duplicate_rho_threshold"),
                    value_type=float,
                ),

                "duplicate_alpha_threshold": ParameterValue(
                    LaunchConfiguration("duplicate_alpha_threshold"),
                    value_type=float,
                ),

                "confidence_level": ParameterValue(
                    LaunchConfiguration("confidence_level"),
                    value_type=float,
                ),

                "confirmed_min_observations": ParameterValue(
                    LaunchConfiguration("confirmed_min_observations"),
                    value_type=int,
                ),

                "max_landmark_segment_length": ParameterValue(
                    LaunchConfiguration("max_landmark_segment_length"),
                    value_type=float,
                ),

                "key_frame_update": ParameterValue(
                    LaunchConfiguration("key_frame_update"),
                    value_type=int,
                ),

                "num_min_key": ParameterValue(
                    LaunchConfiguration("num_min_key"),
                    value_type=int,
                ),

                "map_publish_period": ParameterValue(
                    LaunchConfiguration("map_publish_period"),
                    value_type=float,
                ),

                "visualization_segment_merge_gap": ParameterValue(
                    LaunchConfiguration("visualization_segment_merge_gap"),
                    value_type=float,
                ),
            }
        ],
    )

    arm_tuck = Node(
        package="localization_final",
        executable="arm_tuck",
        name="arm_tuck",
        output="screen",
        emulate_tty=True,
    )

    polar_line_extractor = Node(
        package="localization_final",
        executable="task2_polar_line_extractor",
        name="task2_polar_line_extractor",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "scan_topic": ParameterValue(
                    robot_topic("/scan"),
                    value_type=str,
                ),

                "scan_points_topic": ParameterValue(
                    robot_topic("/scan_points"),
                    value_type=str,
                ),

                "line_segments_topic": ParameterValue(
                    robot_topic("/line_segments"),
                    value_type=str,
                ),

                "scan_frame": ParameterValue(
                    robot_frame("/base_footprint"),
                    value_type=str,
                ),

                "min_points_per_line": ParameterValue(
                    LaunchConfiguration("min_points_per_line"),
                    value_type=int,
                ),

                "max_line_distance": ParameterValue(
                    LaunchConfiguration("max_line_distance"),
                    value_type=float,
                ),

                "max_gap_distance": ParameterValue(
                    LaunchConfiguration("max_gap_distance"),
                    value_type=float,
                ),

                "max_iterations": ParameterValue(
                    LaunchConfiguration("max_iterations"),
                    value_type=int,
                ),

                "min_line_length": ParameterValue(
                    LaunchConfiguration("min_line_length"),
                    value_type=float,
                ),

                "max_range": ParameterValue(
                    LaunchConfiguration("max_range"),
                    value_type=float,
                ),

                "min_inlier_ratio": ParameterValue(
                    LaunchConfiguration("min_inlier_ratio"),
                    value_type=float,
                ),

                "duplicate_rho_threshold": ParameterValue(
                    LaunchConfiguration("duplicate_rho_threshold"),
                    value_type=float,
                ),

                "duplicate_alpha_threshold": ParameterValue(
                    LaunchConfiguration("duplicate_alpha_threshold"),
                    value_type=float,
                ),
            }
        ],
    )

    return LaunchDescription(
        args
        + [
            arm_tuck,
            turtlebot_bringup,
            graph_slam,
            polar_line_extractor,
        ]
    )
