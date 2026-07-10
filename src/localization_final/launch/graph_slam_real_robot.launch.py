#!/usr/bin/env python3
# Graph SLAM bringup for the real TurtleBot.
# /wheel_vels is the primary odometry source; /joint_states serves as fallback.
# publish_tf=True — SLAM broadcasts the corrected map→odom transform.

import tempfile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

SCAN_TOPIC         = '/scan'
WHEEL_VELS_TOPIC   = '/wheel_vels'
JOINT_STATES_TOPIC = '/joint_states'
IMU_TOPIC          = '/imu'
MAP_FRAME          = 'map'
ODOM_FRAME         = 'odom'
BASE_FRAME         = 'base_link'
WHEEL_LEFT_JOINT   = 'left_wheel_joint'
WHEEL_RIGHT_JOINT  = 'right_wheel_joint'
WHEEL_RADIUS        = 0.035
WHEEL_BASE_DISTANCE = 0.230

RVIZ_CONFIG_CONTENT = """
Panels:
  - Class: rviz_common/Displays
    Help Height: 0
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /GraphSLAM_Odometry1
        - /LaserScan1
        - /Landmarks1
      Splitter Ratio: 0.6
    Tree Height: 600
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Views
    Name: Views
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Name: Grid
    - Class: rviz_default_plugins/RobotModel
      Enabled: true
      Name: RobotModel
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Value: /robot_description
    - Class: rviz_default_plugins/Odometry
      Enabled: true
      Name: GraphSLAM_Odometry
      Topic:
        Depth: 3000
        Durability Policy: Volatile
        Value: /odom_graphslam
      Position Tolerance: 0.01
      Angle Tolerance: 0.01
      Keep: 3000
      Shape:
        Alpha: 1
        Axes Length: 0.3
        Axes Radius: 0.01
        Color: 255; 100; 0
        Head Length: 0.07
        Head Radius: 0.03
        Shaft Length: 0.23
        Shaft Radius: 0.01
        Value: Arrow
    - Class: rviz_default_plugins/LaserScan
      Enabled: true
      Name: LaserScan
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Value: /scan
      Size (m): 0.02
      Size (Pixels): 3
      Style: Points
      Color Transformer: Intensity
      Use Fixed Frame: true
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: Landmarks
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Value: /slam_landmarks
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: LineSegments
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Value: /line_segments
    - Class: rviz_default_plugins/Odometry
      Enabled: true
      Name: WheelOdom
      Topic:
        Depth: 1
        Durability Policy: Volatile
        Value: /odom
      Position Tolerance: 0.05
      Angle Tolerance: 0.05
      Keep: 1
      Shape:
        Alpha: 0.5
        Color: 0; 180; 255
        Value: Arrow
  Enabled: true
  Global Options:
    Background Color: 30; 30; 40
    Fixed Frame: map
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
  Views:
    Current:
      Class: rviz_default_plugins/TopDownOrtho
      Name: Current View
      Near Clip Distance: 0.01
      Scale: 100
      Target Frame: map
      X: 0
      Y: 0
Window Geometry:
  Displays:
    collapsed: false
  Height: 900
  QMainWindow State: ""
  Width: 1400
"""


def generate_rviz_config(context, *args, **kwargs):
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='_real_robot.rviz', delete=False)
    tmp.write(RVIZ_CONFIG_CONTENT)
    tmp.flush()
    tmp.close()
    return [Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', tmp.name],
        output='screen',
    )]


def generate_launch_description():

    args = [
        DeclareLaunchArgument('wheel_vel_noise_left',  default_value='0.08'),
        DeclareLaunchArgument('wheel_vel_noise_right', default_value='0.08'),
        DeclareLaunchArgument('min_points_per_line',   default_value='8'),
        DeclareLaunchArgument('max_line_distance',     default_value='0.05'),
        DeclareLaunchArgument('max_gap_distance',      default_value='0.25'),
        DeclareLaunchArgument('max_iterations',        default_value='120'),
        DeclareLaunchArgument('min_line_length',       default_value='0.20'),
        DeclareLaunchArgument('max_range',             default_value='6.0'),
        DeclareLaunchArgument('min_inlier_ratio',      default_value='0.03'),
        DeclareLaunchArgument('duplicate_rho_threshold',   default_value='0.08'),
        DeclareLaunchArgument('duplicate_alpha_threshold', default_value='0.08'),
        DeclareLaunchArgument('confidence_level',          default_value='0.97'),
        DeclareLaunchArgument('confirmed_min_observations',    default_value='3'),
        DeclareLaunchArgument('new_landmark_min_observations', default_value='20'),
        DeclareLaunchArgument('allow_landmark_reuse',          default_value='true'),
        DeclareLaunchArgument('max_landmark_segment_length',   default_value='2.50'),
        DeclareLaunchArgument('measurement_sigma_rho_floor',   default_value='0.04'),
        DeclareLaunchArgument('measurement_sigma_alpha_floor', default_value='0.06'),
        DeclareLaunchArgument('segment_merge_gap_distance',      default_value='0.30'),
        DeclareLaunchArgument('visualization_segment_merge_gap', default_value='0.30'),
        DeclareLaunchArgument('key_frame_update',   default_value='20'),
        DeclareLaunchArgument('num_min_key',        default_value='10'),
        DeclareLaunchArgument('map_publish_period', default_value='0.50'),
    ]

    arm_tuck = Node(
        package='localization_final',
        executable='arm_tuck',
        name='arm_tuck',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'joint_state_topic': JOINT_STATES_TOPIC,
            'command_topic': '/turtlebot/swiftpro/joint_velocity_controller/command',
            'target_joint1': 0.00,
            'target_joint2': -0.25,
            'target_joint3': -1.50,
            'target_joint4': 0.00,
        }],
    )

    graph_slam = Node(
        package='localization_final',
        executable='task2_data_association_localization',
        name='task2_data_association_localization',
        output='screen',
        emulate_tty=True,
        respawn=False,
        parameters=[{
            # SLAM publishes corrected map→odom; loop closures appear as odom frame drift
            'odom_frame':  MAP_FRAME,
            'base_frame':  BASE_FRAME,

            'wheel_left_joint_name':  WHEEL_LEFT_JOINT,
            'wheel_right_joint_name': WHEEL_RIGHT_JOINT,

            'scan_topic':             SCAN_TOPIC,
            'joint_states_topic':     JOINT_STATES_TOPIC,
            'imu_topic':              IMU_TOPIC,
            'wheel_vels_topic':       WHEEL_VELS_TOPIC,
            'odom_topic':             '/odom_graphslam',
            'ground_truth_topic':     '/odom',
            'ground_truth_enu_topic': '/odom',
            'landmark_topic':         '/slam_landmarks',

            # graphslam_tf_child_frame=odom closes the TF chain: map→odom→base_link
            'publish_tf':               True,
            'real_robot':               True,
            'graphslam_tf_child_frame': ODOM_FRAME,

            'wheel_radius':        WHEEL_RADIUS,
            'wheel_base_distance': WHEEL_BASE_DISTANCE,

            'wheel_vel_noise_left': ParameterValue(
                LaunchConfiguration('wheel_vel_noise_left'), value_type=float),
            'wheel_vel_noise_right': ParameterValue(
                LaunchConfiguration('wheel_vel_noise_right'), value_type=float),

            'min_points_per_line': ParameterValue(
                LaunchConfiguration('min_points_per_line'), value_type=int),
            'max_line_distance': ParameterValue(
                LaunchConfiguration('max_line_distance'), value_type=float),
            'max_gap_distance': ParameterValue(
                LaunchConfiguration('max_gap_distance'), value_type=float),
            'max_iterations': ParameterValue(
                LaunchConfiguration('max_iterations'), value_type=int),
            'min_line_length': ParameterValue(
                LaunchConfiguration('min_line_length'), value_type=float),
            'max_range': ParameterValue(
                LaunchConfiguration('max_range'), value_type=float),
            'min_inlier_ratio': ParameterValue(
                LaunchConfiguration('min_inlier_ratio'), value_type=float),

            'duplicate_rho_threshold': ParameterValue(
                LaunchConfiguration('duplicate_rho_threshold'), value_type=float),
            'duplicate_alpha_threshold': ParameterValue(
                LaunchConfiguration('duplicate_alpha_threshold'), value_type=float),
            'confidence_level': ParameterValue(
                LaunchConfiguration('confidence_level'), value_type=float),
            'confirmed_min_observations': ParameterValue(
                LaunchConfiguration('confirmed_min_observations'), value_type=int),
            'new_landmark_min_observations': ParameterValue(
                LaunchConfiguration('new_landmark_min_observations'), value_type=int),
            'allow_landmark_reuse': ParameterValue(
                LaunchConfiguration('allow_landmark_reuse'), value_type=bool),

            'measurement_sigma_rho_floor': ParameterValue(
                LaunchConfiguration('measurement_sigma_rho_floor'), value_type=float),
            'measurement_sigma_alpha_floor': ParameterValue(
                LaunchConfiguration('measurement_sigma_alpha_floor'), value_type=float),

            'max_landmark_segment_length': ParameterValue(
                LaunchConfiguration('max_landmark_segment_length'), value_type=float),
            'segment_merge_gap_distance': ParameterValue(
                LaunchConfiguration('segment_merge_gap_distance'), value_type=float),
            'visualization_segment_merge_gap': ParameterValue(
                LaunchConfiguration('visualization_segment_merge_gap'), value_type=float),

            'key_frame_update': ParameterValue(
                LaunchConfiguration('key_frame_update'), value_type=int),
            'num_min_key': ParameterValue(
                LaunchConfiguration('num_min_key'), value_type=int),
            'map_publish_period': ParameterValue(
                LaunchConfiguration('map_publish_period'), value_type=float),
        }],
    )

    polar_line_extractor = Node(
        package='localization_final',
        executable='task2_polar_line_extractor',
        name='task2_polar_line_extractor',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'real_robot':          True,
            'scan_topic':          SCAN_TOPIC,
            'scan_points_topic':   '/scan_points',
            'line_segments_topic': '/line_segments',
            'scan_frame':          BASE_FRAME,
            'min_points_per_line': ParameterValue(
                LaunchConfiguration('min_points_per_line'), value_type=int),
            'max_line_distance': ParameterValue(
                LaunchConfiguration('max_line_distance'), value_type=float),
            'max_gap_distance': ParameterValue(
                LaunchConfiguration('max_gap_distance'), value_type=float),
            'max_iterations': ParameterValue(
                LaunchConfiguration('max_iterations'), value_type=int),
            'min_line_length': ParameterValue(
                LaunchConfiguration('min_line_length'), value_type=float),
            'max_range': ParameterValue(
                LaunchConfiguration('max_range'), value_type=float),
            'min_inlier_ratio': ParameterValue(
                LaunchConfiguration('min_inlier_ratio'), value_type=float),
            'duplicate_rho_threshold': ParameterValue(
                LaunchConfiguration('duplicate_rho_threshold'), value_type=float),
            'duplicate_alpha_threshold': ParameterValue(
                LaunchConfiguration('duplicate_alpha_threshold'), value_type=float),
        }],
    )

    # Identity prior for map→odom; the SLAM node overrides this with a corrected transform at runtime
    static_map_odom = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_odom_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen',
    )

    rviz_launcher = OpaqueFunction(function=generate_rviz_config)

    return LaunchDescription(
        args + [
            static_map_odom,
            arm_tuck,
            graph_slam,
            polar_line_extractor,
            rviz_launcher,
        ]
    )