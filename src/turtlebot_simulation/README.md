# turtlebot_simulation — the Stonefish world, robot and sensors (Part 3)

This package holds **everything that describes the simulated world** for Part 3:
the circuit the robot drives, the robot itself, and the sensors bolted to it. It
contains almost no code — it's assets plus one bringup launch file.

Think of it as the Stonefish equivalent of what `tb3_multi_robot` is for Gazebo
in Parts 2 and 4.

## Scenario files (`scenarios/*.scn`)

A `.scn` file is XML describing a Stonefish world. They compose by inclusion, so
the one Part 3 launches is small and pulls in the rest:

```
turtlebot_hoi_circuit1.scn        <- what the SLAM launch loads
   ├── include circuit1.scn        the walls the robot maps
   └── include turtlebot_featherstone.scn   the robot + its sensors
```

Assets are referenced with Stonefish's own `$(find package_name)/...` syntax —
the same idea as ROS's `FindPackageShare` — so nothing is tied to one machine's
folder layout.

## The robot's sensors (declared in `turtlebot_featherstone.scn`)

| Sensor | Type | Rate | Topic | Used by SLAM? |
|---|---|---|---|---|
| `rplidar` | `multibeam` | 5 Hz | `/turtlebot/scan` | **yes** — the whole line-feature pipeline |
| `imu` | `imu` | 10 Hz | `/turtlebot/sensors/imu_data` | **yes** — heading correction |
| `odometry` | `odometry` | 100 Hz | `/turtlebot/odom_ground_truth` | evaluation only |
| `realsense_color` | `camera` | 30 Hz | — | no |
| `realsense_depth` | `depthcamera` | 30 Hz | — | no |

The `multibeam` LiDAR is a **CPU raycast**; the two cameras are **GPU-rendered**.
That's why the headless (`nogpu`) Stonefish binary can still run Part 3's SLAM:
it loses the cameras, which SLAM never uses.

## `launch/turtlebot_basic.launch.py`

The bringup. It starts the Stonefish simulator node, publishes the robot's URDF
via `robot_state_publisher`, runs `scripts/swiftpro_controller.py` (the arm
controller — note this is a *script here*, **not** the separate `swiftpro`
package) and `scripts/diff_drive_odometry.py` (which broadcasts the odom→base
TF; this is why `localization_final` sets `publish_tf: false`).

`localization_final/launch/graph_slam_updated.launch.py` includes this launch and
overrides `scenario_description` to point at `turtlebot_hoi_circuit1.scn`.

## Depends on

`stonefish_ros2` (the simulator bridge), plus the three description packages that
supply meshes: `kobuki_description`, `swiftpro_description`,
`turtlebot_description`.
