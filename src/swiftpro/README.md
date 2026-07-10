# swiftpro — ros2_control hardware interface for the real SwiftPro arm (Part 3)

The `ros2_control` driver that talks to a **physical** uArm SwiftPro over serial.

**The simulation never uses this package.** In simulation the arm is driven by
`turtlebot_simulation/scripts/swiftpro_controller.py`, a plain ROS node — not by
this hardware interface. This package exists so the real-robot launch path
(`localization_final/launch/graph_slam_real_robot.launch.py`) keeps working.

**Why it still gets built:** it declares `controller_manager`,
`hardware_interface` and `pluginlib` dependencies, so the root `Dockerfile`
installs `ros-jazzy-ros2-control` / `ros-jazzy-ros2-controllers` to let the whole
workspace build in one pass. If you want a smaller image and don't care about the
real-robot path, dropping a `COLCON_IGNORE` file in this folder is safe and
nothing in the simulation will notice.

> Fixed on import into this repo: the upstream `package.xml` declared
> `rclcpp_lifcycle` (typo). `rosdep` cannot resolve that key. Corrected to
> `rclcpp_lifecycle`.

**Contains:** `src/` (the hardware interface plugin), `urdf/`, `config/`
(controller YAML), `firmware/`.
