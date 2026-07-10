# stonefish_ros2 — the ROS 2 bridge to the Stonefish simulator (Part 3)

**Third-party, vendored.** This is the official ROS 2 interface to the
[Stonefish](https://github.com/patrykcieslak/stonefish) simulator by Patryk
Cieślak. It is included in `src/` unmodified so the Docker build needs no extra
clone step. Original docs preserved in `UPSTREAM_README.md`.

## What it does

Stonefish itself is a plain C++ physics/rendering library — it knows nothing
about ROS. This package wraps it: it reads a **scenario file** (`.scn`, an XML
description of the world, the robot, and its sensors), runs the simulation, and
publishes each sensor onto a ROS topic while listening for actuator commands.

For Part 3 that means: the LiDAR in the `.scn` becomes `/turtlebot/scan`, the IMU
becomes `/turtlebot/sensors/imu_data`, wheel joints become
`/turtlebot/joint_states`, and so on. Nothing else in this project talks to
Stonefish directly.

## The two executables (this matters for Docker)

| Executable | Needs a GPU? | Use |
|---|---|---|
| `stonefish_simulator` | **yes** — OpenGL 4.3+ | the normal, graphical simulator with a 3D window |
| `stonefish_simulator_nogpu` | no | headless: physics + non-visual sensors only |

Stonefish's rendering is genuinely GPU-bound (see
[the install docs](https://stonefish.readthedocs.io/en/latest/install.html)).
Cameras are rendered on the GPU; the **`multibeam` LiDAR that Part 3's SLAM
depends on is a CPU raycast**, so the headless binary still produces `/scan`,
`/joint_states` and IMU — everything the SLAM needs. The 3D window and the
RealSense camera topics are what you lose.

## Build dependency you cannot skip

`CMakeLists.txt` contains:

```cmake
find_package(Stonefish REQUIRED 1.6.0)
```

The **Stonefish C++ library must already be installed system-wide** before this
package will compile. It is not a ROS package and cannot live in `src/`. The
root `Dockerfile` builds it from source (pinned tag) before running
`colcon build`; `install_native.sh` does the same on a native machine.

## Ports / topics

Everything is driven by the scenario file, not by code here. See
`turtlebot_simulation/scenarios/` for the `.scn` files and which
`<ros_publisher topic="...">` each sensor declares.

## License / attribution

Stonefish and `stonefish_ros2` are by Patryk Cieślak. See `docs/SOURCES.md`.
