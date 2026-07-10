# Omokai Part 3 — Graph SLAM with line features

The robot is dropped into a circuit it has never seen, with **no map**. You drive
it by keyboard. As it moves it turns raw LiDAR points into straight **wall
segments**, and uses those walls to correct its own drifting wheel odometry —
building the map and working out where it is **at the same time**. That is SLAM
(Simultaneous Localization And Mapping).

```
LiDAR scan → RANSAC line features → data association → factor graph → iSAM2
             (walls, not pixels)    (seen before?)     (poses+walls)  (solve)
```

This is **Part 3** of the Omokai take-home. Parts 1, 2 and 4 (the LLM mission
pipeline, multi-robot formations, and vision follow) live in the main repository:

**→ [Omokai — Parts 1, 2, 4](https://github.com/IAMHAADICOOL/omokai_part4_ai_robot_following_submission/tree/main)**


### Why it's a separate repository

Part 3 uses a **different simulator** ([Stonefish](https://stonefish.readthedocs.io/),
not Gazebo) and a **different solver** ([GTSAM](https://gtsam.org/), a factor-graph
library). Neither is needed by the other three parts. Splitting them means you
don't have to build a marine-robotics physics engine from source just to run an
LLM mission planner.

**Every package here has its own `README.md`** explaining what it does. Start with
[`src/localization_final/README.md`](src/localization_final/README.md) — that's
where the SLAM actually lives, and it explains RANSAC, Mahalanobis distance,
factor graphs and iSAM2 in plain language.

---

## What's in here

| Package | What it is |
|---|---|
| **`localization_final`** | **The SLAM.** RANSAC line extraction → ICNN/χ² data association → GTSAM factor graph solved incrementally with iSAM2. |
| `stonefish_ros2` | *Third-party.* The ROS 2 bridge to the Stonefish simulator. |
| `turtlebot_simulation` | The Stonefish world: `.scn` scenario files (circuit + robot + sensors) and the bringup launch. |
| `kobuki_description` | Meshes/URDF for the drive base. |
| `swiftpro_description` | Meshes/URDF for the arm mounted on the robot. |
| `swiftpro` | `ros2_control` driver for the **real** arm. Unused in simulation. |
| `tutorial_interfaces` | Small custom message definitions. |

---

## 1. Prerequisites

- **Ubuntu 24.04** with **ROS 2 Jazzy** ([install guide](https://docs.ros.org/en/jazzy/Installation.html))
- **A GPU with OpenGL 4.3+.** Stonefish renders on the GPU and says so plainly in
  its own docs. There is a headless fallback — see the note at the bottom.

---

## 2. Install Stonefish (the C++ library)

Stonefish is a plain C++ library, **not** a ROS package. `stonefish_ros2`'s
`CMakeLists.txt` does `find_package(Stonefish REQUIRED 1.6.0)`, so the library
**must already be installed system-wide before you build this workspace.**

Follow the official guide — reproduced here for convenience:
**→ https://stonefish.readthedocs.io/en/latest/install.html**

### 2a. Dependencies

```bash
sudo apt install -y libglm-dev libsdl2-dev libfreetype6-dev cmake build-essential git
```

- `libglm-dev` — OpenGL Mathematics (version ≥ 0.9.9.0)
- `libsdl2-dev` — window + input handling
- `libfreetype6-dev` — font rendering for the on-screen GUI

### 2b. The SDL2 CMake fix

SDL2 ships a CMake config file with a stray space after `-lSDL2`, which breaks
linking. Stonefish's own docs call this out. Remove the space:

```bash
sudo sed -i 's/-lSDL2 /-lSDL2/g' /usr/lib/x86_64-linux-gnu/cmake/SDL2/sdl2-config.cmake
```

### 2c. Build and install

```bash
git clone https://github.com/patrykcieslak/stonefish.git
cd stonefish
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

Do **not** add `-DCMAKE_BUILD_TYPE=Release` or any other flags. Build it exactly
as the docs say.

---

## 3. Install GTSAM (the factor-graph solver)

GTSAM is the backend that actually solves the SLAM problem — iSAM2, custom
factors, and the covariance ellipses you'll see in RViz. It is **pip-only**;
`rosdep` cannot resolve it.

```bash
pip3 install --break-system-packages "gtsam" "numpy<2"
```

Two things worth knowing:

- **`--break-system-packages`** is needed because Ubuntu 24.04 marks its system
  Python as externally managed. This is expected, not a hack.
- **`numpy<2` is deliberate.** ROS's `cv_bridge` ships a compiled extension built
  against NumPy 1.x's ABI. If pip resolves a NumPy 2.x for you, `import cv_bridge`
  starts failing at runtime with a confusing, unrelated-looking error. Pinning it
  in the same command stops the resolver from silently upgrading it. GTSAM's
  iSAM2 and `Marginals` both work fine under NumPy 1.x.

---

## 4. Install the remaining ROS dependencies

```bash
sudo apt install -y \
  ros-jazzy-pcl-conversions ros-jazzy-image-transport libpcl-dev \
  ros-jazzy-irobot-create-msgs \
  ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
  ros-jazzy-xacro ros-jazzy-joint-state-publisher ros-jazzy-robot-state-publisher \
  ros-jazzy-realsense2-description \
  python3-colcon-common-extensions
```

Why each one is there — these are not obvious, and each caused a real failure:

- `pcl-conversions`, `image-transport`, `libpcl-dev` → required to **build**
  `stonefish_ros2`.
- `irobot-create-msgs` → `graph_slam.py` imports `WheelVels` and subscribes to
  `/wheel_vels` **unconditionally**, even in simulation. Without it the node
  won't even import.
- `ros2-control`, `ros2-controllers` → required to **build** the `swiftpro`
  package. The simulation never uses it (the arm is driven by
  `turtlebot_simulation/scripts/swiftpro_controller.py`), but it's built so the
  real-robot path stays intact. If you want a smaller install, dropping a
  `COLCON_IGNORE` file into `src/swiftpro/` is safe.
- `realsense2-description` → `kobuki.urdf.xacro` does
  `<xacro:include filename="$(find realsense2_description)/urdf/_d435i.urdf.xacro"/>`.
  That's resolved by **xacro at launch time**, not by colcon at build time, and
  upstream declared it in no `package.xml` — so the workspace builds perfectly
  and then dies at `ros2 launch` with `PackageNotFoundError`.

---

## 5. Build the workspace

```bash
git clone https://github.com/IAMHAADICOOL/omokai_part3_slam.git slam
cd slam
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## 6. Run it (2 terminals)

```bash
# Terminal A — Stonefish + RViz + the whole SLAM pipeline
ros2 launch localization_final graph_slam_updated.launch.py

# Terminal B — drive the robot so it can see walls and build the map
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args -r __ns:=/turtlebot
```

`-r __ns:=/turtlebot` **remaps the node's namespace** so teleop publishes to
`/turtlebot/cmd_vel` — where the simulated robot is listening — rather than the
bare `/cmd_vel`.

On a hybrid-graphics (Optimus) laptop, force the discrete GPU:
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  ros2 launch localization_final graph_slam_updated.launch.py
```

### What to watch

Drive slowly around the circuit with Terminal B. In RViz:

| You do | What happens |
|---|---|
| Drive along a wall | raw scan points collapse into a coloured **line segment** |
| Drive past the same wall again | the line is **recognised**, not duplicated (data association) |
| Complete a full lap | `/odom_graphslam` stays on the circuit while raw odometry visibly drifts off |
| Keep watching a landmark | its **uncertainty ellipse shrinks** the more times it's seen |

The launch starts four things: the Stonefish simulator, `arm_tuck` (folds the
robot's arm out of the LiDAR's scan plane — otherwise the robot maps its own arm
as a wall), the line extractor, and the Graph-SLAM node.

---

## 7. No GPU?

Stonefish also builds a headless binary, `stonefish_simulator_nogpu`. The LiDAR
this SLAM depends on is a **CPU raycast** (`multibeam`) — only the unused
RealSense cameras are GPU-rendered — so **the SLAM still runs**. You lose the 3D
simulator window, but RViz still shows the scan, the extracted lines, the
landmarks and the estimated path, which is everything the demo is about.

Point `turtlebot_simulation/launch/turtlebot_basic.launch.py` at
`stonefish_simulator_nogpu` instead of `stonefish_simulator`.

---

## 8. Attribution

`localization_final` and its supporting packages originate from a university
**Hands-On Localization** course project (report preserved as
`HOL_Final_Report__Final_version_.pdf`), authored by a team of three, and are
included here as prior work — extended and integrated. Original package READMEs
are preserved as `UPSTREAM_README.md` next to each new one.

| Component | Author / URL | License |
|---|---|---|
| Stonefish simulator | [patrykcieslak/stonefish](https://github.com/patrykcieslak/stonefish) | see repo |
| `stonefish_ros2` | [patrykcieslak/stonefish_ros2](https://github.com/patrykcieslak/stonefish_ros2) | see repo |
| GTSAM | [borglab/gtsam](https://github.com/borglab/gtsam) | BSD-3-Clause |
| `realsense2_description` | [realsense-ros](https://github.com/IntelRealSense/realsense-ros) | Apache-2.0 |
| `localization_final` + supporting packages | university course project (team of three) | MIT (per `package.xml`) |

Repo-side fixes applied on import: removed a hardcoded `/home/<user>/ros2_ws`
path from `localization_final`'s README; corrected `rclcpp_lifcycle` →
`rclcpp_lifecycle` in `swiftpro/package.xml`; declared the previously-undeclared
`realsense2_description` dependency in `kobuki_description/package.xml`.
