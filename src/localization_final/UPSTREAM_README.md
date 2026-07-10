# localization_final
**ROS 2 Package — Graph SLAM with Polar Line Features | TurtleBot + Stonefish / Real Robot**

---

## 1  Overview

This ROS 2 package implements a complete **Graph SLAM** pipeline for a differential-drive robot operating in a structured indoor environment. Line features are extracted from a 2-D LiDAR scan using **RANSAC** and represented in polar form (ρ, α). These features are matched to a growing map via an **Individual Compatibility Nearest-Neighbour (ICNN)** gate based on Mahalanobis distance, and the full pose-and-landmark graph is optimised incrementally using **iSAM2** (Incremental Smoothing and Mapping 2) from the GTSAM library.

The architecture separates concerns cleanly across three layers:

| Layer | Responsibility |
|---|---|
| **Perception** | RANSAC polar line extraction from `LaserScan` |
| **Data Association** | ICNN Mahalanobis gate + geometric duplicate recovery |
| **Estimation** | iSAM2 factor graph — odometry, IMU heading prior, polar line landmark factors |

Odometry is propagated through a unicycle kinematic model driven by wheel encoder velocities (or wheel angular rates on the real robot). IMU heading enters as a `PoseRotationPrior2D` factor. Loop closures are detected automatically whenever a previously observed landmark is re-associated, and iSAM2 applies a nonlinear batch correction every `key_frame_update` sensor-fused nodes.

The package supports both **Stonefish simulation** (NED IMU, simulated LiDAR) and the **real TurtleBot** (Create3 IMU, RPLidar) via separate launch files with distinct parameter tuning.

---

## 2  Requirements

- ROS 2 **Jazzy** on Ubuntu 24.04 (Humble compatible with minor substitutions)
- Python 3.10+ | `numpy`, `scipy`
- **GTSAM** Python bindings (`pip install gtsam` or build from source)
- `irobot_create_msgs` — for `WheelVels` message type (real robot)
- `turtlebot_simulation` package — Stonefish scenario and bringup (simulation only)
- Stonefish simulator with the `turtlebot_hoi_circuit1.scn` scenario file

---

## 3  How to Run

### 3.1  Simulation


```bash
source install/setup.bash
```

**Terminal 1** — launch the full Graph SLAM stack:

```bash
ros2 launch localization_final graph_slam_updated.launch.py
```
### 3.2  Real Robot

**Step 1** — confirm hardware topics are live:

```bash
ros2 topic hz /scan
ros2 topic hz /imu
```

**Step 2** — launch the full SLAM stack with real-robot parameters:

```bash
ros2 launch localization_final graph_slam_real_robot.launch.py
```

The real-robot launch automatically:
- Broadcasts an identity `map → odom` static transform as a prior (overridden by the SLAM node once landmarks are observed)
- Tucks the arm to the stored pose via the `arm_tuck` node
- Opens an RViz2 window with `map` as the fixed frame


---

## 4  Code Structure

| File / Folder | Purpose |
|---|---|
| `localization_final/graph_slam.py` | Main SLAM node (`GraphSlam`). Wheel odometry prediction, IMU heading prior, iSAM2 factor graph construction and incremental optimisation, landmark promotion, map publication. ~2 160 lines. |
| `localization_final/line_extractor.py` | RANSAC polar line extractor (`RansacLineExtractor`) and its ROS wrapper (`LineExtractionNode`). Extracts line segments from `LaserScan`, computes polar coordinates (ρ, α) and per-line covariance via Jacobian propagation. |
| `localization_final/data_association.py` | ICNN data association (`DataAssociation`). Computes predicted observations `hfj`, full innovation covariance `S`, squared Mahalanobis distances, individual-compatibility gate, and geometry-based duplicate recovery. |
| `localization_final/utils.py` | Quaternion ↔ Euler conversions, NED → ENU quaternion rotation, and `warp_angle` normalisation to (−π, π]. |
| `localization_final/arm_tuck.py` | Proportional arm controller (`ArmTuck`). Drives the uArm Swift Pro to a tucked joint configuration at startup; holds position once within deadband. |
| `tools/ransac_param_tester.py` | Offline RANSAC parameter sweep. Generates a synthetic 4-wall room point cloud with configurable noise and benchmarks line extraction quality across `(max_iterations, max_line_distance)` combinations. |
| `launch/graph_slam_updated.launch.py` | Simulation launch. Starts Stonefish bringup, `arm_tuck`, `graph_slam_node`, and `polar_line_extractor_node` with simulator-tuned parameters. |
| `launch/graph_slam_real_robot.launch.py` | Real-robot launch. Starts a `map → odom` identity static TF prior, `arm_tuck`, `graph_slam_node`, `polar_line_extractor_node`, and RViz2 with an inline-generated config. |
| `config/task2_params.yaml` | YAML parameter file for the EKF localisation node (prior task). Defines process noise Q and initial covariance P₀ for dead-reckoning and EKF variants. |
| `rviz/task2_graphslam_minimal.rviz` | Minimal RViz2 configuration for simulation: displays SLAM odometry, laser scan, landmark line segments, and uncertainty envelopes. |
| `rviz/task2_ekf.rviz` | RViz2 configuration for the EKF localisation variant. |
| `setup.py` | Package entry points. Declares executables `task2_data_association_localization`, `task2_polar_line_extractor`, and `arm_tuck`. |

---

## 5  SLAM Pipeline

The following stages execute on every wheel-encoder callback (the primary tick):

### Stage 1 — Unicycle Prediction

The robot state `x = [x, y, θ]ᵀ` is propagated forward via the unicycle kinematic model:

```
x̄ₖ = f(xₖ₋₁, uₖ, dt)
P̄ₖ = Jfₓ Pₖ₋₁ Jfₓᵀ + Jf_w Q Jf_wᵀ
```

where `Q = diag(σ²_L, σ²_R)` is the wheel velocity noise covariance and the Jacobians `Jfₓ`, `Jf_w` are computed analytically. The relative displacement `Δx_rel` and its covariance are accumulated separately for the iSAM2 `BetweenFactorPose2`.

### Stage 2 — RANSAC Line Extraction

The `LineExtractionNode` processes each `LaserScan` independently:

1. Convert polar LiDAR ranges to Cartesian points (with NED → ENU flip in simulation).
2. Iterative RANSAC: sample 2 points, fit a line, count inliers within `max_line_distance`.
3. Refit surviving inliers via SVD to obtain a minimum-variance line estimate.
4. Gap filter: split the inlier set at gaps > `max_gap_distance`; retain the longest contiguous segment.
5. Reject segments shorter than `min_line_length` or with inlier ratio below `min_inlier_ratio`.
6. Merge duplicate segments on the same underlying line using a polar similarity check.

Each accepted segment is converted to polar form (ρ, α) and its 2 × 2 covariance is computed by propagating endpoint uncertainty through the polar parameterisation Jacobian.

### Stage 3 — IMU Heading Prior

On the real robot, the Create3 IMU orientation quaternion is frozen; angular rate `ω_z` is integrated directly with a deadband at 0.01 rad/s. In simulation, the Stonefish NED quaternion is converted to ENU with a +π/2 heading correction. The resulting scalar heading enters the graph as a `PoseRotationPrior2D` factor.

### Stage 4 — Data Association (ICNN)

For each observed polar line `zᵢ = (ρᵢ, αᵢ)` and each map landmark `fⱼ = (ρⱼ, αⱼ)`:

1. Predict the expected observation: `ĥⱼ = h(x̄ₖ, fⱼ)`.
2. Compute the full innovation covariance:  
   `Sⱼ = Hₓ P̄ₖ Hₓᵀ + Hf Pf Hfᵀ`
3. Evaluate squared Mahalanobis distance:  
   `D²ᵢⱼ = (zᵢ − ĥⱼ)ᵀ Sⱼ⁻¹ (zᵢ − ĥⱼ)`
4. Retain pair `(i, j)` if `D²ᵢⱼ ≤ χ²(confidence_level, dof=2)` (individual compatibility).
5. Greedy assignment: sort compatible pairs by `D²`; assign each observation to the nearest unassigned landmark.
6. Geometry-based rescue: unmatched observations whose back-projected world feature falls within `(duplicate_rho_threshold, duplicate_alpha_threshold)` of an existing landmark are re-associated.

Unmatched observations that pass the geometry filter enter a **pending buffer**. A pending feature is promoted to the confirmed map only after `new_landmark_min_observations` independent sightings, preventing spurious landmarks from dynamic clutter.

### Stage 5 — Factor Graph Update (iSAM2)

On every tick a `BetweenFactorPose2` is added for the accumulated relative displacement. When the IMU flag is set, a `PoseRotationPrior2D` is added. When LiDAR lines are associated, a custom `CustomFactor` (polar line factor) is added per matched landmark, encoding the residual:

```
r = [ ρ_pred − ρ_obs,  wrap(α_pred − α_obs) ]ᵀ
```

with analytic Jacobians with respect to robot pose and landmark parameters (ρ_w, α_w).

Graph optimisation is triggered every `key_frame_update` sensor-fused nodes (once the graph has at least `num_min_key` poses). The first optimisation uses Levenberg–Marquardt for a global batch initialisation; subsequent updates use iSAM2 for O(loop-closure-size) incremental re-linearisation. After each optimisation, marginal covariances for all poses and landmarks are extracted and the visual segment geometry is re-aligned to the updated polar parameters.

---

## 6  Key Tuning Parameters

| Parameter | Default | Effect |
|---|---|---|
| `min_points_per_line` | 8 | Minimum RANSAC inlier count for a valid line |
| `max_line_distance` | 0.03 m | RANSAC inlier distance threshold |
| `min_line_length` | 0.20 m | Minimum extracted segment length |
| `duplicate_rho_threshold` | 0.15 m | Geometry duplicate gate in ρ |
| `duplicate_alpha_threshold` | 0.15 rad | Geometry duplicate gate in α |
| `confidence_level` | 0.97 | χ² gate confidence for ICNN |
| `new_landmark_min_observations` | 2 | Sightings required before landmark promotion |
| `key_frame_update` | 20 | Graph optimisation period (sensor-fused nodes) |
| `wheel_vel_noise_left/right` | 0.10 | Wheel velocity std-dev [rad/s] — Q matrix diagonal |
| `visualization_segment_merge_gap` | 0.40 m | Gap tolerance for merging collinear visual segments |

---

## 7  Visualization & Debugging

**RViz2 topics:**

| Topic | Type | Content |
|---|---|---|
| `/slam_landmarks` | `MarkerArray` | Confirmed landmark line segments coloured per landmark ID (golden-ratio hue cycle), floating `L(id)` text labels, and 95 % confidence uncertainty envelopes on each segment |
| `/odom_graphslam` | `Odometry` | SLAM pose estimate in the world frame |
| `/turtlebot/scan_points` | `MarkerArray` | Raw LiDAR points after Cartesian projection |
| `/turtlebot/line_segments` | `MarkerArray` | Live RANSAC-extracted segments in the robot frame |

**Console output** is prefixed by subsystem tag for easy filtering:

```
[GraphSLAM]  — node startup, TF configuration, initial pose insertion
[RANSAC]     — per-scan line count and polar coordinates
[DA]         — per-observation association result, Mahalanobis distance, loop-closure events
[OPT]        — post-optimisation landmark update (ρ, α, observation count)
[MAP]        — periodic map publication statistics
[GT]         — ground-truth ENU conversion (simulation only)
```

**Offline RANSAC tuning:** run the parameter sweep tool against a synthetic room to select `max_iterations` and `max_line_distance` before deploying on hardware:

```bash
python3 src/localization_final/tools/ransac_param_tester.py
```

The tool prints a table of extracted line count, mean inlier fit error, and wall-clock time per `(max_iterations, max_line_distance)` combination.
