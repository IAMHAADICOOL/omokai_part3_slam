# localization_final — Part 3 (Graph SLAM with line features)

This package is **Part 3**. The robot drives around a circuit it has never seen,
and while driving it does two things at once:

- **Mapping** — turns raw LiDAR points into straight **wall segments**, and
  remembers where each one is.
- **Localization** — uses those remembered walls to correct its own position,
  because wheel odometry alone drifts badly.

Doing both at the same time is **SLAM** (Simultaneous Localization And Mapping).
The "graph" part is *how* it's solved: every robot pose and every wall becomes a
node in a graph, every measurement becomes an edge, and a solver finds the
arrangement of nodes that best explains all the edges at once.

The simulator here is **Stonefish**, not Gazebo — this is the only part of the
project that uses it.

> The original course README for this package is preserved as
> `UPSTREAM_README.md`; see the Attribution section at the bottom.

## Why lines instead of a grid map?

Most introductory SLAM uses an occupancy grid: a big array of "is this square
occupied?". Simple, but heavy, and it gives the optimizer nothing crisp to hold
on to.

This uses **line features**. A corridor wall is *one line*, not four hundred grid
cells. Each line is stored in **polar form** as just two numbers:

- **ρ (rho)** — the perpendicular distance from the origin to the line
- **α (alpha)** — the angle of that perpendicular

So a whole wall is `(ρ, α)`. Compact, and — crucially — a thing you can attach
error bars to and optimize.

One subtlety the code handles explicitly (`normalize_polar_line`): `(ρ, α)` and
`(−ρ, α+π)` describe **exactly the same physical line**. Left alone, the
optimizer would treat them as two different landmarks. So every line is forced
into a canonical form with `ρ ≥ 0` before it is ever stored or compared.

## The pipeline, end to end

```
LiDAR scan (/turtlebot/scan)
   │
   ▼
RANSAC line extraction ─────────► line segments, each as (ρ, α) + covariance
   │  line_extractor.py
   ▼
Data association ───────────────► "is this wall one I've seen before, or new?"
   │  data_association.py           ICNN + chi-squared gate
   ▼
Factor graph ───────────────────► poses + landmarks as nodes;
   │  graph_slam.py                 odometry + line sightings as edges
   ▼
iSAM2 incremental optimization ─► corrected robot path + corrected map
   │  (GTSAM)
   ▼
/odom_graphslam   +   /slam_landmarks  (RViz markers)
```

## Step 1 — RANSAC line extraction (`line_extractor.py`)

RANSAC = *RANdom SAmple Consensus*. In one sentence: **guess, count, keep the
best guess.**

`_ransac_best_line` repeatedly picks two random scan points, draws the line
through them, and counts how many other points lie within `max_line_distance` of
it (the "inliers"). After `max_iterations` guesses, the line with the most
inliers wins. This is robust to outliers in a way plain least-squares fitting is
not — one stray point cannot drag the whole fit.

The winning line is then cleaned up:

- **`_refit_line_svd`** — re-fits using *all* the inliers via SVD, not just the
  two random seed points that happened to find them. A far better line than the
  guess that discovered it.
- **`_gap_filter`** — one physical wall should not have a three-metre hole in it.
  Inliers separated by more than `max_gap_distance` along the line get split into
  separate segments, so a wall and a distant fence don't fuse into one imaginary
  line.
- **`merge_duplicate_segments`** — the opposite problem: a wall interrupted by a
  doorway comes back as two collinear pieces. Segments with similar `(ρ, α)` that
  nearly touch get merged.

**`calculate_covariance_matrix`** is what makes any of this usable by the
optimizer. It doesn't just say "here's a line" — it says "here's a line, **and
here's how sure I am** about ρ and about α." A line fitted from 8 points spread
over 20 cm is far less trustworthy than one from 200 points over 3 m, and the
graph must know that, or it will trust bad measurements as much as good ones.

## Step 2 — Data association (`data_association.py`)

The question that makes or breaks any SLAM system: **the robot sees a wall. Is it
a wall already on the map, or a brand-new one?**

Wrong in one direction, the map fills with duplicate walls. Wrong in the other,
the robot "closes a loop" that never existed and the map folds in on itself.

The method is **ICNN** (Individual Compatibility Nearest Neighbour):

1. **`hfj`** — predict what landmark *j* on the map *should* look like from the
   robot's current estimated pose. The measurement model: a world line
   `(ρ_w, α_w)` seen from pose `(x, y, θ)` appears as
   `ρ = ρ_w − x·cos α_w − y·sin α_w` and `α = α_w − θ`.
2. **`SquaredMahalanobisDistance`** — compare that prediction against the actual
   observation, **weighted by how uncertain each is**. Mahalanobis distance is
   ordinary distance divided by uncertainty: being 10 cm off is a big deal if you
   were confident to 1 cm, and irrelevant if you were only confident to a metre.
3. **`IndividualCompatibility`** — a **chi-squared test**. If the errors are
   Gaussian, that squared Mahalanobis distance follows a χ² distribution with 2
   degrees of freedom (one for ρ, one for α). Exceed the `confidence_level`
   threshold (default `0.97`) and the match is rejected as "too unlikely to be
   the same wall."

Two guards, both there because naive ICNN is fragile:

- **The pending buffer** (`_handle_unmatched_observation`,
  `_promote_pending_landmark`). A wall seen exactly once might be a sensor
  glitch. New candidates sit in a pending list and only become real landmarks
  after `new_landmark_min_observations` sightings. Noise never permanently
  pollutes the map.
- **Covariance floors** (`measurement_sigma_rho_floor`,
  `measurement_sigma_alpha_floor`). Fit a line from a very dense scan and the
  maths can report absurdly small uncertainty. The filter becomes
  **overconfident** and starts rejecting perfectly good matches for being "0.5 mm
  off." The floors say: never claim to be more certain than the sensor physically
  allows.

## Step 3 — The factor graph and iSAM2 (`graph_slam.py`)

A **factor graph** has two ingredients:

- **Variables** — the unknowns being solved for: every robot pose `x0, x1, x2…`,
  and every landmark's `ρ` and `α`.
- **Factors** — the constraints. Each says "these variables should satisfy this
  relationship, and here is how strongly I believe it."

The three kinds used here:

| Factor | What it says |
|---|---|
| `PriorFactorPose2` | "The robot started **here**." Pins the graph down — without it the entire map could slide anywhere and still be self-consistent. |
| `BetweenFactorPose2` | "Between pose *k* and *k+1*, wheel odometry says the robot moved by **this much**." |
| **custom polar-line factor** | "From pose *k*, landmark *j* was observed at `(ρ, α)`." |

That third one doesn't exist in GTSAM, so it's built with `gtsam.CustomFactor` in
`make_polar_line_factor`. It supplies both the **residual** (predicted minus
observed, angle properly wrapped) and the hand-derived **Jacobians** — the
partial derivatives telling the optimizer which way to nudge each variable to
reduce the error. The `flipped` branch in that function is the canonical-form
problem returning: if the prediction comes out with negative ρ it gets flipped
back, and the Jacobian signs **must** flip with it, or the optimizer confidently
walks the wrong direction.

**Why iSAM2 rather than plain least-squares?** Re-solving the whole graph from
scratch on every scan gets slower and slower as the map grows — by the end of a
lap you'd re-optimize thousands of poses just to incorporate one new wall.
**iSAM2** (incremental Smoothing And Mapping) updates only the part of the
solution that actually changed. The graph is re-optimized every
`key_frame_update` nodes (after a minimum of `num_min_key`), which keeps it real
time. `LevenbergMarquardtOptimizer` and `Marginals` cover the batch/covariance
paths — `Marginals` is what produces the uncertainty ellipses drawn around
landmarks in RViz.

## Sensor fusion, and the frame trap

- **Prediction** (`Prediction`, `f`, `Jfx`, `Jfw`) — a differential-drive
  (unicycle) motion model integrates left/right wheel velocities from
  `/turtlebot/joint_states` to propagate the pose **and its covariance** forward.
  Process noise comes from `wheel_vel_noise_left` / `wheel_vel_noise_right`.
- **IMU** (`recieve_imu`) — supplies heading, correcting the yaw drift that wheel
  odometry accumulates relentlessly.
- **NED vs ENU.** Stonefish publishes in **NED** (x-North, y-East, z-Down); ROS
  uses **ENU** (x-East, y-North, z-Up). `quaternion_ned_to_enu` in `utils.py`
  does the axis permutation, and the line extractor applies a π offset. Get this
  wrong and the map appears **mirrored** — a bug that looks like "SLAM is broken"
  when the geometry is actually fine.
- **`publish_tf` is deliberately `False`.** Stonefish's own `diff_drive_odometry`
  node already broadcasts the odom→base transform. If this node broadcast one
  too, two publishers would fight over the same TF edge and RViz would show the
  robot jittering between two poses.

## The nodes this package runs

| Executable | File | Role |
|---|---|---|
| `task2_data_association_localization` | `graph_slam.py` | the SLAM node: prediction, data association, factor graph, iSAM2 |
| `task2_polar_line_extractor` | `line_extractor.py` | standalone RANSAC line extraction + RViz visualization |
| `arm_tuck` | `arm_tuck.py` | folds the robot's SwiftPro arm out of the LiDAR's view before driving |

`arm_tuck` looks trivial but isn't optional: the arm sits **in the LiDAR's scan
plane**. Un-tucked, the robot sees its own arm as a wall and cheerfully maps it.

## Inputs and outputs

**Subscribes:** `/turtlebot/scan` (LaserScan) · `/turtlebot/joint_states` (wheel
encoders) · `/turtlebot/sensors/imu_data` (IMU) · `/turtlebot/odom_ground_truth`
(**evaluation only** — never fed to the estimator) · `/wheel_vels` (real robot).

**Publishes:** `/odom_graphslam` (corrected pose estimate) ·
`/slam_landmarks` (MarkerArray: the line map + uncertainty ellipses) ·
`/turtlebot/line_segments` · `/turtlebot/scan_points`.

## Key parameters (all settable on the launch command line)

| Parameter | Default | Meaning |
|---|---|---|
| `max_line_distance` | `0.03` | RANSAC inlier threshold (m) — how close a point must be to count as "on the line" |
| `min_points_per_line` | `8` | reject lines fitted from fewer points |
| `min_line_length` | `0.20` | reject segments shorter than this (noise) |
| `max_gap_distance` | `0.25` | split a segment if its inliers have a bigger hole than this |
| `max_iterations` | `120` | RANSAC guesses per line |
| `confidence_level` | `0.97` | χ² gate for accepting a data association |
| `new_landmark_min_observations` | `2` | sightings before a pending candidate becomes a real landmark |
| `key_frame_update` | `20` | re-run iSAM2 every N nodes |
| `num_min_key` | `10` | don't optimize until at least this many poses exist |
| `measurement_sigma_rho_floor` | `0.04` | minimum believable ρ uncertainty (anti-overconfidence) |
| `measurement_sigma_alpha_floor` | `0.06` | minimum believable α uncertainty |
| `wheel_vel_noise_left` / `_right` | `0.10` | process noise σ (rad/s) |

`tools/ransac_param_tester.py` tunes the RANSAC numbers against a recorded scan
without launching the whole stack.

## How to run it

See the **Part 3** section of the root `README.md` for the full command sequence
(Stonefish needs a GPU — there's a Docker override file for that).

## Attribution

This package originates from a university **Hands-On Localization** course
project (report preserved here as `HOL_Final_Report__Final_version_.pdf`),
authored by a team of three, and is included as prior work — extended and
integrated into this repository. The original package README is preserved as
`UPSTREAM_README.md`.

Note: `package.xml`'s `<description>` still describes the earlier **EKF** task
the project grew out of; the code in this folder is **Graph SLAM**. Full
attribution for this package, Stonefish, and GTSAM is in `docs/SOURCES.md`.
