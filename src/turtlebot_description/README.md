# turtlebot_description — the robot's URDF and meshes (Part 3)

Describes the Part 3 robot (a Kobuki base + a SwiftPro arm) as a **URDF/xacro**
model, plus several saved RViz configs.

## The file that actually matters: `urdf/turtlebot.urdf.xacro`

This is the top-level composition — it doesn't define any geometry itself, it
**welds two other packages' robots together**:

```xml
<xacro:include filename="$(find kobuki_description)/urdf/kobuki.urdf.xacro"/>
<xacro:kobuki prefix="$(arg mobile_base_namespace)"/>

<xacro:include filename="$(find swiftpro_description)/urdf/swiftpro.urdf.xacro"/>
<xacro:swiftpro prefix="$(arg manipulator_namespace)"/>

<joint name="base_to_manipulator" type="fixed">
  <parent link="$(arg mobile_base_namespace)/base_link"/>
  <child link="$(arg manipulator_namespace)/manipulator_base_link"/>
  <origin xyz="${0.0507-0.014} 0.0 0.0" rpy="0.0 0.0 1.5708"/>
</joint>
```

`kobuki_description`'s macro gives the wheeled base; `swiftpro_description`'s
macro gives the arm; the `<joint>` at the end bolts the arm onto the base at a
fixed offset. Neither included package needs to be built or written with the
other in mind — this file is the only place they're actually joined.

**This is the file that crashes at launch if `realsense2_description` isn't
installed** (see `kobuki_description/README.md`): `kobuki.urdf.xacro`, included
here, itself includes a RealSense camera macro that xacro only resolves at
*launch* time, not build time.

## Who loads it

Not `turtlebot_description` itself — `CMakeLists.txt` only installs the file,
it doesn't launch anything with it. The actual consumer is
`turtlebot_simulation/launch/turtlebot_basic.launch.py`, which finds this
package's share directory, points `robot_state_publisher` at
`urdf/turtlebot.urdf.xacro`, and runs it through `xacro` to produce the
`robot_description` parameter that both `robot_state_publisher` and RViz use to
draw the robot and populate the TF tree.

## Two things in this package that are *not* wired into Part 3's sim launch

Worth being upfront about, since finding them unused can otherwise look like a
missing piece:

- **`launch/turtlebot_description.launch.py`** declares three arguments
  (`robot_name`, `base_name`, `arm_name`) and does nothing with them — no node,
  no include. It's a stub, not currently used by either of
  `localization_final`'s launch files.
- **`rviz/*.rviz`** — four saved configs (`turtlebot.rviz`,
  `turtlebot_planning.rviz`, `turtlebot_rtabmap.rviz`, `nav_rtabmap_real.rviz`).
  The RTAB-Map-named ones point at a different SLAM approach from the course
  this package originates from, not the Graph SLAM in `localization_final`.
  Part 3's actual launch (`graph_slam_updated.launch.py`) uses its own config,
  `localization_final/rviz/task2_graphslam_minimal.rviz`, not any of these.

None of that stops the package from doing its one load-bearing job — supplying
`turtlebot.urdf.xacro` to `turtlebot_simulation`'s bringup.

## Depends on

`kobuki_description`, `swiftpro_description` (both `find_package`'d directly in
`CMakeLists.txt`), `urdf`, `xacro`, `robot_state_publisher`,
`joint_state_publisher`.

**Used by:** `turtlebot_simulation` (bringup).
