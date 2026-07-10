# kobuki_description — meshes and URDF for the Kobuki mobile base (Part 3)

The Part 3 robot's **drive base**. Supplies the physical geometry that both
Stonefish and RViz need.

**Why this package must exist even though nothing imports it in Python:** the
Stonefish scenario files reference its meshes directly, e.g.

```xml
<visual filename="$(find kobuki_description)/resources/meshes/main_body.obj"/>
```

so the package must be built and installed, or the simulator fails to load the
world. Its `CMakeLists.txt` installs `resources/`, `urdf/`, `rviz/`, `launch/`.

**Contains:** collision (`*_phy.obj`) and visual (`.obj`/`.dae`) meshes for the
body, plates, wheels, RPLidar and RealSense; `urdf/kobuki.urdf.xacro`.

## External dependency that isn't obvious

`urdf/kobuki.urdf.xacro` contains:

```xml
<xacro:include filename="$(find realsense2_description)/urdf/_d435i.urdf.xacro"/>
```

That `$(find …)` is resolved by **xacro at launch time**, not by `colcon` at build
time — and upstream declared it in no `package.xml`. The result is a package that
compiles perfectly and then dies at `ros2 launch` with
`PackageNotFoundError: package 'realsense2_description' not found`.

Fixed here two ways: `ros-jazzy-realsense2-description` is installed by the
Dockerfile and `install_native.sh`, and the dependency is now **declared** as an
`<exec_depend>` in this package's `package.xml` so `rosdep` can see it too.

**Used by:** `turtlebot_simulation` scenarios; `turtlebot_description`.
