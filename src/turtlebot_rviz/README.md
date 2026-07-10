# turtlebot_rviz — a namespace-aware RViz launcher (Part 3)

A small `ament_python` package holding exactly one thing: a launch file that
starts RViz2 with its TF topics **auto-remapped** if the robot is running under
a namespace. No saved `.rviz` configs live here — those are in
`turtlebot_description/rviz/` and `localization_final/rviz/`.

## What `launch/rviz_launch.py` actually does

```python
def check_topics(context, *args, **kwargs):
    rclpy.init(args=None)
    node = RclpyNode("turtlebot_rviz_checker")
    existing_topics = [t[0] for t in node.get_topic_names_and_types()]
    node.destroy_node()
    rclpy.shutdown()

    remappings = []
    if '/turtlebot/tf' in existing_topics:
        remappings.append(('/tf', '/turtlebot/tf'))
    if '/turtlebot/tf_static' in existing_topics:
        remappings.append(('/tf_static', '/turtlebot/tf_static'))

    return [Node(package='rviz2', executable='rviz2', name='rviz2',
                  remappings=remappings)]
```

This runs as an `OpaqueFunction` — a launch action that executes a plain Python
function at launch time, rather than declaring everything statically up front.
That's the mechanism that makes the trick work: it spins up a throwaway rclpy
node **before** RViz starts, asks the ROS graph "does `/turtlebot/tf` exist right
now?", and only *then* decides whether RViz needs a `/tf` → `/turtlebot/tf`
remapping.

The reason that matters: TurtleBot's TF is published under a `/turtlebot/...`
namespace in this project (see `localization_final/README.md`'s note on
`publish_tf`), not on the bare `/tf`. RViz subscribes to `/tf` by default. Without
a remap, RViz would sit there with no robot model and no frames, because it's
listening on a topic nothing publishes to. Hardcoding that remap would work for
this one robot, but break the moment someone runs this launch file against a
robot published without a namespace (e.g. the real-robot path, or a different
exercise from the same course). Checking at launch time instead of hardcoding
means the one launch file works either way.

## Where Part 3 actually gets its RViz window

Not from here. `localization_final/launch/graph_slam_updated.launch.py` starts
its own `rviz2` node directly, pointed at
`localization_final/rviz/task2_graphslam_minimal.rviz` (the config that shows
`/slam_landmarks` and the SLAM pose estimate). This package's namespace-probing
launcher isn't part of that chain currently — it's a standalone utility you can
run by hand (`ros2 launch turtlebot_rviz rviz_launch.py`) against whatever robot
happens to be running, real or simulated, without needing to know its namespace
in advance.

## Depends on

`rclpy`, `launch`.
