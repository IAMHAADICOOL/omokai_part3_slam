# swiftpro_description — meshes and URDF for the uArm SwiftPro arm (Part 3)

Geometry for the small robotic arm mounted on top of the Part 3 robot.

The arm is not used for manipulation anywhere in this project — but it is not
scenery either. It sits **in the LiDAR's scan plane**, so `localization_final`'s
`arm_tuck` node folds it out of the way before driving. Without that, the SLAM
front-end sees the arm as a wall and maps it.

Like `kobuki_description`, this package exists mainly so the Stonefish scenario
can resolve `$(find swiftpro_description)/resources/meshes/link*.obj`. Building
it is mandatory; importing it is not a thing you do.

**Used by:** `turtlebot_simulation` scenarios (`turtlebot_featherstone.scn`).
