# tutorial_interfaces — small custom message definitions (Part 3)

An `ament_cmake` interface-only package holding custom `.msg`/`.srv` definitions
used by the Part 3 stack. It builds no nodes; its only job is to generate the
message types other packages import.

Interface packages must be built before their consumers, which `colcon` handles
automatically from the dependency graph — there is nothing to run here.
