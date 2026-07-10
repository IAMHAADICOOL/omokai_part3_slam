#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class ArmTuck(Node):
    def __init__(self):
        super().__init__('arm_tuck')

        self.declare_parameter('joint_state_topic', '/turtlebot/joint_states')
        self.declare_parameter('command_topic', '/turtlebot/swiftpro/joint_velocity_controller/command')
        self.declare_parameter('target_joint1', 0.00)
        self.declare_parameter('target_joint2', 0.00)
        self.declare_parameter('target_joint3', -1.40)
        self.declare_parameter('target_joint4', 0.00)
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('v_max_per_joint', [0.6, 0.6, 0.6, 0.6])
        self.declare_parameter('deadband', 0.05)
        self.declare_parameter('hold_time', 0.5)
        self.declare_parameter('control_rate', 20.0)

        self.joint_state_topic = self.get_parameter('joint_state_topic').value
        self.command_topic = self.get_parameter('command_topic').value
        
        self.targets = [
            self.get_parameter('target_joint1').value,
            self.get_parameter('target_joint2').value,
            self.get_parameter('target_joint3').value,
            self.get_parameter('target_joint4').value
        ]
        
        self.kp = self.get_parameter('kp').value
        self.v_max = self.get_parameter('v_max_per_joint').value
        self.deadband = self.get_parameter('deadband').value
        self.hold_time = self.get_parameter('hold_time').value
        self.control_rate = self.get_parameter('control_rate').value

        self.joint_names = [
            'turtlebot/swiftpro/joint1',
            'turtlebot/swiftpro/joint2',
            'turtlebot/swiftpro/joint3',
            'turtlebot/swiftpro/joint4'
        ]

        self.current_positions = {name: None for name in self.joint_names}

        self.sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self.joint_state_cb,
            10
        )

        self.pub = self.create_publisher(
            Float64MultiArray,
            self.command_topic,
            10
        )

        self.timer = self.create_timer(1.0 / self.control_rate, self.timer_cb)

        self.in_deadband_ticks = 0
        self.holding = False
        self.missing_warn_ticks = 0

        self.get_logger().info('ArmTuck node started, waiting for joint states to tuck arm.')

    def joint_state_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                self.current_positions[name] = msg.position[i]

    def timer_cb(self):
        missing = [name for name in self.joint_names if self.current_positions[name] is None]
        if missing:
            if self.missing_warn_ticks % int(self.control_rate) == 0:
                self.get_logger().warn(f'Missing joint states for: {missing}')
            self.missing_warn_ticks += 1
            self.publish_zeros()
            return
        
        self.missing_warn_ticks = 0

        errors = []
        for i, name in enumerate(self.joint_names):
            current = self.current_positions[name]
            target = self.targets[i]
            errors.append(target - current)

        if self.holding:
            if any(abs(e) > 2 * self.deadband for e in errors):
                self.holding = False
                self.in_deadband_ticks = 0
                self.get_logger().info('Joint drifted, resuming P-control.')
            else:
                self.publish_zeros()
                return
        
        all_in_deadband = True
        commands = []
        for i, err in enumerate(errors):
            if abs(err) < self.deadband:
                v = 0.0
            else:
                all_in_deadband = False
                v = self.kp * err
                v = max(-self.v_max[i], min(self.v_max[i], v))
            commands.append(v)

        if all_in_deadband:
            self.in_deadband_ticks += 1
            hold_ticks = int(self.hold_time * self.control_rate)
            if self.in_deadband_ticks >= hold_ticks:
                self.holding = True
                self.get_logger().info('arm_tuck: tucked pose reached')
                self.publish_zeros()
                return
        else:
            self.in_deadband_ticks = 0

        self.publish_commands(commands)

    def publish_commands(self, velocities):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in velocities]
        self.pub.publish(msg)

    def publish_zeros(self):
        self.publish_commands([0.0, 0.0, 0.0, 0.0])

    def destroy_node(self):
        self.publish_zeros()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArmTuck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
