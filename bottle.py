#!/usr/bin/env python3
"""Problem 4: compute a lid approach point and move along a timed trajectory."""
import copy
import math
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped
from hw_arm_bottle.arm import SimpleArm
from hw_arm_bottle.scene import BOTTLE_HEIGHT, LID_CLEARANCE, TRAJECTORY_DURATION
from trajectory import sample_trajectory


class BottleArm(SimpleArm):
    def __init__(self):
        # This separate subclass keeps Problem 3's timer exercise independent.
        super().__init__(0.05, '/robot_arm', True)
        self.q_start = list(self.q)
        self.q_goal = None
        self.motion_started = None
        self.target_pub = self.create_publisher(PoseStamped, '/lid_target', 10)
        self.create_subscription(PoseStamped, '/bottle_pose', self.bottle_callback, 10)

    @staticmethod
    def compute_lid_target(bottle_pose):
        """Return a base_link PoseStamped above the upright bottle's lid.

        Input is the physical bottle BASE pose, not the OBJ marker origin.
        Preserve its header; do not modify the input message.
        """
        # Return None for unsupported frames; the provided callback holds still.
        if bottle_pose.header.frame_id != 'base_link':
            return None

        lid_pose = copy.deepcopy(bottle_pose)

        lid_pose.pose.position.z = bottle_pose.pose.position.z + BOTTLE_HEIGHT + LID_CLEARANCE

        return lid_pose

    @staticmethod
    def generate_trajectory(q_start, q_goal, duration, elapsed):
        """Return three joint positions at elapsed seconds; no sleeps or loops.

        Clamp outside [0, duration]. Use an interpolation with zero joint
        velocity and acceleration at both ends, not an immediate goal jump.
        """
        return sample_trajectory(q_start, q_goal, duration, elapsed)

    def bottle_callback(self, msg):
        if msg.header.frame_id != 'base_link':
            self.get_logger().warning('bottle base pose must be in base_link')
            return
        target = self.compute_lid_target(msg)
        if target is None:
            self.get_logger().warning('lid target unavailable for this frame',
                                      throttle_duration_sec=2.0)
            return
        self.target_pub.publish(target)
        self.target_callback(target)

    def target_callback(self, msg):
        if msg.header.frame_id != 'base_link':
            self.get_logger().warning('target pose must be in base_link')
            return
        p = msg.pose.position
        if not all(math.isfinite(value) for value in (p.x, p.y, p.z)):
            self.get_logger().warning('Target contains non-finite coordinates; current motion unchanged')
            return
        goal = self.ik(p.x, p.y, p.z)
        if goal is None:
            self.get_logger().warning('Target outside workspace; current motion unchanged')
            return
        # Repeated detector messages must not reset the clock on the same goal.
        if self.q_goal is not None and all(abs(a-b) < 1e-8 for a, b in zip(goal, self.q_goal)):
            return
        self.q_start = list(self.q)
        self.q_goal = list(goal)
        self.motion_started = self.get_clock().now()
        self.get_logger().info(
            f'target accepted ({p.x:.2f}, {p.y:.2f}, {p.z:.2f}); '
            f'trajectory duration {TRAJECTORY_DURATION:.1f} s')

    def update(self):
        if self.motion_started is not None:
            elapsed = (self.get_clock().now() - self.motion_started).nanoseconds * 1e-9
            self.q = self.generate_trajectory(
                self.q_start, self.q_goal, TRAJECTORY_DURATION, elapsed)
            if elapsed >= TRAJECTORY_DURATION:
                self.motion_started = None
                self.get_logger().info('trajectory complete; holding lid target')
        super().update()


def main():
    rclpy.init()
    node = BottleArm()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
