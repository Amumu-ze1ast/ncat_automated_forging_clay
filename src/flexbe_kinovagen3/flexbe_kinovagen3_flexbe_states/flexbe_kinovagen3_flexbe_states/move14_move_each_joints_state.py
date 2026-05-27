#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

from flexbe_core import EventState, Logger
from flexbe_core.proxy import ProxyPublisher, ProxySubscriberCached


class Move14MoveEachJointsState(EventState):
    """
    FlexBE state to move Kinova Gen3 joints by relative amounts from current position.
    
    -- joint_1_delta    float   Amount to rotate joint 1 (radians, default: 0.0)
    -- joint_2_delta    float   Amount to rotate joint 2 (radians, default: 0.0)
    -- joint_3_delta    float   Amount to rotate joint 3 (radians, default: 0.0)
    -- joint_4_delta    float   Amount to rotate joint 4 (radians, default: 0.0)
    -- joint_5_delta    float   Amount to rotate joint 5 (radians, default: 0.0)
    -- joint_6_delta    float   Amount to rotate joint 6 (radians, default: 0.0)
    -- joint_7_delta    float   Amount to rotate joint 7 (radians, default: 0.0)
    -- duration         float   Time to complete motion (seconds, default: 5.0)
    -- tolerance        float   Position tolerance for completion (radians, default: 0.05)
    -- timeout          float   Maximum time to wait for motion (seconds, default: 15.0)
    
    <= done                     Motion completed successfully
    <= failed                   Motion failed or timeout
    
    Example usage:
        # Move joint 1 by +0.5 rad, joint 3 by -0.3 rad, others stay at current position
        Move14MoveEachJointsState(joint_1_delta=0.5, joint_3_delta=-0.3, duration=3.0)
    """

    def __init__(self, 
                 joint_1_delta=0.0, 
                 joint_2_delta=0.0, 
                 joint_3_delta=0.0, 
                 joint_4_delta=0.0, 
                 joint_5_delta=0.0, 
                 joint_6_delta=0.0, 
                 joint_7_delta=0.0,
                 duration=5.0,
                 tolerance=0.05,
                 timeout=15.0):
        
        super(Move14MoveEachJointsState, self).__init__(
            outcomes=['done', 'failed'],
            input_keys=[],
            output_keys=[]
        )

        # Store delta values for each joint
        self._deltas = [
            float(joint_1_delta),
            float(joint_2_delta),
            float(joint_3_delta),
            float(joint_4_delta),
            float(joint_5_delta),
            float(joint_6_delta),
            float(joint_7_delta)
        ]

        # Motion parameters
        self._duration = float(duration)
        self._tolerance = float(tolerance)
        self._timeout = float(timeout)

        # Joint names (must match your robot)
        self._joint_names = [
            'joint_1', 'joint_2', 'joint_3', 'joint_4',
            'joint_5', 'joint_6', 'joint_7'
        ]

        # Topics
        self._pub_topic = '/joint_trajectory_controller/joint_trajectory'
        self._sub_topic = '/joint_states'

        # Proxies
        self._pub = ProxyPublisher({self._pub_topic: JointTrajectory})
        self._sub = ProxySubscriberCached({self._sub_topic: JointState})

        # State variables
        self._start_time = None
        self._target_positions = None
        self._command_sent = False

    def execute(self, userdata):
        """Execute - called periodically while state is active."""
        
        if not self._command_sent:
            return 'failed'

        # Check timeout
        if self._start_time is not None:
            elapsed = (self._node.get_clock().now() - self._start_time).nanoseconds / 1e9
            if elapsed > self._timeout:
                Logger.logwarn(f'Motion timed out after {self._timeout}s')
                return 'failed'

        # Get current joint positions
        current_positions = self._get_current_positions()
        if current_positions is None:
            return None  # Keep waiting for joint states

        # Check if reached target
        errors = [abs(current - target) for current, target in zip(current_positions, self._target_positions)]
        max_error = max(errors)

        if max_error < self._tolerance:
            Logger.loginfo(f'✓ Target reached! Max error: {max_error:.4f} rad')
            return 'done'

        # Still moving
        return None

    def on_enter(self, userdata):
        """Called when state becomes active."""
        
        # Get current joint positions
        current_positions = self._get_current_positions()
        if current_positions is None:
            Logger.logerr('Failed to read current joint positions')
            self._command_sent = False
            return

        # Calculate target positions (current + delta)
        self._target_positions = [
            current + delta 
            for current, delta in zip(current_positions, self._deltas)
        ]

        # Log the motion plan
        Logger.loginfo('=== Relative Joint Motion ===')
        Logger.loginfo(f'Duration: {self._duration}s')
        for i in range(7):
            if abs(self._deltas[i]) > 0.001:  # Only log non-zero deltas
                Logger.loginfo(f'  Joint {i+1}: {current_positions[i]:+.3f} → {self._target_positions[i]:+.3f} (Δ{self._deltas[i]:+.3f})')

        # Create and send trajectory
        traj = JointTrajectory()
        traj.header.stamp = self._node.get_clock().now().to_msg()
        traj.joint_names = self._joint_names

        # Point 1: Current position at t=0
        pt1 = JointTrajectoryPoint()
        pt1.positions = current_positions
        pt1.time_from_start = Duration(sec=0, nanosec=0)

        # Point 2: Target position at t=duration
        pt2 = JointTrajectoryPoint()
        pt2.positions = self._target_positions
        pt2.time_from_start = Duration(
            sec=int(self._duration),
            nanosec=int((self._duration % 1.0) * 1e9)
        )

        traj.points = [pt1, pt2]

        # Publish trajectory
        try:
            self._pub.publish(self._pub_topic, traj)
            self._command_sent = True
            self._start_time = self._node.get_clock().now()
            Logger.loginfo('✓ Trajectory command sent')
        except Exception as e:
            Logger.logerr(f'Failed to publish trajectory: {str(e)}')
            self._command_sent = False

    def on_exit(self, userdata):
        """Called when state is exited."""
        self._command_sent = False
        self._start_time = None
        self._target_positions = None

    def _get_current_positions(self):
        """Get current joint positions from /joint_states."""
        try:
            if not self._sub.has_msg(self._sub_topic):
                return None

            msg = self._sub.get_last_msg(self._sub_topic)
            
            # Create mapping from joint name to position
            name_to_pos = dict(zip(msg.name, msg.position))

            # Extract positions in correct order
            positions = []
            for joint_name in self._joint_names:
                if joint_name not in name_to_pos:
                    Logger.logwarn(f'Joint {joint_name} not found in /joint_states')
                    return None
                positions.append(name_to_pos[joint_name])

            return positions

        except Exception as e:
            Logger.logwarn(f'Error reading joint states: {str(e)}')
            return None