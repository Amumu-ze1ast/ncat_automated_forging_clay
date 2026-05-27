#!/usr/bin/env python3

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS

from flexbe_core import EventState, Logger
from flexbe_core.proxy import ProxyPublisher, ProxySubscriberCached
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np


class Move1MoveToJointPositionState(EventState):
    """
    FlexBE state to move Kinova Gen3 to specified joint positions.
    
    Parameters:
    -- target_joints    list    List of 7 joint angles in radians [j1, j2, j3, j4, j5, j6, j7]
    -- duration         float   Time to reach target in seconds (default: 5.0)
    -- tolerance        float   Position tolerance to consider reached (default: 0.01 rad)
    
    Outcomes:
    <= done             Target position reached successfully
    <= failed           Failed to reach target or timeout
    
    Output Keys:
    #> final_error      Final error between desired and actual positions
    """

    def __init__(self, target_joints, duration=5.0, tolerance=0.01):
        super(Move1MoveToJointPositionState, self).__init__(
            outcomes=['done', 'failed'],
            output_keys=['final_error']
        )
        
        # Parameters
        self._target_joints = np.array(target_joints, dtype=float)
        self._duration = Duration(seconds=duration)
        self._tolerance = tolerance
        
        # Joint names for Kinova Gen3
        self._joint_names = [
            'joint_1', 'joint_2', 'joint_3',
            'joint_4', 'joint_5', 'joint_6', 'joint_7'
        ]
        
        # Topic names
        self._traj_topic = '/joint_trajectory_controller/joint_trajectory'
        self._joint_topic = '/joint_states'
        
        # Proxies
        self._pub = ProxyPublisher({self._traj_topic: JointTrajectory})
        self._sub = ProxySubscriberCached({self._joint_topic: JointState})
        
        # State management
        self._start_time = None
        self._timeout = Duration(seconds=duration + 10.0)
        self._return = None

    def execute(self, userdata):
        """
        Execute state - called periodically while state is active.
        """
        if self._return is not None:
            # Already finished, return cached result
            return self._return
        
        # Get current joint positions
        actual = self._get_actual_positions()
        
        if actual is None:
            Logger.logwarn('No joint states received yet')
            return None  # Keep executing
        
        # Check if target reached
        error = np.abs(actual - self._target_joints)
        
        if np.all(error < self._tolerance):
            Logger.loginfo('Target position reached!')
            userdata.final_error = error.tolist()
            self._return = 'done'
            return 'done'
        
        # Check timeout
        elapsed = Move1MoveToJointPositionState._node.get_clock().now() - self._start_time
        if elapsed >= self._timeout:
            Logger.logwarn(f'Timeout after {elapsed.nanoseconds / S_TO_NS:.1f}s. Max error: {np.max(error):.4f}')
            userdata.final_error = error.tolist()
            self._return = 'failed'
            return 'failed'
        
        # Still moving
        return None

    def on_enter(self, userdata):
        """
        Called when state becomes active - send trajectory command.
        """
        # Reset return value
        self._return = None
        
        # Send trajectory command
        self._send_trajectory()
        
        # Record start time using class node reference
        self._start_time = Move1MoveToJointPositionState._node.get_clock().now()
        
        Logger.loginfo(f'Moving to joint positions: {self._target_joints}')

    def on_exit(self, userdata):
        """
        Called when leaving the state.
        """
        Logger.loginfo('MoveToJointPosition state exiting')

    def on_start(self):
        """
        Called when behavior starts.
        """
        Logger.loginfo('MoveToJointPosition state started')

    def on_stop(self):
        """
        Called when behavior stops.
        """
        Logger.loginfo('MoveToJointPosition state stopped')

    # ==================== Helper Methods ====================

    def _send_trajectory(self):
        """Send joint trajectory command to robot."""
        traj = JointTrajectory()
        traj.joint_names = self._joint_names
        
        point = JointTrajectoryPoint()
        point.positions = self._target_joints.tolist()
        point.time_from_start.sec = int(self._duration.nanoseconds / S_TO_NS)
        point.time_from_start.nanosec = int(self._duration.nanoseconds % S_TO_NS)
        
        traj.points = [point]
        
        self._pub.publish(self._traj_topic, traj)
        Logger.loginfo('Joint trajectory command sent')

    def _get_actual_positions(self):
        """Get current joint positions as numpy array."""
        if not self._sub.has_msg(self._joint_topic):
            return None
        
        msg = self._sub.get_last_msg(self._joint_topic)
        
        name_to_pos = dict(zip(msg.name, msg.position))
        
        actual = []
        for joint_name in self._joint_names:
            if joint_name not in name_to_pos:
                return None
            actual.append(name_to_pos[joint_name])
        
        return np.array(actual, dtype=float)