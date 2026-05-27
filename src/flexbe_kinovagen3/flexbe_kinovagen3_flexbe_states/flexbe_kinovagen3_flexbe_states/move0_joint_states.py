#!/usr/bin/env python3

from flexbe_core import EventState, Logger
from flexbe_core.proxy import ProxySubscriberCached
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np


class Move0ReadJointState(EventState):
    """
    FlexBE state for reading current joint positions from joint trajectory controller.
    
    Reads from /joint_trajectory_controller/state topic and outputs actual joint positions.
    
    Parameters:
    -- timeout          float   Timeout in seconds to wait for message (default: 5.0)
    
    Outcomes:
    <= done             Successfully read joint state
    <= failed           Failed to read joint state (timeout or error)
    
    Output Keys:
    #> joint_positions  List of 7 joint positions (radians)
    #> joint_1          Joint 1 position (radians)
    #> joint_2          Joint 2 position (radians)
    #> joint_3          Joint 3 position (radians)
    #> joint_4          Joint 4 position (radians)
    #> joint_5          Joint 5 position (radians)
    #> joint_6          Joint 6 position (radians)
    #> joint_7          Joint 7 position (radians)
    """

    def __init__(self, timeout=5.0):
        super(Move0ReadJointState, self).__init__(
            outcomes=['done', 'failed'],
            output_keys=['joint_positions', 'joint_1', 'joint_2', 'joint_3', 
                        'joint_4', 'joint_5', 'joint_6', 'joint_7']
        )
        
        # Parameters
        self._timeout = timeout
        
        # Topic name
        self._state_topic = '/joint_trajectory_controller/state'
        
        # Subscriber
        self._sub = ProxySubscriberCached({self._state_topic: JointTrajectoryControllerState})
        
        # State management
        self._start_time = None
        self._return = None

    def execute(self, userdata):
        """Execute state - called periodically while state is active."""
        if self._return is not None:
            return self._return
        
        # Check if message received
        if self._sub.has_msg(self._state_topic):
            msg = self._sub.get_last_msg(self._state_topic)
            
            # Extract actual positions
            actual_positions = list(msg.actual.positions)
            
            if len(actual_positions) >= 7:
                # Store as list
                userdata.joint_positions = actual_positions[:7]
                
                # Store individual joint values
                userdata.joint_1 = actual_positions[0]
                userdata.joint_2 = actual_positions[1]
                userdata.joint_3 = actual_positions[2]
                userdata.joint_4 = actual_positions[3]
                userdata.joint_5 = actual_positions[4]
                userdata.joint_6 = actual_positions[5]
                userdata.joint_7 = actual_positions[6]
                
                Logger.loginfo(f'Read joint positions: [{actual_positions[0]:.4f}, {actual_positions[1]:.4f}, {actual_positions[2]:.4f}, {actual_positions[3]:.4f}, {actual_positions[4]:.4f}, {actual_positions[5]:.4f}, {actual_positions[6]:.4f}]')
                
                self._return = 'done'
                return 'done'
            else:
                Logger.logerr(f'Expected 7 joints, got {len(actual_positions)}')
                self._return = 'failed'
                return 'failed'
        
        # Check timeout
        elapsed = Move0ReadJointState._node.get_clock().now() - self._start_time
        elapsed_sec = elapsed.nanoseconds / 1e9
        
        if elapsed_sec >= self._timeout:
            Logger.logwarn(f'Timeout after {elapsed_sec:.1f}s - no message received')
            self._return = 'failed'
            return 'failed'
        
        return None

    def on_enter(self, userdata):
        """Called when state becomes active."""
        self._return = None
        self._start_time = Move0ReadJointState._node.get_clock().now()
        
        Logger.loginfo('Reading joint state from /joint_trajectory_controller/state...')

    def on_exit(self, userdata):
        """Called when leaving the state."""
        Logger.loginfo('Move0ReadJointState exiting')

    def on_start(self):
        """Called when behavior starts."""
        Logger.loginfo('Move0ReadJointState started')

    def on_stop(self):
        """Called when behavior stops."""
        Logger.loginfo('Move0ReadJointState stopped')