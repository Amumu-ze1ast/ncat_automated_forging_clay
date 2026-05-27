#!/usr/bin/env python3

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS

from flexbe_core import EventState, Logger
from flexbe_core.proxy import ProxyPublisher, ProxySubscriberCached
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np

import subprocess
import re


class Move3MoveToPoseState(EventState):
    """
    FlexBE state to move robot to a specific Cartesian pose (XYZ + RPY).
    
    Uses the move2_xyz_rpy C++ TRAC-IK solver to compute joint angles.
    
    Parameters:
    -- target_x         float   Target X position (meters)
    -- target_y         float   Target Y position (meters)
    -- target_z         float   Target Z position (meters)
    -- target_roll      float   Target roll orientation (radians)
    -- target_pitch     float   Target pitch orientation (radians)
    -- target_yaw       float   Target yaw orientation (radians)
    -- speed_scale      float   Speed scaling factor 0.1-1.0 (default: 0.3)
    -- tolerance        float   Position tolerance (default: 0.01 rad)
    
    Outcomes:
    <= done             Movement completed successfully
    <= failed           IK solving or movement failed
    
    Output Keys:
    #> joint_solution   The computed joint angles as list
    """

    def __init__(self, target_x, target_y, target_z, target_roll, target_pitch, target_yaw, 
                 speed_scale=0.3, tolerance=0.01):
        super(Move3MoveToPoseState, self).__init__(
            outcomes=['done', 'failed'],
            output_keys=['joint_solution']
        )
        
        # Parameters
        self._target_x = target_x
        self._target_y = target_y
        self._target_z = target_z
        self._target_roll = target_roll
        self._target_pitch = target_pitch
        self._target_yaw = target_yaw
        self._speed_scale = max(0.1, min(1.0, speed_scale))
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
        self._timeout = Duration(seconds=30.0)
        self._return = None
        self._target_joints = None

    def execute(self, userdata):
        """Execute state - called periodically while state is active."""
        if self._return is not None:
            return self._return
        
        # Get current joint positions
        actual = self._get_actual_positions()
        
        if actual is None:
            Logger.logwarn('No joint states received yet')
            return None
        
        # Check if target reached
        error = np.abs(actual - np.array(self._target_joints))
        
        if np.all(error < self._tolerance):
            Logger.loginfo('Target pose reached!')
            userdata.joint_solution = self._target_joints
            self._return = 'done'
            return 'done'
        
        # Check timeout
        elapsed = Move3MoveToPoseState._node.get_clock().now() - self._start_time
        if elapsed >= self._timeout:
            Logger.logwarn(f'Timeout after {elapsed.nanoseconds / S_TO_NS:.1f}s')
            self._return = 'failed'
            return 'failed'
        
        return None

    def on_enter(self, userdata):
        """Called when state becomes active."""
        self._return = None
        
        Logger.loginfo(f'Moving to pose: XYZ=({self._target_x:.3f}, {self._target_y:.3f}, {self._target_z:.3f})')
        Logger.loginfo(f'              RPY=({self._target_roll:.3f}, {self._target_pitch:.3f}, {self._target_yaw:.3f})')
        Logger.loginfo(f'Speed scale: {self._speed_scale}')
        
        # Call C++ TRAC-IK solver to get joint solution
        joint_solution = self._call_ik_solver()
        
        if not joint_solution:
            Logger.logerr('IK solving failed!')
            self._return = 'failed'
            return
        
        self._target_joints = joint_solution
        Logger.loginfo(f'IK Solution: [{joint_solution[0]:.4f}, {joint_solution[1]:.4f}, {joint_solution[2]:.4f}, {joint_solution[3]:.4f}, {joint_solution[4]:.4f}, {joint_solution[5]:.4f}, {joint_solution[6]:.4f}]')
        
        # Send trajectory command
        self._send_trajectory()
        self._start_time = Move3MoveToPoseState._node.get_clock().now()

    def on_exit(self, userdata):
        """Called when leaving the state."""
        Logger.loginfo('MoveToPose state exiting')

    def on_start(self):
        """Called when behavior starts."""
        Logger.loginfo('MoveToPose state started')

    def on_stop(self):
        """Called when behavior stops."""
        Logger.loginfo('MoveToPose state stopped')

    # ==================== Helper Methods ====================

    def _call_ik_solver(self):
        """
        Call the C++ move2_xyz_rpy executable.
        Returns joint angles as list or None if failed.
        """
        try:
            # Build command
            cmd = [
                'ros2', 'run', 'trac_ik_examples', 'move2_xyz_rpy',
                str(self._target_x),
                str(self._target_y),
                str(self._target_z),
                str(self._target_roll),
                str(self._target_pitch),
                str(self._target_yaw)
            ]
            
            Logger.loginfo('Calling TRAC-IK solver...')
            
            # Call executable
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                Logger.logerr(f'TRAC-IK solver failed with code {result.returncode}')
                Logger.logerr(f'Error: {result.stderr}')
                return None
            
            # Parse output to extract joint angles
            joint_solution = self._parse_ik_output(result.stdout)
            
            if joint_solution:
                Logger.loginfo('Successfully computed IK solution')
            
            return joint_solution
            
        except subprocess.TimeoutExpired:
            Logger.logerr('TRAC-IK solver timeout!')
            return None
        except Exception as e:
            Logger.logerr(f'Error calling TRAC-IK: {str(e)}')
            return None

    def _parse_ik_output(self, output):
        """
        Parse the C++ solver output to extract joint angles.
        Expected format: np.array([j1, j2, j3, j4, j5, j6, j7])
        """
        # Look for np.array format
        lines = output.split('\n')
        
        for line in lines:
            if 'np.array([' in line:
                # Extract content between square brackets
                start = line.find('[')
                end = line.find(']')
                if start != -1 and end != -1:
                    content = line[start+1:end]
                    
                    try:
                        joints = [float(x.strip()) for x in content.split(',')]
                        
                        if len(joints) == 7:
                            Logger.loginfo(f'Parsed IK solution: [{joints[0]:.3f}, {joints[1]:.3f}, ...]')
                            return joints
                    except ValueError:
                        continue
        
        Logger.logerr('Failed to parse IK solution from output')
        return None

    def _compute_duration(self):
        """
        Compute movement duration based on joint distance and speed scaling.
        """
        actual = self._get_actual_positions()
        if actual is None:
            return 2.0  # Default fallback
        
        target = np.array(self._target_joints)
        
        # Maximum joint displacement
        dist = np.max(np.abs(target - actual))
        
        # Base velocity (rad/s)
        base_vel = 0.8
        
        # Apply speed scaling
        scaled_vel = base_vel * self._speed_scale
        
        # Compute time
        t = dist / scaled_vel
        
        # Never go below 0.5 seconds
        return max(t, 0.5)

    def _send_trajectory(self):
        """Send joint trajectory command."""
        actual = self._get_actual_positions()
        
        # Compute duration based on distance
        duration = self._compute_duration()
        
        Logger.loginfo(f'Movement duration: {duration:.2f}s')
        
        traj = JointTrajectory()
        traj.joint_names = self._joint_names
        
        point = JointTrajectoryPoint()
        point.positions = self._target_joints
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1.0) * 1e9)
        
        traj.points = [point]
        
        self._pub.publish(self._traj_topic, traj)
        Logger.loginfo('Trajectory command sent')

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