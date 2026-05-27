#!/usr/bin/env python3

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS

from flexbe_core import EventState, Logger
from flexbe_core.proxy import ProxyPublisher, ProxySubscriberCached
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np

import subprocess


class Move4IncrementalXYZMoveState(EventState):
    """
    FlexBE state for incremental XYZ movement while maintaining current orientation.
    
    Uses the move3_increment_xyz C++ executable to compute IK solution.
    
    Parameters:
    -- delta_x          float   Incremental X movement in meters (default: 0.0)
    -- delta_y          float   Incremental Y movement in meters (default: 0.0)
    -- delta_z          float   Incremental Z movement in meters (default: 0.0)
    -- speed_scale      float   Speed scaling factor 0.1-1.0 (default: 0.3)
    -- tolerance        float   Position tolerance in radians (default: 0.01)
    
    Outcomes:
    <= done             Movement completed successfully
    <= failed           IK solving or movement failed
    
    Output Keys:
    #> final_joints     Final joint configuration reached
    """

    def __init__(self, delta_x=0.0, delta_y=0.0, delta_z=0.0, speed_scale=0.3, tolerance=0.01):
        super(Move4IncrementalXYZMoveState, self).__init__(
            outcomes=['done', 'failed'],
            output_keys=['final_joints']
        )
        
        # Parameters
        self._delta_x = delta_x
        self._delta_y = delta_y
        self._delta_z = delta_z
        self._speed_scale = max(0.1, min(1.0, speed_scale))  # Clamp between 0.1 and 1.0
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
        if self._target_joints is not None:
            error = np.abs(actual - np.array(self._target_joints))
            
            if np.all(error < self._tolerance):
                Logger.loginfo('Target position reached!')
                userdata.final_joints = self._target_joints
                self._return = 'done'
                return 'done'
        
        # Check timeout
        elapsed = Move4IncrementalXYZMoveState._node.get_clock().now() - self._start_time
        if elapsed >= self._timeout:
            Logger.logwarn(f'Timeout after {elapsed.nanoseconds / S_TO_NS:.1f}s')
            self._return = 'failed'
            return 'failed'
        
        return None

    def on_enter(self, userdata):
        """Called when state becomes active."""
        self._return = None
        self._target_joints = None
        
        Logger.loginfo(f'Incremental XYZ movement: ΔX={self._delta_x}, ΔY={self._delta_y}, ΔZ={self._delta_z}')
        Logger.loginfo(f'Speed scale: {self._speed_scale}')
        
        # Call C++ IK solver
        joint_solution = self._call_incremental_xyz_solver()
        
        if not joint_solution:
            Logger.logerr('IK solving failed!')
            self._return = 'failed'
            return
        
        self._target_joints = joint_solution
        Logger.loginfo(f'IK Solution: [{joint_solution[0]:.4f}, {joint_solution[1]:.4f}, {joint_solution[2]:.4f}, {joint_solution[3]:.4f}, {joint_solution[4]:.4f}, {joint_solution[5]:.4f}, {joint_solution[6]:.4f}]')
        
        # Send trajectory command
        self._start_time = Move4IncrementalXYZMoveState._node.get_clock().now()
        self._send_trajectory(joint_solution)

    def on_exit(self, userdata):
        """Called when leaving the state."""
        Logger.loginfo('IncrementalXYZMove state exiting')

    def on_start(self):
        """Called when behavior starts."""
        Logger.loginfo('IncrementalXYZMove state started')

    def on_stop(self):
        """Called when behavior stops."""
        Logger.loginfo('IncrementalXYZMove state stopped')

    # ==================== Helper Methods ====================

    def _call_incremental_xyz_solver(self):
        """
        Call the C++ move3_increment_xyz executable.
        Returns joint angle solution or None if failed.
        """
        try:
            # Build command
            cmd = [
                'ros2', 'run', 'trac_ik_examples', 'move3_increment_xyz',
                str(self._delta_x),
                str(self._delta_y),
                str(self._delta_z)
            ]
            
            Logger.loginfo('Calling incremental XYZ IK solver...')
            
            # Call executable
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                Logger.logerr(f'IK solver failed with code {result.returncode}')
                Logger.logerr(f'Error: {result.stderr}')
                return None
            
            # Parse output to extract joint angles
            joint_solution = self._parse_ik_output(result.stdout)
            
            if joint_solution:
                Logger.loginfo('Successfully parsed IK solution')
            else:
                Logger.logerr('Failed to parse IK solution from output')
            
            return joint_solution
            
        except subprocess.TimeoutExpired:
            Logger.logerr('IK solver timeout!')
            return None
        except Exception as e:
            Logger.logerr(f'Error calling IK solver: {str(e)}')
            return None

    def _parse_ik_output(self, output):
        """
        Parse the C++ solver output to extract joint angle solution.
        Expected format: np.array([j1, j2, j3, j4, j5, j6, j7])
        """
        lines = output.split('\n')
        
        for line in lines:
            # Look for line containing np.array
            if 'np.array([' in line:
                # Extract content between square brackets
                start = line.find('[')
                end = line.find(']')
                if start != -1 and end != -1:
                    content = line[start+1:end]
                    
                    try:
                        # Split by comma and parse as floats
                        joints = [float(x.strip()) for x in content.split(',')]
                        
                        # Verify we have exactly 7 joints
                        if len(joints) == 7:
                            return joints
                    except ValueError:
                        continue
        
        return None

    def _compute_time_for_movement(self, start, target):
        """
        Compute movement time based on joint distance and speed scaling.
        """
        start = np.array(start)
        target = np.array(target)
        
        # Maximum joint displacement
        dist = np.max(np.abs(target - start))
        
        # Base velocity (rad/s)
        base_vel = 0.8
        
        # Apply speed scaling
        scaled_vel = base_vel * self._speed_scale
        
        # Compute time
        t = dist / scaled_vel
        
        # Never go below 0.5 seconds
        return max(t, 0.5)

    def _send_trajectory(self, target_joints):
        """Send joint trajectory command."""
        # Get current position
        actual = self._get_actual_positions()
        if actual is None:
            Logger.logwarn('Cannot send trajectory - no joint states available')
            return
        
        # Compute dynamic time based on distance
        duration = self._compute_time_for_movement(actual, target_joints)
        
        Logger.loginfo(f'Movement duration: {duration:.2f}s (based on joint distance)')
        
        traj = JointTrajectory()
        traj.joint_names = self._joint_names
        
        point = JointTrajectoryPoint()
        point.positions = target_joints
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