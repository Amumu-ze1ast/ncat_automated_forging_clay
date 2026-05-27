#!/usr/bin/env python3

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS

from flexbe_core import EventState, Logger
from flexbe_core.proxy import ProxyPublisher, ProxySubscriberCached
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np

import subprocess


class Move7IncrementalWaypointsState(EventState):
    """
    FlexBE state for incremental XYZ movement with multiple waypoints while maintaining orientation.
    
    Uses the incremental_xyz_waypoints_ik_solver C++ executable to compute waypoint IK solutions.
    
    Parameters:
    -- delta_x          float   Incremental X movement in meters (default: 0.0)
    -- delta_y          float   Incremental Y movement in meters (default: 0.0)
    -- delta_z          float   Incremental Z movement in meters (default: 0.0)
    -- num_waypoints    int     Number of intermediate waypoints (default: 5)
    -- speed_scale      float   Speed scaling factor 0.1-1.0 (default: 0.3)
    -- tolerance        float   Position tolerance in radians (default: 0.01)
    
    Outcomes:
    <= done             Movement completed successfully
    <= failed           IK solving or movement failed
    
    Output Keys:
    #> waypoints_count  Number of waypoints generated
    #> final_joints     Final joint configuration reached
    """

    def __init__(self, delta_x=0.0, delta_y=0.0, delta_z=0.0, num_waypoints=5, speed_scale=0.3, tolerance=0.01):
        super(Move7IncrementalWaypointsState, self).__init__(
            outcomes=['done', 'failed'],
            output_keys=['waypoints_count', 'final_joints']
        )
        
        # Parameters
        self._delta_x = delta_x
        self._delta_y = delta_y
        self._delta_z = delta_z
        self._num_waypoints = num_waypoints
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
        self._timeout = Duration(seconds=60.0)
        self._return = None
        self._waypoint_trajectories = []
        self._current_waypoint_index = 0

    def execute(self, userdata):
        """Execute state - called periodically while state is active."""
        if self._return is not None:
            return self._return
        
        # Get current joint positions
        actual = self._get_actual_positions()
        
        if actual is None:
            Logger.logwarn('No joint states received yet')
            return None
        
        # Check if all waypoints completed
        if self._current_waypoint_index >= len(self._waypoint_trajectories):
            Logger.loginfo('All waypoints completed!')
            userdata.waypoints_count = len(self._waypoint_trajectories)
            userdata.final_joints = self._waypoint_trajectories[-1] if self._waypoint_trajectories else None
            self._return = 'done'
            return 'done'
        
        # Check if current waypoint reached
        current_target = self._waypoint_trajectories[self._current_waypoint_index]
        error = np.abs(actual - np.array(current_target))
        
        if np.all(error < self._tolerance):
            Logger.loginfo(f'Waypoint {self._current_waypoint_index + 1}/{len(self._waypoint_trajectories)} reached')
            self._current_waypoint_index += 1
            
            # Send next waypoint if available
            if self._current_waypoint_index < len(self._waypoint_trajectories):
                self._send_waypoint(self._current_waypoint_index)
            
            return None
        
        # Check timeout
        elapsed = Move7IncrementalWaypointsState._node.get_clock().now() - self._start_time
        if elapsed >= self._timeout:
            Logger.logwarn(f'Timeout after {elapsed.nanoseconds / S_TO_NS:.1f}s')
            self._return = 'failed'
            return 'failed'
        
        return None

    def on_enter(self, userdata):
        """Called when state becomes active."""
        self._return = None
        self._current_waypoint_index = 0
        
        Logger.loginfo(f'Incremental XYZ movement: ΔX={self._delta_x}, ΔY={self._delta_y}, ΔZ={self._delta_z}')
        Logger.loginfo(f'Number of waypoints: {self._num_waypoints}')
        Logger.loginfo(f'Speed scale: {self._speed_scale}')
        
        # Call C++ IK solver
        joint_waypoints = self._call_incremental_xyz_waypoints_solver()
        
        if not joint_waypoints:
            Logger.logerr('IK solving failed!')
            self._return = 'failed'
            return
        
        self._waypoint_trajectories = joint_waypoints
        Logger.loginfo(f'Generated {len(joint_waypoints)} joint waypoints')
        
        # Print all IK solutions
        Logger.loginfo('========== IK Solutions ==========')
        for i, waypoint in enumerate(joint_waypoints):
            Logger.loginfo(f'Waypoint {i+1}: [{waypoint[0]:.4f}, {waypoint[1]:.4f}, {waypoint[2]:.4f}, {waypoint[3]:.4f}, {waypoint[4]:.4f}, {waypoint[5]:.4f}, {waypoint[6]:.4f}]')
        Logger.loginfo('==================================')
        
        # Send first waypoint
        self._start_time = Move7IncrementalWaypointsState._node.get_clock().now()
        self._send_waypoint(0)

    def on_exit(self, userdata):
        """Called when leaving the state."""
        Logger.loginfo('IncrementalXYZWaypointsMove state exiting')

    def on_start(self):
        """Called when behavior starts."""
        Logger.loginfo('IncrementalXYZWaypointsMove state started')

    def on_stop(self):
        """Called when behavior stops."""
        Logger.loginfo('IncrementalXYZWaypointsMove state stopped')

    # ==================== Helper Methods ====================

    def _call_incremental_xyz_waypoints_solver(self):
        """
        Call the C++ incremental_xyz_waypoints_ik_solver executable.
        Returns list of joint waypoints or empty list if failed.
        """
        try:
            # Build command
            cmd = [
                'ros2', 'run', 'trac_ik_examples', 'move5_increment_xyz_waypoint',
                str(self._delta_x),
                str(self._delta_y),
                str(self._delta_z),
                str(self._num_waypoints)
            ]
            
            Logger.loginfo('Calling incremental XYZ waypoints IK solver...')
            
            # Call executable
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                Logger.logerr(f'IK solver failed with code {result.returncode}')
                Logger.logerr(f'Error: {result.stderr}')
                return []
            
            # Parse output to extract joint angles
            joint_waypoints = self._parse_ik_output(result.stdout)
            
            Logger.loginfo(f'Successfully parsed {len(joint_waypoints)} IK solutions')
            return joint_waypoints
            
        except subprocess.TimeoutExpired:
            Logger.logerr('IK solver timeout!')
            return []
        except Exception as e:
            Logger.logerr(f'Error calling IK solver: {str(e)}')
            return []

    def _parse_ik_output(self, output):
        """
        Parse the C++ solver output to extract joint angle solutions.
        Expected format: np.array([j1, j2, j3, j4, j5, j6, j7]),
        """
        joint_waypoints = []
        
        Logger.loginfo('Parsing IK solver output...')
        
        # Look for lines with np.array format
        lines = output.split('\n')
        
        for line in lines:
            # Look for lines containing np.array
            if 'np.array([' in line:
                # Extract content between square brackets
                start = line.find('[')
                end = line.find(']')
                if start != -1 and end != -1:
                    content = line[start+1:end]
                    
                    try:
                        # Split by comma and parse as floats
                        joints = [float(x.strip()) for x in content.split(',')]
                        
                        # Only accept if we have exactly 7 joints
                        if len(joints) == 7:
                            joint_waypoints.append(joints)
                            Logger.loginfo(f'Parsed waypoint {len(joint_waypoints)}: [{joints[0]:.3f}, {joints[1]:.3f}, ...]')
                    except ValueError:
                        continue
        
        Logger.loginfo(f'Total waypoints parsed: {len(joint_waypoints)}')
        return joint_waypoints

    def _compute_time_for_waypoint(self, start, target):
        """
        Compute time_from_start based on joint distance and speed scaling.
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

    def _send_waypoint(self, waypoint_index):
        """Send joint trajectory command for a specific waypoint with dynamic timing."""
        if waypoint_index >= len(self._waypoint_trajectories):
            return
        
        # Get current actual position
        actual = self._get_actual_positions()
        if actual is None:
            Logger.logwarn('Cannot send waypoint - no joint states available')
            return
        
        target = self._waypoint_trajectories[waypoint_index]
        
        # Compute dynamic time based on distance
        duration = self._compute_time_for_waypoint(actual, target)
        
        Logger.loginfo(f'Waypoint {waypoint_index + 1} duration: {duration:.2f}s (based on joint distance)')
        
        traj = JointTrajectory()
        traj.joint_names = self._joint_names
        
        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1.0) * 1e9)
        
        traj.points = [point]
        
        self._pub.publish(self._traj_topic, traj)
        Logger.loginfo(f'Sent waypoint {waypoint_index + 1}/{len(self._waypoint_trajectories)}')

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