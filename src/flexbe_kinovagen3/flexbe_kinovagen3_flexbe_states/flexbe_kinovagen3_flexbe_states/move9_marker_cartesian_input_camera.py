#!/usr/bin/env python3

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS

from flexbe_core import EventState, Logger
from flexbe_core.proxy import ProxyPublisher, ProxySubscriberCached
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np

import subprocess


class Move9CartesianMoveState(EventState):
    """
    FlexBE state for Cartesian movement to marker position with offset and waypoints.
    
    Moves to marker position (with offset) through multiple waypoints for smooth motion.
    
    Parameters:
    -- target_roll      float   Target roll orientation (radians, default: 0.0)
    -- target_pitch     float   Target pitch orientation (radians, default: 3.14159)
    -- target_yaw       float   Target yaw orientation (radians, default: 0.0)
    -- x_offset         float   X-axis offset from marker (meters, default: 0.0)
    -- num_waypoints    int     Number of intermediate waypoints (default: 10)
    -- speed_scale      float   Speed scaling factor 0.1-1.0 (default: 0.3)
    -- tolerance        float   Position tolerance (default: 0.01 rad)
    
    Input Keys:
    <= marker_x         X position from previous state (e.g., ArUco detection)
    <= marker_y         Y position from previous state
    <= marker_z         Z position from previous state
    
    Outcomes:
    <= done             Movement completed successfully
    <= failed           IK solving or movement failed
    
    Output Keys:
    #> waypoints_count  Number of waypoints generated
    """

    def __init__(self, target_roll=0.0, target_pitch=3.14159, target_yaw=0.0,
                 x_offset=0.0, num_waypoints=10, speed_scale=0.3, tolerance=0.01):
        super(Move9CartesianMoveState, self).__init__(
            outcomes=['done', 'failed'],
            input_keys=['marker_x', 'marker_y', 'marker_z'],
            output_keys=['waypoints_count']
        )
        
        # Parameters
        self._target_roll = target_roll
        self._target_pitch = target_pitch
        self._target_yaw = target_yaw
        self._x_offset = x_offset
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
        
        # Target position
        self._target_x = None
        self._target_y = None
        self._target_z = None

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
        elapsed = Move9CartesianMoveState._node.get_clock().now() - self._start_time
        if elapsed >= self._timeout:
            Logger.logwarn(f'Timeout after {elapsed.nanoseconds / S_TO_NS:.1f}s')
            self._return = 'failed'
            return 'failed'
        
        return None

    def on_enter(self, userdata):
        """Called when state becomes active."""
        self._return = None
        self._current_waypoint_index = 0
        
        # Get marker position from input keys
        try:
            marker_x = userdata.marker_x
            marker_y = userdata.marker_y
            marker_z = userdata.marker_z
            Logger.loginfo(f'Received marker position: ({marker_x:.3f}, {marker_y:.3f}, {marker_z:.3f})')
        except KeyError as e:
            Logger.logerr(f'Missing required input key: {str(e)}')
            Logger.logerr('This state requires marker_x, marker_y, marker_z from previous state (e.g., ArUcoDetectionState)')
            self._return = 'failed'
            return
        
        # Apply offset: target_x = marker_x - x_offset
        self._target_x = marker_x - self._x_offset
        self._target_y = marker_y
        self._target_z = marker_z
        
        if self._x_offset != 0.0:
            Logger.loginfo(f'X offset: {self._x_offset:.3f}m')
        Logger.loginfo(f'Marker XYZ: ({marker_x:.3f}, {marker_y:.3f}, {marker_z:.3f})')
        Logger.loginfo(f'Target XYZ: ({self._target_x:.3f}, {self._target_y:.3f}, {self._target_z:.3f})')
        Logger.loginfo(f'Target orientation: RPY=({self._target_roll:.3f}, {self._target_pitch:.3f}, {self._target_yaw:.3f})')
        Logger.loginfo(f'Number of waypoints: {self._num_waypoints}')
        Logger.loginfo(f'Speed scale: {self._speed_scale}')
        
        # Generate waypoints from current position to target
        joint_waypoints = self._generate_waypoints_to_target()
        
        if not joint_waypoints:
            Logger.logerr('Waypoint generation failed!')
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
        self._start_time = Move9CartesianMoveState._node.get_clock().now()
        self._send_waypoint(0)

    def on_exit(self, userdata):
        """Called when leaving the state."""
        Logger.loginfo('CartesianMove state exiting')

    def on_start(self):
        """Called when behavior starts."""
        Logger.loginfo('CartesianMove state started')

    def on_stop(self):
        """Called when behavior stops."""
        Logger.loginfo('CartesianMove state stopped')

    # ==================== Helper Methods ====================

    def _generate_waypoints_to_target(self):
        """
        Generate waypoints from current position to target position.
        Uses C++ solver to compute IK for each waypoint.
        """
        import tf2_ros
        import rclpy
        
        try:
            # Get current end-effector position from TF
            tf_buffer = tf2_ros.Buffer()
            tf_listener = tf2_ros.TransformListener(tf_buffer, Move9CartesianMoveState._node)
            
            # Wait for TF to be ready
            import time
            time.sleep(0.5)
            
            transform = tf_buffer.lookup_transform(
                'base_link',
                'end_effector_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
            
            current_x = transform.transform.translation.x
            current_y = transform.transform.translation.y
            current_z = transform.transform.translation.z
            
            Logger.loginfo(f'Current end-effector position: ({current_x:.3f}, {current_y:.3f}, {current_z:.3f})')
            
        except Exception as e:
            Logger.logerr(f'Failed to get current position from TF: {str(e)}')
            return []
        
        # Generate Cartesian waypoints by linear interpolation
        waypoints = []
        for i in range(self._num_waypoints + 1):
            t = i / self._num_waypoints
            waypoint_x = current_x + t * (self._target_x - current_x)
            waypoint_y = current_y + t * (self._target_y - current_y)
            waypoint_z = current_z + t * (self._target_z - current_z)
            waypoints.append((waypoint_x, waypoint_y, waypoint_z))
        
        Logger.loginfo(f'Generated {len(waypoints)} Cartesian waypoints')
        
        # Compute IK for each waypoint
        joint_waypoints = []
        for i, (wx, wy, wz) in enumerate(waypoints):
            joints = self._solve_ik_for_pose(wx, wy, wz)
            if joints:
                joint_waypoints.append(joints)
                Logger.loginfo(f'IK solved for waypoint {i+1}/{len(waypoints)}')
            else:
                Logger.logerr(f'IK failed for waypoint {i+1} at ({wx:.3f}, {wy:.3f}, {wz:.3f})')
                return []  # Fail if any waypoint fails
        
        return joint_waypoints

    def _solve_ik_for_pose(self, x, y, z):
        """
        Solve IK for a single XYZ pose with target orientation.
        Returns joint angles as list or None if failed.
        """
        try:
            cmd = [
                'ros2', 'run', 'trac_ik_examples', 'move1_cartesian',
                str(x),
                str(y),
                str(z),
                str(self._target_roll),
                str(self._target_pitch),
                str(self._target_yaw)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None
            
            # Parse output
            return self._parse_ik_output(result.stdout)
            
        except Exception as e:
            Logger.logerr(f'IK solver error: {str(e)}')
            return None

    def _parse_ik_output(self, output):
        """
        Parse the C++ solver output to extract joint angles.
        Expected format: np.array([j1, j2, j3, j4, j5, j6, j7])
        """
        lines = output.split('\n')
        
        for line in lines:
            if 'np.array([' in line:
                start = line.find('[')
                end = line.find(']')
                if start != -1 and end != -1:
                    content = line[start+1:end]
                    
                    try:
                        joints = [float(x.strip()) for x in content.split(',')]
                        
                        if len(joints) == 7:
                            return joints
                    except ValueError:
                        continue
        
        return None

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