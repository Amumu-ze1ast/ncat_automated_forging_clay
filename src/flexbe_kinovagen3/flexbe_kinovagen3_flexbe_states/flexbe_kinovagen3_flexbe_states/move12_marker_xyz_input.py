#!/usr/bin/env python3

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS

from flexbe_core import EventState, Logger
from flexbe_core.proxy import ProxyPublisher, ProxySubscriberCached
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np

import subprocess


class Move12MoveArucoState(EventState):
    """
    FlexBE state for Cartesian movement to marker Y,Z position while keeping current X.
    
    Takes marker_x, marker_y, marker_z from previous state (e.g., ArUco detection).
    - X: Uses current robot X position (marker_x is IGNORED)
    - Y: Moves to marker_y
    - Z: Moves to marker_z
    - Orientation: Maintains current RPY
    
    Parameters:
    -- speed_scale      float   Speed scaling factor 0.1-1.0 (default: 0.3)
    -- tolerance        float   Position tolerance (default: 0.01 rad)
    
    Input Keys:
    <= marker_x         X position from previous state (IGNORED - uses current robot X)
    <= marker_y         Y position from previous state (USED as target Y)
    <= marker_z         Z position from previous state (USED as target Z)
    
    Outcomes:
    <= done             Movement completed successfully
    <= failed           IK solving or movement failed
    
    Output Keys:
    #> joint_solution   Joint angles solution (7 values)
    #> target_x         Final target X used (current robot X)
    #> target_y         Final target Y used (marker Y)
    #> target_z         Final target Z used (marker Z)
    """

    def __init__(self, speed_scale=0.3, tolerance=0.01):
        super(Move12MoveArucoState, self).__init__(
            outcomes=['done', 'failed'],
            input_keys=['marker_x', 'marker_y', 'marker_z'],
            output_keys=['joint_solution', 'target_x', 'target_y', 'target_z']
        )
        
        # Parameters
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
        
        if self._target_joints is None:
            Logger.logerr('No target joints set!')
            self._return = 'failed'
            return 'failed'
        
        # Check if target reached
        error = np.abs(actual - np.array(self._target_joints))
        max_error = np.max(error)
        
        # Log progress every 2 seconds
        elapsed = Move12MoveArucoState._node.get_clock().now() - self._start_time
        elapsed_sec = elapsed.nanoseconds / 1e9
        
        # Check if robot is stuck (error not decreasing)
        if not hasattr(self, '_last_max_error'):
            self._last_max_error = max_error
            self._stuck_counter = 0
        
        # If error hasn't changed in 5 seconds, robot is stuck
        if abs(max_error - self._last_max_error) < 0.001:  # Error not changing
            self._stuck_counter += 1
            if self._stuck_counter > 10:  # 10 cycles of no change (~5 seconds)
                Logger.logerr(f'Robot appears stuck! Error not decreasing: {max_error:.4f} rad')
                Logger.logerr('Possible issues:')
                Logger.logerr('  1. Joint trajectory controller not running')
                Logger.logerr('  2. Controller not accepting commands')
                Logger.logerr('  3. Robot in error state')
                self._return = 'failed'
                return 'failed'
        else:
            self._stuck_counter = 0
        
        self._last_max_error = max_error
        
        if int(elapsed_sec) % 2 == 0 and int(elapsed_sec) > 0:
            Logger.loginfo(f'Moving... max error: {max_error:.4f} rad (tolerance: {self._tolerance})')
        
        if np.all(error < self._tolerance):
            Logger.loginfo(f'Target position reached! Final error: {max_error:.4f} rad')
            userdata.joint_solution = self._target_joints
            self._return = 'done'
            return 'done'
        
        # Check timeout
        if elapsed >= self._timeout:
            Logger.logwarn(f'Timeout after {elapsed_sec:.1f}s, max error: {max_error:.4f} rad')
            self._return = 'failed'
            return 'failed'
        
        # Still moving
        return None

    def on_enter(self, userdata):
        """Called when state becomes active."""
        self._return = None
        
        # Reset stuck detection
        if hasattr(self, '_last_max_error'):
            delattr(self, '_last_max_error')
        if hasattr(self, '_stuck_counter'):
            delattr(self, '_stuck_counter')
        
        # Get marker position from input keys
        try:
            marker_x = userdata.marker_x  # Will be ignored
            marker_y = userdata.marker_y  # Target Y
            marker_z = userdata.marker_z  # Target Z
            Logger.loginfo(f'Received marker position: ({marker_x:.3f}, {marker_y:.3f}, {marker_z:.3f})')
            Logger.loginfo('Note: marker_x will be IGNORED, using current robot X instead')
        except KeyError as e:
            Logger.logerr(f'Missing required input key: {str(e)}')
            Logger.logerr('This state requires marker_x, marker_y, marker_z from previous state (e.g., ArUco detection)')
            self._return = 'failed'
            return
        
        # Get current end-effector X position from TF
        import tf2_ros
        import rclpy
        import time
        
        try:
            tf_buffer = tf2_ros.Buffer()
            tf_listener = tf2_ros.TransformListener(tf_buffer, Move12MoveArucoState._node)
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
            self._return = 'failed'
            return
        
        # Target position: current X, marker Y and Z
        target_x = current_x  # KEEP CURRENT X
        target_y = marker_y   # USE MARKER Y
        target_z = marker_z   # USE MARKER Z
        
        Logger.loginfo(f'Target position: X={target_x:.3f} (CURRENT X), Y={target_y:.3f} (MARKER Y), Z={target_z:.3f} (MARKER Z)')
        Logger.loginfo('Orientation: Maintaining current RPY')
        Logger.loginfo(f'Speed scale: {self._speed_scale}')
        
        # Store in userdata for output
        userdata.target_x = target_x
        userdata.target_y = target_y
        userdata.target_z = target_z
        
        # Solve IK for target position
        joint_solution = self._solve_ik(target_x, target_y, target_z)
        
        if not joint_solution:
            Logger.logerr('IK solving failed!')
            self._return = 'failed'
            return
        
        # Validate IK solution - check for large jumps
        current = self._get_actual_positions()
        if current is not None:
            max_jump = np.max(np.abs(np.array(joint_solution) - current))
            if max_jump > 3.5:  # More than 200 degrees
                Logger.logerr(f'IK solution requires too large joint movement: {max_jump:.2f} rad ({np.degrees(max_jump):.1f} deg)')
                Logger.logerr('Target position may be unreachable from current configuration')
                self._return = 'failed'
                return
        
        self._target_joints = joint_solution
        Logger.loginfo(f'IK solution: {[f"{j:.4f}" for j in joint_solution]}')
        
        # Send trajectory
        self._send_trajectory(joint_solution)
        
        self._start_time = Move12MoveArucoState._node.get_clock().now()

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

    def _solve_ik(self, x, y, z):
        """
        Solve IK for XYZ position using current orientation.
        Calls C++ single_pose_ik_solver.
        """
        try:
            cmd = [
                'ros2', 'run', 'trac_ik_examples', 'move2_xyz',
                str(x),
                str(y),
                str(z)
            ]
            
            Logger.loginfo(f'Calling IK solver: ({x:.3f}, {y:.3f}, {z:.3f})')
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                Logger.logerr(f'IK solver failed with code {result.returncode}')
                Logger.logerr(f'Error: {result.stderr}')
                return None
            
            # Parse output
            joint_solution = self._parse_ik_output(result.stdout)
            
            if joint_solution:
                Logger.loginfo('IK solution found!')
                return joint_solution
            else:
                Logger.logerr('Failed to parse IK output')
                return None
            
        except subprocess.TimeoutExpired:
            Logger.logerr('IK solver timeout!')
            return None
        except Exception as e:
            Logger.logerr(f'Error calling IK solver: {str(e)}')
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

    def _compute_duration(self, start, target):
        """Compute trajectory duration based on joint distance."""
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
        """Send joint trajectory command with waypoints."""
        # Get current actual position
        actual = self._get_actual_positions()
        if actual is None:
            Logger.logwarn('Cannot send trajectory - no joint states available')
            return
        
        # Compute dynamic time based on distance
        total_duration = self._compute_duration(actual, target_joints)
        
        # Check if movement is reasonable
        max_delta = np.max(np.abs(np.array(target_joints) - actual))
        Logger.loginfo(f'Maximum joint change: {max_delta:.4f} rad ({np.degrees(max_delta):.1f} deg)')
        
        if max_delta > 3.14:  # More than 180 degrees
            Logger.logwarn(f'WARNING: Large joint movement detected! Max delta: {max_delta:.4f} rad')
        
        Logger.loginfo(f'Total trajectory duration: {total_duration:.2f}s')
        Logger.loginfo(f'Current joints: {[f"{j:.3f}" for j in actual]}')
        Logger.loginfo(f'Target joints:  {[f"{j:.3f}" for j in target_joints]}')
        
        # Create waypoints (5 intermediate points)
        num_waypoints = 5
        waypoints = []
        
        for i in range(1, num_waypoints + 1):
            t = i / num_waypoints
            waypoint = actual + t * (np.array(target_joints) - actual)
            waypoint_time = t * total_duration
            waypoints.append((waypoint.tolist(), waypoint_time))
        
        Logger.loginfo(f'Generated {num_waypoints} waypoints')
        
        # Create trajectory with multiple points
        traj = JointTrajectory()
        traj.joint_names = self._joint_names
        traj.header.stamp = Move12MoveArucoState._node.get_clock().now().to_msg()
        
        for i, (waypoint_joints, waypoint_time) in enumerate(waypoints):
            point = JointTrajectoryPoint()
            point.positions = waypoint_joints
            point.time_from_start.sec = int(waypoint_time)
            point.time_from_start.nanosec = int((waypoint_time % 1.0) * 1e9)
            traj.points.append(point)
            Logger.loginfo(f'  Waypoint {i+1}: time={waypoint_time:.2f}s')
        
        self._pub.publish(self._traj_topic, traj)
        Logger.loginfo('Multi-waypoint trajectory published')

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