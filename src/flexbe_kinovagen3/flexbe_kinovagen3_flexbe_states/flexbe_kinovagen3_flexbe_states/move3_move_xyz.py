#!/usr/bin/env python3

from rclpy.duration import Duration
from rclpy.constants import S_TO_NS

from flexbe_core import EventState, Logger
from flexbe_core.proxy import ProxyPublisher, ProxySubscriberCached
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import numpy as np

import subprocess


class Move3MoveState(EventState):
    """
    FlexBE state for Cartesian movement to XYZ position maintaining current orientation.
    
    Moves to target XYZ position while keeping current end-effector orientation (RPY).
    
    Parameters:
    -- target_x         float   Target X position (meters)
    -- target_y         float   Target Y position (meters)
    -- target_z         float   Target Z position (meters)
    -- speed_scale      float   Speed scaling factor 0.1-1.0 (default: 0.3)
    -- tolerance        float   Position tolerance (default: 0.01 rad)
    
    Outcomes:
    <= done             Movement completed successfully
    <= failed           IK solving or movement failed
    
    Output Keys:
    #> joint_solution   Joint angles solution (7 values)
    """

    def __init__(self, target_x, target_y, target_z, speed_scale=0.3, tolerance=0.01):
        super(Move3MoveState, self).__init__(
            outcomes=['done', 'failed'],
            output_keys=['joint_solution']
        )
        
        # Parameters
        self._target_x = target_x
        self._target_y = target_y
        self._target_z = target_z
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
            Logger.loginfo('Target position reached!')
            userdata.joint_solution = self._target_joints
            self._return = 'done'
            return 'done'
        
        # Check timeout
        elapsed = Move3MoveState._node.get_clock().now() - self._start_time
        if elapsed >= self._timeout:
            Logger.logwarn(f'Timeout after {elapsed.nanoseconds / S_TO_NS:.1f}s')
            self._return = 'failed'
            return 'failed'
        
        return None

    def on_enter(self, userdata):
        """Called when state becomes active."""
        self._return = None
        
        Logger.loginfo(f'Target XYZ: ({self._target_x:.3f}, {self._target_y:.3f}, {self._target_z:.3f})')
        Logger.loginfo('Orientation: Maintaining current RPY')
        Logger.loginfo(f'Speed scale: {self._speed_scale}')
        
        # Solve IK for target position
        joint_solution = self._solve_ik(self._target_x, self._target_y, self._target_z)
        
        if not joint_solution:
            Logger.logerr('IK solving failed!')
            self._return = 'failed'
            return
        
        self._target_joints = joint_solution
        Logger.loginfo(f'IK solution: {[f"{j:.4f}" for j in joint_solution]}')
        
        # Send trajectory
        self._send_trajectory(joint_solution)
        
        self._start_time = Move3MoveState._node.get_clock().now()

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
        Calls C++ move2_xyz.
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
        """Send joint trajectory command."""
        # Get current actual position
        actual = self._get_actual_positions()
        if actual is None:
            Logger.logwarn('Cannot send trajectory - no joint states available')
            return
        
        # Compute dynamic time based on distance
        duration = self._compute_duration(actual, target_joints)
        
        Logger.loginfo(f'Trajectory duration: {duration:.2f}s')
        
        traj = JointTrajectory()
        traj.joint_names = self._joint_names
        
        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1.0) * 1e9)
        
        traj.points = [point]
        
        self._pub.publish(self._traj_topic, traj)
        Logger.loginfo('Trajectory sent')

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