#!/usr/bin/env python3

from flexbe_core import EventState, Logger
import tf2_ros
import rclpy


class Move00ReadEndEffectorPoseState(EventState):
    """
    FlexBE state for reading current end-effector XYZ position and RPY orientation.
    
    Reads from TF transform between base_link and end_effector_link.
    
    Parameters:
    -- timeout          float   Timeout in seconds to wait for transform (default: 5.0)
    
    Outcomes:
    <= done             Successfully read end-effector pose
    <= failed           Failed to read pose (timeout or error)
    
    Output Keys:
    #> ee_x             End-effector X position (meters)
    #> ee_y             End-effector Y position (meters)
    #> ee_z             End-effector Z position (meters)
    #> ee_roll          End-effector roll orientation (radians)
    #> ee_pitch         End-effector pitch orientation (radians)
    #> ee_yaw           End-effector yaw orientation (radians)
    """

    def __init__(self, timeout=5.0):
        super(Move00ReadEndEffectorPoseState, self).__init__(
            outcomes=['done', 'failed'],
            output_keys=['ee_x', 'ee_y', 'ee_z', 'ee_roll', 'ee_pitch', 'ee_yaw']
        )
        
        # Parameters
        self._timeout = timeout
        
        # TF
        self._tf_buffer = None
        self._tf_listener = None
        
        # State management
        self._start_time = None
        self._return = None

    def execute(self, userdata):
        """Execute state - called periodically while state is active."""
        if self._return is not None:
            return self._return
        
        # Try to get transform
        try:
            transform = self._tf_buffer.lookup_transform(
                'base_link',
                'end_effector_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0)
            )
            
            # Extract position
            ee_x = transform.transform.translation.x
            ee_y = transform.transform.translation.y
            ee_z = transform.transform.translation.z
            
            # Extract orientation (quaternion to RPY)
            import math
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            
            # Convert quaternion to roll, pitch, yaw
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            ee_roll = math.atan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (qw * qy - qz * qx)
            if abs(sinp) >= 1:
                ee_pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
            else:
                ee_pitch = math.asin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            ee_yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # Store in userdata
            userdata.ee_x = ee_x
            userdata.ee_y = ee_y
            userdata.ee_z = ee_z
            userdata.ee_roll = ee_roll
            userdata.ee_pitch = ee_pitch
            userdata.ee_yaw = ee_yaw
            
            Logger.loginfo(f'End-effector position: ({ee_x:.4f}, {ee_y:.4f}, {ee_z:.4f})')
            Logger.loginfo(f'End-effector orientation (RPY): ({ee_roll:.4f}, {ee_pitch:.4f}, {ee_yaw:.4f})')
            
            self._return = 'done'
            return 'done'
            
        except Exception as e:
            # Check timeout
            elapsed = Move00ReadEndEffectorPoseState._node.get_clock().now() - self._start_time
            elapsed_sec = elapsed.nanoseconds / 1e9
            
            if elapsed_sec >= self._timeout:
                Logger.logerr(f'Timeout after {elapsed_sec:.1f}s - failed to get transform')
                self._return = 'failed'
                return 'failed'
            
            # Still waiting
            return None

    def on_enter(self, userdata):
        """Called when state becomes active."""
        self._return = None
        
        # Initialize TF2
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, Move00ReadEndEffectorPoseState._node)
        
        self._start_time = Move00ReadEndEffectorPoseState._node.get_clock().now()
        
        Logger.loginfo('Reading end-effector pose from TF...')

    def on_exit(self, userdata):
        """Called when leaving the state."""
        Logger.loginfo('ReadEndEffectorPose state exiting')

    def on_start(self):
        """Called when behavior starts."""
        Logger.loginfo('ReadEndEffectorPose state started')

    def on_stop(self):
        """Called when behavior stops."""
        Logger.loginfo('ReadEndEffectorPose state stopped')