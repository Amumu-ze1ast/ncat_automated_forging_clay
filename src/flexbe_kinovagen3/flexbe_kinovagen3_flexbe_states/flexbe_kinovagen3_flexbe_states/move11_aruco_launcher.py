#!/usr/bin/env python3

from flexbe_core import EventState, Logger
from flexbe_core.proxy import ProxySubscriberCached

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import rclpy

# Don't import cv2 at module level - delay until needed


class Move11ArucoDisplayState(EventState):
    """
    FlexBE state for displaying ArUco markers with 3D position in OpenCV window.
    
    Shows live video with detected markers and their positions in both camera frame and base_link.
    Press 'q' to exit.
    
    Parameters:
    -- timeout          float   Timeout in seconds (default: 60.0, use 0.0 for infinite)
    
    Outcomes:
    <= done             User pressed 'q' to exit
    <= timeout          Timeout reached
    <= failed           Detection failed
    """

    def __init__(self, timeout=60.0):
        super(Move11ArucoDisplayState, self).__init__(
            outcomes=['done', 'timeout', 'failed']
        )
        
        # Parameters
        self._timeout = timeout
        
        # CV Bridge
        self._bridge = CvBridge()
        
        # ArUco detector (will be initialized lazily)
        self._aruco_dict = None
        self._aruco_params = None
        self._cv2_initialized = False
        self._cv2 = None
        
        # Topic names
        self._rgb_topic = '/camera/color/image_raw'
        self._depth_topic = '/camera/depth/image_raw'
        self._camera_info_topic = '/camera/color/camera_info'
        
        # Subscribers
        self._rgb_sub = ProxySubscriberCached({self._rgb_topic: Image})
        self._depth_sub = ProxySubscriberCached({self._depth_topic: Image})
        self._camera_info_sub = ProxySubscriberCached({self._camera_info_topic: CameraInfo})
        
        # TF2 for transformations
        self._tf_buffer = None
        self._tf_listener = None
        
        # Camera intrinsics
        self._camera_info = None
        
        # State management
        self._start_time = None
        self._return = None

    def execute(self, userdata):
        """Execute state - called periodically while state is active."""
        if self._return is not None:
            return self._return
        
        # Initialize cv2 on first execute call
        if not self._cv2_initialized:
            if not self._initialize_cv2():
                self._return = 'failed'
                return 'failed'
        
        # Check timeout
        if self._timeout > 0.0:
            elapsed = Move11ArucoDisplayState._node.get_clock().now() - self._start_time
            elapsed_sec = elapsed.nanoseconds / 1e9
            
            if elapsed_sec >= self._timeout:
                Logger.logwarn(f'ArUco display timeout after {elapsed_sec:.1f}s')
                self._cv2.destroyAllWindows()
                self._return = 'timeout'
                return 'timeout'
        
        # Process and display frame
        self._process_frame()
        
        # Check for 'q' key press
        key = self._cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            Logger.loginfo('User pressed q - exiting ArUco display')
            self._cv2.destroyAllWindows()
            self._return = 'done'
            return 'done'
        
        return None

    def on_enter(self, userdata):
        """Called when state becomes active."""
        self._return = None
        self._cv2_initialized = False
        self._camera_info = None
        
        # Initialize TF2
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, Move11ArucoDisplayState._node)
        
        self._start_time = Move11ArucoDisplayState._node.get_clock().now()
        
        Logger.loginfo('ArUco Display started - press q to exit')

    def on_exit(self, userdata):
        """Called when leaving the state."""
        if self._cv2 is not None:
            self._cv2.destroyAllWindows()
        Logger.loginfo('ArUco Display state exiting')

    def on_start(self):
        """Called when behavior starts."""
        Logger.loginfo('ArUco Display state started')

    def on_stop(self):
        """Called when behavior stops."""
        if self._cv2 is not None:
            self._cv2.destroyAllWindows()
        Logger.loginfo('ArUco Display state stopped')

    # ==================== Helper Methods ====================

    def _initialize_cv2(self):
        """Initialize cv2 and ArUco detector."""
        try:
            import cv2
            
            Logger.loginfo(f'OpenCV version: {cv2.__version__}')
            
            # Use legacy API for OpenCV 4.6.0
            self._aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            self._aruco_params = cv2.aruco.DetectorParameters_create()
            
            Logger.loginfo('ArUco detector initialized (LEGACY API for OpenCV 4.6.0)')
            
            self._cv2_initialized = True
            self._cv2 = cv2
            return True
            
        except Exception as e:
            Logger.logerr(f'Failed to initialize ArUco detector: {str(e)}')
            import traceback
            traceback.print_exc()
            return False

    def _load_camera_info(self):
        """Load camera intrinsics from camera_info topic."""
        if self._camera_info is not None:
            return True
            
        if not self._camera_info_sub.has_msg(self._camera_info_topic):
            return False
        
        try:
            self._camera_info = self._camera_info_sub.get_last_msg(self._camera_info_topic)
            Logger.loginfo('Camera info loaded')
            return True
            
        except Exception as e:
            Logger.logerr(f'Error loading camera info: {str(e)}')
            return False

    def _pixel_to_3d(self, pixel_x, pixel_y, depth_meters):
        """Convert pixel coordinates and depth to 3D point in camera frame."""
        if self._camera_info is None:
            return None
        
        # Camera intrinsics
        fx = self._camera_info.k[0]
        fy = self._camera_info.k[4]
        cx = self._camera_info.k[2]
        cy = self._camera_info.k[5]
        
        # Convert to 3D in camera frame
        x = (pixel_x - cx) * depth_meters / fx
        y = (pixel_y - cy) * depth_meters / fy
        z = depth_meters
        
        return (x, y, z)

    def _transform_to_base_link(self, x, y, z):
        """Transform point from camera frame to base_link."""
        try:
            # Create PointStamped in camera frame
            point_camera = PointStamped()
            point_camera.header.frame_id = 'camera_color_frame'
            point_camera.header.stamp = Move11ArucoDisplayState._node.get_clock().now().to_msg()
            point_camera.point.x = x
            point_camera.point.y = y
            point_camera.point.z = z
            
            # Transform to base_link
            transform = self._tf_buffer.lookup_transform(
                'base_link',
                'camera_color_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            point_base = tf2_geometry_msgs.do_transform_point(point_camera, transform)
            
            return (point_base.point.x, point_base.point.y, point_base.point.z)
            
        except Exception as e:
            return None

    def _process_frame(self):
        """Process and display current frame with ArUco detection."""
        # Check if we have all necessary data
        if not self._rgb_sub.has_msg(self._rgb_topic):
            return
        
        if not self._depth_sub.has_msg(self._depth_topic):
            return
        
        # Load camera info if needed
        if self._camera_info is None:
            self._load_camera_info()
            if self._camera_info is None:
                return
        
        try:
            cv2 = self._cv2
            
            # Get RGB image
            rgb_msg = self._rgb_sub.get_last_msg(self._rgb_topic)
            cv_image = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            
            # Get depth image
            depth_msg = self._depth_sub.get_last_msg(self._depth_topic)
            depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            
            # Detect ArUco markers (Legacy API)
            corners, ids, rejected = cv2.aruco.detectMarkers(
                cv_image, 
                self._aruco_dict, 
                parameters=self._aruco_params
            )
            
            # Draw detected markers
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
                
                # Get image dimensions
                rgb_height, rgb_width = cv_image.shape[:2]
                depth_height, depth_width = depth_image.shape[:2]
                
                # Process each detected marker
                for i, marker_id in enumerate(ids):
                    try:
                        # Get center of marker in RGB coordinates
                        center_x_rgb = int(np.mean(corners[i][0][:, 0]))
                        center_y_rgb = int(np.mean(corners[i][0][:, 1]))
                        
                        # Scale to depth image size
                        center_x_depth = int(center_x_rgb * depth_width / rgb_width)
                        center_y_depth = int(center_y_rgb * depth_height / rgb_height)
                        
                        # Get depth - average over region
                        if 0 <= center_y_depth < depth_height and 0 <= center_x_depth < depth_width:
                            # Sample 5x5 region
                            y_start = max(0, center_y_depth - 2)
                            y_end = min(depth_height, center_y_depth + 3)
                            x_start = max(0, center_x_depth - 2)
                            x_end = min(depth_width, center_x_depth + 3)
                            
                            depth_region = depth_image[y_start:y_end, x_start:x_end]
                            valid_depths = depth_region[depth_region > 0]
                            
                            if len(valid_depths) > 0:
                                depth_value = np.mean(valid_depths)
                                depth_meters = float(depth_value) / 1000.0
                                
                                # Convert to 3D in camera frame
                                point_camera = self._pixel_to_3d(center_x_rgb, center_y_rgb, depth_meters)
                                
                                if point_camera:
                                    x_cam, y_cam, z_cam = point_camera
                                    
                                    # Transform to base_link
                                    point_base = self._transform_to_base_link(x_cam, y_cam, z_cam)
                                    
                                    # Display info on image
                                    text1 = f"ID: {marker_id[0]}"
                                    text2 = f"Cam: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})"
                                    
                                    if point_base:
                                        x_base, y_base, z_base = point_base
                                        text3 = f"Base: ({x_base:.3f}, {y_base:.3f}, {z_base:.3f})"
                                    else:
                                        text3 = "Base: TF unavailable"
                                    
                                    # Draw text on image
                                    cv2.putText(cv_image, text1, (center_x_rgb - 80, center_y_rgb - 60),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    cv2.putText(cv_image, text2, (center_x_rgb - 80, center_y_rgb - 40),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                                    cv2.putText(cv_image, text3, (center_x_rgb - 80, center_y_rgb - 20),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            else:
                                cv2.putText(cv_image, f"ID: {marker_id[0]} - No depth", 
                                           (center_x_rgb - 60, center_y_rgb - 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                
                    except Exception as e:
                        Logger.logerr(f'Error processing marker {i}: {e}')
            
            # Add instructions at top of image
            cv2.putText(cv_image, "Press 'q' to exit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display result
            cv2.imshow('ArUco 3D Detection - FlexBE', cv_image)
            
        except Exception as e:
            Logger.logerr(f'Error processing frame: {str(e)}')