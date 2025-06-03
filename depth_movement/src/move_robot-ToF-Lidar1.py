#!/usr/bin/env python3

import rospy
import numpy as np
import os
import signal
import subprocess
import sys
import time
from sensor_msgs.msg import LaserScan, Image, PointCloud2
from std_msgs.msg import String, Bool
from rtabmap_msgs.msg import MapData
from nav_msgs.msg import OccupancyGrid
import tf
import random
import math
from geometry_msgs.msg import PoseStamped
from nav_msgs.srv import GetMap
import cv2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
try:
    import pyransac3d as pyrsc
except ImportError:
    rospy.logwarn("pyransac3d not found. Installing...")
    subprocess.call([sys.executable, "-m", "pip", "install", "pyransac3d"])
    import pyransac3d as pyrsc

class SafeAutonomousExplorer:
    def __init__(self):
        rospy.init_node("safe_autonomous_explorer", anonymous=True)
       
        # Core movement and navigation parameters
        self.max_speed = rospy.get_param("~max_speed", 200)
        self.min_speed = rospy.get_param("~min_speed", 100)
        self.min_safe_distance = rospy.get_param("~min_safe_distance", 1.2)
        self.critical_distance = rospy.get_param("~critical_distance", 0.5)
        self.map_check_interval = rospy.get_param("~map_check_interval", 5)
        self.rotation_min_time = rospy.get_param("~rotation_min_time", 1.5)
        self.rotation_max_time = rospy.get_param("~rotation_max_time", 3.0)
        self.target_coverage = rospy.get_param("~target_coverage", 0.95)
        self.frontier_search_distance = rospy.get_param("~frontier_search_distance", 3.0)
        self.lidar_safety_angle = math.radians(rospy.get_param("~lidar_safety_angle", 30))
        self.scan_timeout = rospy.Duration(rospy.get_param("~scan_timeout", 5.0))
        
        # Stair detector parameters with more conservative defaults
        self.ransac_threshold = rospy.get_param('~ransac_threshold', 0.03)  # Reduced from 0.05 to 3cm for better precision
        self.floor_height_threshold = rospy.get_param('~floor_height_threshold', 0.03)  # Reduced from 0.05 to 3cm
        self.drop_threshold = rospy.get_param('~drop_threshold', 0.12)  # Reduced from 0.15 to 12cm for earlier detection
        self.danger_distance = rospy.get_param('~danger_distance', 1.0)  # Increased from 0.8 to 1.0m for safer operation
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_link')
        self.avoid_stairs_enabled = rospy.get_param("~avoid_stairs", True)
        
        # YOLO parameters with additional validation
        self.yolo_weights = rospy.get_param("~yolo_weights", "/home/jetson/catkin_ws/src/yolov7/runs/exp_custom17/weights/best.pt")
        if not os.path.exists(self.yolo_weights):
            rospy.logerr(f"YOLO weights file not found at: {self.yolo_weights}")
        self.yolo_img_size = rospy.get_param("~yolo_img_size", 640)
        self.yolo_conf_thres = rospy.get_param("~yolo_conf_thres", 0.25)
        self.yolo_camera_source = rospy.get_param("~yolo_camera_source", "0")
        self.yolo_capture_interval = rospy.get_param("~yolo_capture_interval", 2.0)
        self.yolo_process = None
        
        # Camera parameters (adjust according to your ToF camera)
        self.fx = 525.0  # Focal length in x
        self.fy = 525.0  # Focal length in y
        self.cx = 319.5  # Optical center in x
        self.cy = 239.5  # Optical center in y
        
        # LiDAR scan zone configuration with additional safety margin
        self.front_angle = math.radians(rospy.get_param("~front_angle", 25))  # Reduced from 30 to 25 degrees for narrower front zone
        self.side_angle = math.radians(rospy.get_param("~side_angle", 50))  # Reduced from 60 to 50 degrees
        
        # State variables with additional safety flags
        self.running = True
        self.exploration_state = "INITIAL_SCAN"
        self.last_lidar_time = rospy.Time.now()
        self.laser_scan = None
        self.tf_listener = tf.TransformListener()
        self.initial_scan_start = None
        self.map_data = None
        self.map_save_path = os.path.expanduser("~/saved_maps")
        self.last_map_check_time = rospy.Time.now()
        self.map_coverage = 0.0
        self.stair_detected = False
        self.floor_plane_eq = None  # Floor plane equation [a, b, c, d]
        self.danger_zone_detected = False
        self.emergency_stop_flag = False  # New flag for emergency stop conditions
        
        # Movement control state with additional safety checks
        self.current_move_direction = "STOP"
        self.current_motor_speeds = [0, 0, 0, 0]
        self.movement_lock = False
        self.last_movement_command_time = rospy.Time.now()  # Track last command time
        
        # Ensure map directory exists with proper permissions
        try:
            os.makedirs(self.map_save_path, exist_ok=True)
            # Verify we can write to the directory
            test_file = os.path.join(self.map_save_path, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            rospy.logerr(f"Cannot access map save directory: {e}")
            self.map_save_path = os.path.expanduser("~/")  # Fallback to home directory
        
        # Utility instances
        self.bridge = CvBridge()
        self.ransac = pyrsc.Plane()  # Reuse RANSAC instance for efficiency
        
        # Publishers with larger queue sizes for critical topics
        self.cmd_pub = rospy.Publisher("/robot/move/raw", String, queue_size=20)
        self.move_cmd_pub = rospy.Publisher("/robot/move/direction", String, queue_size=20)
        self.danger_pub = rospy.Publisher('/stair_detector/danger', Bool, queue_size=5)
        self.stair_mask_pub = rospy.Publisher('/stair_detector/stair_mask', Image, queue_size=5)
        self.status_pub = rospy.Publisher('/explorer/status', String, queue_size=5)  # New status publisher
        
        # Subscribers with buffered messages
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback, queue_size=1, buff_size=2**24)
        self.map_sub = rospy.Subscriber("/rtabmap/mapData", MapData, self.map_callback, queue_size=1)
        self.grid_map_sub = rospy.Subscriber("/map", OccupancyGrid, self.grid_map_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback, queue_size=1, buff_size=2**24)
        
        # Configure logging with timestamp
        if rospy.get_param("~debug", False):
            rospy.loginfo("Debug mode enabled - verbose logging activated")
            self.log_level = rospy.DEBUG
        else:
            self.log_level = rospy.INFO
        
        rospy.loginfo("Safe autonomous explorer with stair detection and YOLO initialized")
        
        # Initialize YOLO detector with retry logic
        self.start_yolo_detector()
        
        # Start exploration with safety delay
        rospy.Timer(rospy.Duration(1.0), lambda event: self.start_initial_scan(), oneshot=True)
    
    def start_yolo_detector(self, retry_count=3):
        """Start the YOLO detector as a separate process with retry logic"""
        try:
            # Build YOLO command
            yolo_script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "yolov7/detect_timed.py")
            
            # Try to find the script in common locations if specific path doesn't exist
            if not os.path.exists(yolo_script_path):
                possible_paths = [
                    "/home/jetson/catkin_ws/src/yolov7/detect_timed.py",
                    os.path.expanduser("~/yolov7/detect_timed.py"),
                    "./detect_timed.py"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        yolo_script_path = path
                        break
                else:
                    rospy.logerr("Could not find detect_timed.py script. Please specify the full path.")
                    if retry_count > 0:
                        rospy.loginfo(f"Retrying YOLO detector startup ({retry_count} attempts remaining)")
                        rospy.Timer(rospy.Duration(2.0), lambda event: self.start_yolo_detector(retry_count-1), oneshot=True)
                    return
            
            # Prepare YOLO command with additional error handling
            yolo_cmd = [
                "python3", 
                yolo_script_path, 
                "--weights", self.yolo_weights,
                "--img", str(self.yolo_img_size),
                "--conf", str(self.yolo_conf_thres),
                "--source", self.yolo_camera_source,
                "--interval", str(self.yolo_capture_interval),
                "--project", os.path.join(self.map_save_path, "yolo_detections"),
                "--name", "yolo_detections_" + str(int(time.time()))  # Unique run name
            ]
            
            # Start YOLO process with environment preservation
            rospy.loginfo(f"Starting YOLO detector: {' '.join(yolo_cmd)}")
            self.yolo_process = subprocess.Popen(
                yolo_cmd,
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                env=os.environ  # Preserve current environment
            )
            
            # Start a thread to monitor YOLO output with error detection
            import threading
            def monitor_yolo_output():
                while self.yolo_process and self.running:
                    line = self.yolo_process.stdout.readline()
                    if not line and self.yolo_process.poll() is not None:
                        break
                    rospy.loginfo(f"YOLO: {line.strip()}")
                    
                    # Check for common error patterns
                    if "error" in line.lower() or "exception" in line.lower():
                        rospy.logwarn(f"YOLO detector error detected: {line.strip()}")
                        if retry_count > 0:
                            rospy.loginfo(f"Attempting YOLO detector restart ({retry_count} attempts remaining)")
                            self.stop_yolo_detector()
                            rospy.Timer(rospy.Duration(2.0), lambda event: self.start_yolo_detector(retry_count-1), oneshot=True)
                            break
            
            yolo_monitor = threading.Thread(target=monitor_yolo_output)
            yolo_monitor.daemon = True
            yolo_monitor.start()
            
            rospy.loginfo("YOLO detector started successfully")
        except Exception as e:
            rospy.logerr(f"Error starting YOLO detector: {e}")
            if retry_count > 0:
                rospy.loginfo(f"Retrying YOLO detector startup ({retry_count} attempts remaining)")
                rospy.Timer(rospy.Duration(2.0), lambda event: self.start_yolo_detector(retry_count-1), oneshot=True)
            else:
                self.yolo_process = None
    
    # ... [Previous methods remain mostly the same until depth_to_pointcloud] ...

    def depth_to_pointcloud(self, depth_image):
        """Convert a depth image to 3D point cloud with optimized processing"""
        height, width = depth_image.shape
        
        # Create coordinate meshes using vectorized operations for better performance
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        
        # Vectorized coordinate conversion
        z = depth_image
        valid_mask = np.logical_and(z > 0.1, z < 5.0)  # Between 10cm and 5m
        
        # Only compute coordinates for valid points
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = z[valid_mask]
        
        x = (u_valid - self.cx) * z_valid / self.fx
        y = (v_valid - self.cy) * z_valid / self.fy
        
        # Stack coordinates
        points = np.column_stack([x, y, z_valid])
        
        return points, valid_mask
    
    def detect_stairs(self, depth_image, header):
        """Enhanced stair detection with better plane fitting and visualization"""
        try:
            # Convert depth image to point cloud
            points_3d, depth_mask = self.depth_to_pointcloud(depth_image)
            
            if len(points_3d) < 50:  # Increased minimum points for better reliability
                rospy.logwarn("Not enough valid points for stair detection")
                return
            
            # 1. Detect floor plane using RANSAC with optimized parameters
            best_eq, inliers = self.ransac.fit(points_3d, self.ransac_threshold, maxIteration=1000)
            
            # Check if the plane is horizontal with stricter criteria
            a, b, c, d = best_eq
            normal_vector = np.array([a, b, c])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            
            # Check alignment with vertical (assuming camera looks forward)
            vertical_alignment = np.abs(normal_vector[1])  # Y axis points up in camera coords
            
            if vertical_alignment > 0.9:  # More strict horizontal plane requirement (was 0.8)
                self.floor_plane_eq = best_eq
                
                # Create visualization image with more detailed information
                visualization = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
                visualization = cv2.normalize(visualization, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Calculate floor distance image
                height, width = depth_image.shape
                floor_distance = np.zeros_like(depth_image)
                
                # Vectorized floor distance calculation
                u, v = np.meshgrid(np.arange(width), np.arange(height))
                z = depth_image
                valid_depth = np.logical_and(z > 0.1, z < 5.0)
                
                # Calculate expected floor depth for all pixels
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                expected_z = (-a * x - b * y - d) / np.clip(c, 1e-6, None)  # Avoid division by zero
                
                # Calculate depth differences
                depth_diff = z - expected_z
                stair_mask = (depth_diff > self.drop_threshold) & valid_depth
                
                # Process mask to remove noise with adaptive morphology
                stair_mask = stair_mask.astype(np.uint8) * 255
                kernel_size = max(3, int(width/100))  # Adaptive kernel size based on image width
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                stair_mask = cv2.morphologyEx(stair_mask, cv2.MORPH_OPEN, kernel)
                stair_mask = cv2.morphologyEx(stair_mask, cv2.MORPH_CLOSE, kernel)
                
                # Find contours with minimum area requirement
                contours, _ = cv2.findContours(stair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]  # Reduced from 500
                
                # Determine danger zones with distance weighting
                self.stair_detected = len(significant_contours) > 0
                self.danger_zone_detected = False
                
                if self.stair_detected:
                    # Calculate minimum distance to any stair region
                    min_distance = float('inf')
                    for contour in significant_contours:
                        # Get all points in the contour
                        points = contour.squeeze()
                        if points.ndim == 1:
                            points = points[np.newaxis, :]
                        
                        # Get depths for contour points
                        contour_depths = depth_image[points[:, 1], points[:, 0]]
                        valid_depths = contour_depths[contour_depths > 0.1]
                        
                        if len(valid_depths) > 0:
                            contour_min_dist = np.min(valid_depths)
                            if contour_min_dist < min_distance:
                                min_distance = contour_min_dist
                    
                    # Determine danger based on distance and size of stair region
                    if min_distance < self.danger_distance:
                        self.danger_zone_detected = True
                        cv2.putText(visualization, "DANGER: STAIR DETECTED", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif min_distance < self.danger_distance * 1.5:  # Warning zone
                        cv2.putText(visualization, "WARNING: STAIR NEARBY", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Publish visualization with contour information
                cv2.drawContours(visualization, significant_contours, -1, (0, 0, 255), 2)
                self.danger_pub.publish(Bool(self.danger_zone_detected))
                
                # Publish visualization image
                stair_mask_msg = self.bridge.cv2_to_imgmsg(visualization, encoding="bgr8")
                stair_mask_msg.header = header
                self.stair_mask_pub.publish(stair_mask_msg)
                
                # Log detection status
                if self.danger_zone_detected:
                    rospy.logwarn(f"DANGER! Stair detected at {min_distance:.2f}m")
                elif self.stair_detected:
                    rospy.loginfo(f"Stair detected at safe distance: {min_distance:.2f}m")
            
        except Exception as e:
            rospy.logerr(f"Error in stair detection: {e}")
            import traceback
            traceback.print_exc()

    # ... [Previous methods remain until process_lidar_for_navigation] ...

    def process_lidar_for_navigation(self):
        """Enhanced LiDAR processing with better obstacle classification"""
        if self.laser_scan is None or self.movement_lock:
            return
            
        # Check for emergency stop conditions
        if self.emergency_stop_flag:
            self.stop_robot()
            return
            
        # Check if LiDAR data is too old
        if (rospy.Time.now() - self.last_lidar_time) > self.scan_timeout:
            rospy.logwarn("LiDAR data timeout - stopping robot and initiating recovery")
            self.stop_robot()
            self.start_recovery_behavior()
            return
            
        # First check for stair danger with additional conditions
        if self.danger_zone_detected and self.avoid_stairs_enabled:
            # Don't allow forward movement if there's a drop hazard
            rospy.logwarn("Avoiding forward movement - stair detected in danger zone")
            if self.current_move_direction in ["FORWARD", "FORWARD_LEFT", "FORWARD_RIGHT"]:
                self.execute_evasion_maneuver()
            return
            
        # Handle initial scan state
        if self.exploration_state == "INITIAL_SCAN":
            if (rospy.Time.now() - self.initial_scan_start) > rospy.Duration(10):
                self.stop_robot()
                self.exploration_state = "SCANNING"
                rospy.loginfo("Initial scan complete, beginning exploration")
                self.status_pub.publish("INITIAL_SCAN_COMPLETE")
            return
            
        # Analyze LiDAR data by zones (front, left, right)
        zones_analysis = self.analyze_lidar_zones()
        
        # Log zone analysis for debugging
        rospy.logdebug(f"LiDAR zone analysis - Front: {zones_analysis['front']}")
        rospy.logdebug(f"LiDAR zone analysis - Left: {zones_analysis['left']}")
        rospy.logdebug(f"LiDAR zone analysis - Right: {zones_analysis['right']}")
        
        # Enhanced obstacle classification
        front_obstacle = zones_analysis['front']['min_distance'] < self.min_safe_distance
        left_obstacle = zones_analysis['left']['min_distance'] < self.min_safe_distance * 0.8  # More sensitive on sides
        right_obstacle = zones_analysis['right']['min_distance'] < self.min_safe_distance * 0.8
        
        # Check for critical obstacles
        if zones_analysis['front']['min_distance'] < self.critical_distance:
            rospy.logwarn(f"Critical obstacle at {zones_analysis['front']['min_distance']:.2f}m - EMERGENCY STOP")
            self.emergency_stop_flag = True
            self.stop_robot()
            self.start_rotation()
            return
            
        # Enhanced navigation logic
        if front_obstacle:
            if not left_obstacle and not right_obstacle:
                # Choose rotation direction based on which side has more space
                if zones_analysis['left']['min_distance'] > zones_analysis['right']['min_distance']:
                    self.move("ROTATE_LEFT")
                else:
                    self.move("ROTATE_RIGHT")
            elif left_obstacle and not right_obstacle:
                self.move("FORWARD_RIGHT")
            elif right_obstacle and not left_obstacle:
                self.move("FORWARD_LEFT")
            else:
                # Completely blocked, rotate randomly
                self.start_rotation()
        else:
            # No front obstacle, check if we should find frontier
            if self.should_find_frontier():
                self.find_exploration_direction(zones_analysis)
            else:
                self.move("FORWARD")

    # ... [Rest of the methods remain with similar improvements] ...

    def shutdown(self):
        """Enhanced shutdown procedure with more thorough cleanup"""
        rospy.loginfo("Initiating comprehensive shutdown...")
        self.running = False
        self.emergency_stop_flag = True
        
        # Stop robot with higher priority
        self.stop_robot()
        
        # Stop YOLO detector with timeout
        self.stop_yolo_detector()
        
        # Publish shutdown status
        self.status_pub.publish("SHUTDOWN")
        
        # Small delay to ensure messages are sent
        rospy.sleep(0.5)
        
        rospy.loginfo("Safe autonomous explorer shutdown complete")

if __name__ == "__main__":
    try:
        explorer = SafeAutonomousExplorer()
        explorer.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)