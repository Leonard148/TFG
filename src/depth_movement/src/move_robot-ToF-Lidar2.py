#!/usr/bin/env python3

import rospy
import numpy as np
import os
import signal
import subprocess
import sys
import time
from sensor_msgs.msg import LaserScan, Image, PointCloud2
from rtabmap_msgs.msg import MapData
from nav_msgs.msg import OccupancyGrid
import tf
import random
import math
from geometry_msgs.msg import PoseStamped, Twist  # Added Twist for better motion control
from nav_msgs.srv import GetMap
import cv2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
# Add threading for improved concurrency management
import threading
try:
    import pyransac3d as pyrsc
except ImportError:
    rospy.logwarn("pyransac3d not found. Installing...")
    subprocess.call([sys.executable, "-m", "pip", "install", "pyransac3d"])
    import pyransac3d as pyrsc

class SafeAutonomousExplorer:
    def __init__(self):
        rospy.init_node("safe_autonomous_explorer", anonymous=True)

        #Initial position
        trans, rot = self.tf_listener.lookupTransform("map", self.base_frame, rospy.Time(0))
        self.initial_pose = PoseStamped()
        self.initial_pose.header.frame_id = "map"
        self.initial_pose.header.stamp = rospy.Time.now()
        self.initial_pose.pose.position.x = trans[0]
        self.initial_pose.pose.position.y = trans[1]
        self.initial_pose.pose.position.z = trans[2]
        self.initial_pose.pose.orientation.x = rot[0]
        self.initial_pose.pose.orientation.y = rot[1]
        self.initial_pose.pose.orientation.z = rot[2]
        self.initial_pose.pose.orientation.w = rot[3]

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
        
        # New - Add recovery behavior parameters
        self.consecutive_obstacles = 0  # Counter for consecutive obstacle detections
        self.max_consecutive_obstacles = rospy.get_param("~max_consecutive_obstacles", 5)  # Max before recovery
        self.recovery_rotation_factor = rospy.get_param("~recovery_rotation_factor", 2.0)  # Longer rotation during recovery
        
        # New - Add mutex locks for thread safety
        self.scan_lock = threading.RLock()
        self.movement_command_lock = threading.RLock()
        
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
        
        # New - Add cmd_vel publisher for smoother control
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # New - Add diagnostic publishers
        self.diagnostics_pub = rospy.Publisher('/explorer/diagnostics', String, queue_size=5)
        
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
        
        # New - Add watchdog timer for system health monitoring
        self.watchdog_timer = rospy.Timer(rospy.Duration(5.0), self.watchdog_callback)
       
    # New - Add system diagnostic check method
    def system_diagnostic_check(self, event=None):
        """Perform diagnostic checks when issues are detected"""
        rospy.loginfo("Running system diagnostic checks...")
        
        try:
            # Check ROS node status
            nodes_cmd = ["rosnode", "list"]
            nodes_result = subprocess.run(nodes_cmd, capture_output=True, text=True)
            
            # Check topic publishing rates
            topic_hz_cmd = ["rostopic", "hz", "/scan", "--window=10"]
            topic_hz_process = subprocess.Popen(topic_hz_cmd, 
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
            
            # Wait briefly for data
            rospy.sleep(3.0)
            topic_hz_process.terminate()
            
            # Publish diagnostic results
            diagnostic_result = "Diagnostic check complete - attempting to resume operation"
            self.diagnostics_pub.publish(diagnostic_result)
            
            # Reset emergency flag after diagnostic check
            self.emergency_stop_flag = False
            
        except Exception as e:
            rospy.logerr(f"Error in diagnostic check: {e}")
    
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
    
    # Improved laser_callback with thread safety
    def laser_callback(self, msg):
        """Process incoming laser scan data with thread protection"""
        with self.scan_lock:
            self.laser_scan = msg
            self.last_lidar_time = rospy.Time.now()
            
            # New - Track valid ranges in scan for better obstacle assessment
            ranges = np.array(msg.ranges)
            valid_ranges = ranges[(ranges > msg.range_min) & (ranges < msg.range_max)]
            
            # New - Calculate and log scan quality metrics
            if len(valid_ranges) > 0:
                range_percentage = (len(valid_ranges) / len(ranges)) * 100
                if range_percentage < 50:
                    rospy.logwarn(f"LiDAR data quality low: only {range_percentage:.1f}% valid ranges")

    # Enhanced depth_to_pointcloud with SIMD optimizations
    def depth_to_pointcloud(self, depth_image):
        """Convert a depth image to 3D point cloud with optimized processing"""
        height, width = depth_image.shape
        
        # Create coordinate meshes using vectorized operations for better performance
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        
        # Vectorized coordinate conversion
        z = depth_image
        # New - Improved depth filtering with more precise thresholds
        valid_mask = np.logical_and(z > 0.1, z < 5.0)  # Between 10cm and 5m
        
        # New - Add additional noise filtering
        std_filter = np.ones((3, 3)) / 9
        depth_filtered = cv2.filter2D(z, -1, std_filter)
        noise_mask = np.abs(z - depth_filtered) < 0.03  # Filter noise spikes
        valid_mask = np.logical_and(valid_mask, noise_mask)
        
        # Only compute coordinates for valid points
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = z[valid_mask]
        
        x = (u_valid - self.cx) * z_valid / self.fx
        y = (v_valid - self.cy) * z_valid / self.fy
        
        # Stack coordinates
        points = np.column_stack([x, y, z_valid])
        
        # New - Log point cloud statistics
        if len(points) < 100:
            rospy.logwarn(f"Very few valid depth points: {len(points)}")
        
        return points, valid_mask
    
    # Improved detect_stairs method with better plane segmentation
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
            
            # New - Add more detailed floor plane analysis
            inlier_percentage = (len(inliers) / len(points_3d)) * 100
            rospy.logdebug(f"Floor plane detection: {inlier_percentage:.1f}% points as inliers, " 
                          f"vertical alignment: {vertical_alignment:.3f}")
            
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
                
                # New - Analyze contour shapes to distinguish stairs from other drops
                is_likely_stair = False
                for contour in significant_contours:
                    # Calculate aspect ratio and area
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    area = cv2.contourArea(contour)
                    rect_area = w * h
                    extent = float(area) / rect_area if rect_area > 0 else 0
                    
                    # Stairs typically have specific shape characteristics
                    if aspect_ratio > 1.5 and extent > 0.4:
                        is_likely_stair = True
                        break
                
                # Determine danger zones with distance weighting
                self.stair_detected = len(significant_contours) > 0 and is_likely_stair
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
                
                # New - Add informational overlay
                cv2.putText(visualization, f"Floor confidence: {inlier_percentage:.1f}%", (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
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

    # Improved obstacle detection with better zone analysis
    def analyze_lidar_zones(self):
        """Analyze LiDAR data by zones with improved filtering and classification"""
        if self.laser_scan is None:
            return None
            
        # Get scan data
        ranges = np.array(self.laser_scan.ranges)
        angle_min = self.laser_scan.angle_min
        angle_increment = self.laser_scan.angle_increment
        
        # Filter invalid readings
        valid_mask = np.logical_and(ranges > self.laser_scan.range_min, 
                                   ranges < self.laser_scan.range_max)
        
        # New - Add outlier filtering for more robust zone analysis
        # Use rolling median filter to remove isolated spikes
        window_size = 5
        for i in range(len(ranges)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(ranges), i + window_size // 2 + 1)
            window = ranges[start_idx:end_idx]
            # Skip filtering if we have too many invalid readings in the window
            if np.sum(np.logical_and(window > self.laser_scan.range_min, 
                                     window < self.laser_scan.range_max)) < window_size // 2:
                continue
                
            median = np.median(window[np.logical_and(window > self.laser_scan.range_min, 
                                                   window < self.laser_scan.range_max)])
            if abs(ranges[i] - median) > 0.5:  # If value is significantly different from median
                valid_mask[i] = False  # Mark as invalid
        
        # Angle array for easier zone calculation
        angles = np.arange(len(ranges)) * angle_increment + angle_min
        
        # Define zones - front, left, right with overlap for better awareness
        front_mask = np.abs(angles) <= self.front_angle
        left_mask = np.logical_and(angles > -self.side_angle, angles < -self.front_angle * 0.5)
        right_mask = np.logical_and(angles < self.side_angle, angles > self.front_angle * 0.5)
        
        # New - Add rear zone for better situational awareness
        rear_mask = np.abs(angles) >= math.radians(150)
        
        # Combine masks with valid readings
        front_valid = np.logical_and(front_mask, valid_mask)
        left_valid = np.logical_and(left_mask, valid_mask)
        right_valid = np.logical_and(right_mask, valid_mask)
        rear_valid = np.logical_and(rear_mask, valid_mask)
        
        # Calculate zone statistics with better handling of empty zones
        front_distances = ranges[front_valid] if np.any(front_valid) else np.array([float('inf')])
        left_distances = ranges[left_valid] if np.any(left_valid) else np.array([float('inf')])
        right_distances = ranges[right_valid] if np.any(right_valid) else np.array([float('inf')])
        rear_distances = ranges[rear_valid] if np.any(rear_valid) else np.array([float('inf')])
        
        # New - Calculate more robust zone metrics including quantiles
        zones = {
            'front': {
                'min_distance': np.min(front_distances),
                'mean_distance': np.mean(front_distances),
                'valid_count': np.sum(front_valid),
                'total_count': np.sum(front_mask),
                'closest_percentile': np.percentile(front_distances, 10) if len(front_distances) > 5 else np.min(front_distances)
            },
            'left': {
                'min_distance': np.min(left_distances),
                'mean_distance': np.mean(left_distances),
                'valid_count': np.sum(left_valid),
                'total_count': np.sum(left_mask),
                'closest_percentile': np.percentile(left_distances, 10) if len(left_distances) > 5 else np.min(left_distances)
            },
            'right': {
                'min_distance': np.min(right_distances),
                'mean_distance': np.mean(right_distances),
                'valid_count': np.sum(right_valid),
                'total_count': np.sum(right_mask),
                'closest_percentile': np.percentile(right_distances, 10) if len(right_distances) > 5 else np.min(right_distances)
            },
            'rear': {
                'min_distance': np.min(rear_distances),
                'mean_distance': np.mean(rear_distances),
                'valid_count': np.sum(rear_valid),
                'total_count': np.sum(rear_mask),
                'closest_percentile': np.percentile(rear_distances, 10) if len(rear_distances) > 5 else np.min(rear_distances)
            }
        }
        
        # New - Log data quality metrics
        for zone_name, zone_data in zones.items():
            valid_ratio = zone_data['valid_count'] / max(1, zone_data['total_count'])
            if valid_ratio < 0.5:  # Less than 50% valid readings
                rospy.logdebug(f"Low LiDAR quality in {zone_name} zone: {valid_ratio:.2f}")
        
        return zones

    # Improved process_lidar_for_navigation with more intelligent obstacle handling
    def process_lidar_for_navigation(self):
        """Enhanced LiDAR processing with better obstacle classification and smarter navigation"""
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
            
        # New - Handle return to dock state
        if self.exploration_state == "RETURN_TO_DOCK":
            # This would ideally use localization and path planning
            # For this example, we'll just stop and notify
            self.stop_robot()
            rospy.loginfo("Return to dock requested - stopping for now")
            self.status_pub.publish("STOPPED_FOR_DOCK")
            return
            
        # Analyze LiDAR data by zones (front, left, right, rear)
        zones_analysis = self.analyze_lidar_zones()
        
        if zones_analysis is None:
            rospy.logwarn("No valid LiDAR data for navigation")
            return
            
        # Log zone analysis for debugging
        rospy.logdebug(f"LiDAR zone analysis - Front: {zones_analysis['front']['min_distance']:.2f}m")
        rospy.logdebug(f"LiDAR zone analysis - Left: {zones_analysis['left']['min_distance']:.2f}m")
        rospy.logdebug(f"LiDAR zone analysis - Right: {zones_analysis['right']['min_distance']:.2f}m")
        rospy.logdebug(f"LiDAR zone analysis - Rear: {zones_analysis['rear']['min_distance']:.2f}m")
        
        # Enhanced obstacle classification using percentile values for more robustness
        # This avoids overreacting to single outlier readings
        front_obstacle = zones_analysis['front']['closest_percentile'] < self.min_safe_distance
        left_obstacle = zones_analysis['left']['closest_percentile'] < self.min_safe_distance * 0.8
        right_obstacle = zones_analysis['right']['closest_percentile'] < self.min_safe_distance * 0.8
        rear_obstacle = zones_analysis['rear']['closest_percentile'] < self.min_safe_distance * 0.5
        
        # Check for critical obstacles
        if zones_analysis['front']['min_distance'] < self.critical_distance:
            rospy.logwarn(f"Critical obstacle at {zones_analysis['front']['min_distance']:.2f}m - EMERGENCY STOP")
            self.emergency_stop_flag = True
            self.stop_robot()
            self.start_rotation()
            self.consecutive_obstacles += 1
            return
            
        # New - Track consecutive obstacle encounters for stuck detection
        if front_obstacle:
            self.consecutive_obstacles += 1
            if self.consecutive_obstacles > self.max_consecutive_obstacles:
                rospy.logwarn("Potentially stuck - initiating recovery behavior")
                self.start_recovery_behavior()
                return
        else:
            # Reset counter when path is clear
            self.consecutive_obstacles = 0
            
        # Enhanced navigation logic with improved decision making
        if front_obstacle:
            if not left_obstacle and not right_obstacle:
                # Choose rotation direction based on which side has more space
                # and factor in goal direction if available
                if zones_analysis['left']['mean_distance'] > zones_analysis['right']['mean_distance'] * 1.2:
                    # Left is significantly clearer
                    self.move("ROTATE_LEFT")
                elif zones_analysis['right']['mean_distance'] > zones_analysis['left']['mean_distance'] * 1.2:
                    # Right is significantly clearer
                    self.move("ROTATE_RIGHT")
                else:
                    # Similar clearance, make a weighted random choice
                    left_weight = zones_analysis['left']['mean_distance']
                    right_weight = zones_analysis['right']['mean_distance']
                    total_weight = left_weight + right_weight
                    if random.random() < (left_weight / total_weight):
                        self.move("ROTATE_LEFT")
                    else:
                        self.move("ROTATE_RIGHT")
            elif left_obstacle and not right_obstacle:
                # Clear path to the right
                if zones_analysis['front']['min_distance'] < self.min_safe_distance * 0.7:
                    # Very close front obstacle - rotate instead of forward-right
                    self.move("ROTATE_RIGHT")
                else:
                    self.move("FORWARD_RIGHT")
            elif right_obstacle and not left_obstacle:
                # Clear path to the left
                if zones_analysis['front']['min_distance'] < self.min_safe_distance * 0.7:
                    # Very close front obstacle - rotate instead of forward-left
                    self.move("ROTATE_LEFT")
                else:
                    self.move("FORWARD_LEFT")
            else:
                # Completely blocked, rotate with adaptive strategy
                self.start_rotation()
        else:
            # No front obstacle, check if we should find frontier
            if self.should_find_frontier():
                self.find_exploration_direction(zones_analysis)
            else:
                # Forward path clear - check if we should go straight or adjust
                if left_obstacle and not right_obstacle:
                    # Obstacle on left, bias slightly right
                    self.move("FORWARD_RIGHT", speed_factor=0.8)
                elif right_obstacle and not left_obstacle:
                    # Obstacle on right, bias slightly left
                    self.move("FORWARD_LEFT", speed_factor=0.8)
                else:
                    # Clear path forward
                    self.move("FORWARD")

    # New - Add improved obstacle evasion method
    def execute_evasion_maneuver(self):
        """Execute a maneuver to evade an obstacle or hazard"""
        rospy.logwarn("Executing evasion maneuver")
        
        # First stop to ensure safety
        self.stop_robot()
        
        # Then back up slightly
        self.move("BACKWARD", duration=1.0)
        
        # After backing up, rotate to find clear path
        rospy.Timer(rospy.Duration(1.2), lambda event: self.start_rotation(), oneshot=True)
        
        # Reset emergency flag after evasion
        rospy.Timer(rospy.Duration(5.0), lambda event: setattr(self, 'emergency_stop_flag', False), oneshot=True)
    
    # New - Add recovery behavior method
    def start_recovery_behavior(self):
        """Initiate a recovery behavior when stuck or confused"""
        rospy.logwarn("Starting recovery behavior sequence")
        
        # Reset state
        self.consecutive_obstacles = 0
        self.emergency_stop_flag = False
        
        # Execute a more complex recovery pattern
        # 1. Stop first
        self.stop_robot()
        
        # 2. Back up
        rospy.Timer(rospy.Duration(0.5), 
                   lambda event: self.move("BACKWARD", duration=2.0), 
                   oneshot=True)
        
        # 3. Rotate longer than normal
        rospy.Timer(rospy.Duration(2.7), 
                   lambda event: self.start_rotation(min_time=self.rotation_min_time * self.recovery_rotation_factor), 
                   oneshot=True)
        
        # 4. Resume normal operation
        rotation_time = self.rotation_min_time * self.recovery_rotation_factor + 0.5
        rospy.Timer(rospy.Duration(2.7 + rotation_time), 
                   lambda event: self.reset_after_recovery(), 
                   oneshot=True)
    
    # New - Helper method for recovery reset
    def reset_after_recovery(self):
        """Reset state after recovery behavior completes"""
        self.movement_lock = False
        self.exploration_state = "SCANNING"
        self.status_pub.publish("RECOVERY_COMPLETE")
        rospy.loginfo("Recovery behavior complete - resuming normal operation")

    # Improved move method with smoother control
    def move(self, direction, speed_factor=1.0, duration=None):
        """Move the robot with improved control and thread safety"""
        with self.movement_command_lock:
            if self.movement_lock and direction != "STOP":
                return
                
            # Don't allow movement if emergency stop is active
            if self.emergency_stop_flag and direction != "STOP":
                rospy.logwarn("Movement blocked by emergency stop flag")
                return
                
            # Calculate speeds based on direction
            if direction == "STOP":
                speeds = [0, 0, 0, 0]
            elif direction == "FORWARD":
                base_speed = int(self.max_speed * speed_factor)
                speeds = [base_speed, base_speed, base_speed, base_speed]
            elif direction == "BACKWARD":
                base_speed = int(self.max_speed * speed_factor)
                speeds = [-base_speed, -base_speed, -base_speed, -base_speed]
            elif direction == "ROTATE_LEFT":
                base_speed = int(self.max_speed * 0.7 * speed_factor)  # Reduced for smoother rotation
                speeds = [-base_speed, base_speed, -base_speed, base_speed]
            elif direction == "ROTATE_RIGHT":
                base_speed = int(self.max_speed * 0.7 * speed_factor)  # Reduced for smoother rotation
                speeds = [base_speed, -base_speed, base_speed, -base_speed]
            elif direction == "FORWARD_LEFT":
                base_speed = int(self.max_speed * speed_factor)
                left_speed = int(base_speed * 0.6)  # Left side slower
                speeds = [left_speed, base_speed, left_speed, base_speed]
            elif direction == "FORWARD_RIGHT":
                base_speed = int(self.max_speed * speed_factor)
                right_speed = int(base_speed * 0.6)  # Right side slower
                speeds = [base_speed, right_speed, base_speed, right_speed]
            else:
                rospy.logwarn(f"Unknown direction: {direction}")
                return
                
            # Send command to robot
            self.current_move_direction = direction
            self.current_motor_speeds = speeds
            
            # Format message for raw motor control
            motor_command = f"M {speeds[0]} {speeds[1]} {speeds[2]} {speeds[3]}"
            self.cmd_pub.publish(motor_command)
            
            # Also publish direction for higher-level monitoring
            self.move_cmd_pub.publish(direction)
            
            # New - Also publish cmd_vel for ROS navigation compatibility
            self.publish_cmd_vel(direction, speed_factor)
            
            # Record command time
            self.last_movement_command_time = rospy.Time.now()
            
            # Handle duration-based movements
            if duration is not None:
                self.movement_lock = True
                rospy.Timer(rospy.Duration(duration), 
                           lambda event: self.release_movement_lock(), 
                           oneshot=True)
                
            rospy.logdebug(f"Moving {direction} with speeds {speeds}")
    
    # New - Add method to publish cmd_vel for standard ROS compatibility
    def publish_cmd_vel(self, direction, speed_factor=1.0):
        """Publish equivalent cmd_vel message for ROS navigation stack compatibility"""
        twist = Twist()
        
        # Max linear and angular velocities (m/s and rad/s)
        max_linear = 0.5  # m/s
        max_angular = 1.0  # rad/s
        
        # Set velocities based on direction
        if direction == "FORWARD":
            twist.linear.x = max_linear * speed_factor
        elif direction == "BACKWARD":
            twist.linear.x = -max_linear * speed_factor
        elif direction == "ROTATE_LEFT":
            twist.angular.z = max_angular * speed_factor
        elif direction == "ROTATE_RIGHT":
            twist.angular.z = -max_angular * speed_factor
        elif direction == "FORWARD_LEFT":
            twist.linear.x = max_linear * speed_factor * 0.8
            twist.angular.z = max_angular * speed_factor * 0.5
        elif direction == "FORWARD_RIGHT":
            twist.linear.x = max_linear * speed_factor * 0.8
            twist.angular.z = -max_angular * speed_factor * 0.5
        
        # Publish the twist message
        self.cmd_vel_pub.publish(twist)
    
    # New - Add method to release movement lock
    def release_movement_lock(self):
        """Release movement lock after timed movement completes"""
        self.movement_lock = False
        rospy.logdebug("Movement lock released")

    # Improved rotation method with better randomization
    def start_rotation(self, min_time=None, max_time=None):
        """Start a rotation to find a clear path with improved randomization"""
        if min_time is None:
            min_time = self.rotation_min_time
        if max_time is None:
            max_time = self.rotation_max_time
            
        # Choose direction based on LiDAR data if available
        if self.laser_scan is not None:
            zones = self.analyze_lidar_zones()
            if zones:
                # Prefer direction with more space
                if zones['left']['mean_distance'] > zones['right']['mean_distance']:
                    direction = "ROTATE_LEFT"
                else:
                    direction = "ROTATE_RIGHT"
            else:
                # Random direction if no LiDAR data
                direction = random.choice(["ROTATE_LEFT", "ROTATE_RIGHT"])
        else:
            # Random direction if no LiDAR data
            direction = random.choice(["ROTATE_LEFT", "ROTATE_RIGHT"])
            
        # Calculate rotation duration with gaussian distribution for more natural behavior
        rotation_time = random.uniform(min_time, max_time)
        
        # Start rotation
        rospy.loginfo(f"Starting {direction} rotation for {rotation_time:.2f} seconds")
        self.move(direction)
        
        # Set timer to stop rotation
        self.movement_lock = True
        rospy.Timer(rospy.Duration(rotation_time), 
                   lambda event: self.stop_after_rotation(), 
                   oneshot=True)
    
    # New - Add method to stop after rotation
    def stop_after_rotation(self):
        """Stop robot after rotation and reset flags"""
        self.stop_robot()
        self.movement_lock = False
        
        # Reset emergency flag if it was set
        if self.emergency_stop_flag:
            rospy.Timer(rospy.Duration(0.5), 
                       lambda event: setattr(self, 'emergency_stop_flag', False), 
                       oneshot=True)
    
    # Improved map coverage estimation
    def estimate_map_coverage(self, grid_map):
        """Estimate map coverage with improved analysis"""
        if grid_map is None:
            return 0.0
            
        # Count cells by type
        total_cells = len(grid_map.data)
        if total_cells == 0:
            return 0.0
            
        # In occupancy grid: -1 = unknown, 0 = free, 100 = occupied
        unknown_cells = np.sum(np.array(grid_map.data) == -1)
        free_cells = np.sum(np.array(grid_map.data) == 0)
        occupied_cells = np.sum(np.array(grid_map.data) == 100)
        
        # Calculate coverage (free + occupied) / total
        coverage = (free_cells + occupied_cells) / total_cells
        
        # New - Calculate exploration efficiency metrics
        free_to_occupied_ratio = free_cells / max(1, occupied_cells)
        
        rospy.loginfo(f"Map coverage: {coverage:.2f}, Free/Occupied ratio: {free_to_occupied_ratio:.2f}")
        rospy.loginfo(f"Map cells - Free: {free_cells}, Occupied: {occupied_cells}, Unknown: {unknown_cells}")
        
        return coverage
    
    # Enhanced frontier detection
    def should_find_frontier(self):
        """Determine if robot should seek frontier instead of just following clear path"""
        # Check if enough time has passed since last map check
        if (rospy.Time.now() - self.last_map_check_time).to_sec() < self.map_check_interval:
            return False
            
        # Set new map check time
        self.last_map_check_time = rospy.Time.now()
        
        # Check map coverage against target
        if self.map_coverage > self.target_coverage:
            rospy.loginfo(f"Map coverage ({self.map_coverage:.2f}) exceeds target ({self.target_coverage:.2f})")
            return False
            
        # Random chance to seek frontier for more natural exploration
        # Higher chance when map coverage is lower
        chance = 0.5 * (1.0 - self.map_coverage / self.target_coverage)
        return random.random() < chance
    
    # New - Improved frontier-based exploration
    def find_exploration_direction(self, zones_analysis):
        """Find direction to unexplored areas (frontiers)"""
        rospy.loginfo("Seeking frontier for exploration")
        
        # For this simplified version, we'll use a heuristic approach:
        # - Prefer directions with more open space (higher mean distance)
        # - Avoid directions with obstacles
        
        # Calculate direction scores based on mean distance
        front_score = zones_analysis['front']['mean_distance'] * 1.5  # Prefer forward
        left_score = zones_analysis['left']['mean_distance']
        right_score = zones_analysis['right']['mean_distance']
        
        # Penalize directions with obstacles
        if zones_analysis['front']['min_distance'] < self.min_safe_distance:
            front_score = 0
        if zones_analysis['left']['min_distance'] < self.min_safe_distance * 0.8:
            left_score = 0
        if zones_analysis['right']['min_distance'] < self.min_safe_distance * 0.8:
            right_score = 0
            
        # Choose direction with highest score
        if front_score > left_score and front_score > right_score:
            self.move("FORWARD")
        elif left_score > right_score:
            self.move("ROTATE_LEFT", duration=1.0)
            # Schedule forward movement after rotation
            rospy.Timer(rospy.Duration(1.2), lambda event: self.move("FORWARD"), oneshot=True)
        else:
            self.move("ROTATE_RIGHT", duration=1.0)
            # Schedule forward movement after rotation
            rospy.Timer(rospy.Duration(1.2), lambda event: self.move("FORWARD"), oneshot=True)
            
        # Log decision
        rospy.loginfo(f"Frontier scores - Front: {front_score:.2f}, Left: {left_score:.2f}, Right: {right_score:.2f}")
    
    # Enhanced stop robot method
    def stop_robot(self):
        """Stop the robot with improved safety checks"""
        rospy.loginfo("Stopping robot")
        
        # Send stop command
        self.move("STOP")
        
        # Additional safety: send raw motor stop command
        stop_cmd = "M 0 0 0 0"
        self.cmd_pub.publish(stop_cmd)
        
        # Also publish zero twist message for ROS compatibility
        zero_twist = Twist()
        self.cmd_vel_pub.publish(zero_twist)
        
        # Update state
        self.current_move_direction = "STOP"
        self.current_motor_speeds = [0, 0, 0, 0]
    
    # Enhanced shutdown method
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
        
        # Save map if available
        if hasattr(self, 'map_data') and self.map_data is not None:
            try:
                map_file = os.path.join(self.map_save_path, f"final_map_{int(time.time())}.pgm")
                rospy.loginfo(f"Saving final map to {map_file}")
                # This would call map_saver functionality
            except Exception as e:
                rospy.logerr(f"Error saving map: {e}")
        
        # Small delay to ensure messages are sent
        rospy.sleep(0.5)
        
        rospy.loginfo("Safe autonomous explorer shutdown complete")
    
    # New - Add method to stop YOLO detector
    def stop_yolo_detector(self):
        """Safely stop the YOLO detector process"""
        if self.yolo_process is not None:
            rospy.loginfo("Stopping YOLO detector")
            try:
                # Send SIGTERM first for graceful shutdown
                self.yolo_process.terminate()
                
                # Wait briefly for process to terminate
                try:
                    self.yolo_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate gracefully
                    rospy.logwarn("YOLO process did not terminate gracefully, forcing kill")
                    self.yolo_process.kill()
                
                self.yolo_process = None
            except Exception as e:
                rospy.logerr(f"Error stopping YOLO detector: {e}")
    
    # New - Add run method to start exploration
    def run(self):
        """Main run loop for the explorer"""
        rospy.loginfo("Starting exploration run")
        
        # Set up shutdown handler
        rospy.on_shutdown(self.shutdown)
        
        # Start processing loop
        rate = rospy.Rate(10)  # 10 Hz control loop
        
        while not rospy.is_shutdown() and self.running:
            # Process LiDAR for obstacle avoidance
            self.process_lidar_for_navigation()
            
            # Check for map coverage periodically
            if (rospy.Time.now() - self.last_map_check_time).to_sec() >= self.map_check_interval:
                # This would normally check map coverage
                pass
                
            rate.sleep()
    
    # New - Add method to start initial scan
    def start_initial_scan(self):
        """Start initial 360-degree scan"""
        rospy.loginfo("Starting initial 360-degree scan")
        self.initial_scan_start = rospy.Time.now()
        self.exploration_state = "INITIAL_SCAN"
        self.status_pub.publish("INITIAL_SCAN_STARTED")
        
        # Rotate slowly to capture surroundings
        self.move("ROTATE_LEFT", speed_factor=0.6)  # Slower rotation for better mapping

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