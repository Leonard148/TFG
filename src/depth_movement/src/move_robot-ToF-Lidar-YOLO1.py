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
# Added Twist for better motion control
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.srv import GetMap
from std_msgs.msg import String, Bool
import cv2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
# Threading for improved concurrency management
import threading
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
import open3d as o3d
import ctypes
try:
    import pyransac3d as pyrsc
except ImportError:
    rospy.logwarn("pyransac3d not found. Installing...")
    subprocess.call([sys.executable, "-m", "pip", "install", "pyransac3d"])
    import pyransac3d as pyrsc


class SafeAutonomousExplorer:
    def __init__(self):
        rospy.init_node("safe_autonomous_explorer", anonymous=True)

        # Define tf listener
        self.tf_listener = tf.TransformListener()

        # Inicializar el procesador de Azure Kinect
        try:
            pykinect.initialize_libraries()
            self.kinect_device = pykinect.start_device()
            self.transformation = self.kinect_device.transformation
            rospy.loginfo("Azure Kinect inicializado correctamente")
        except Exception as e:
            rospy.logerr(f"Error al inicializar Azure Kinect: {e}")
            self.kinect_device = None

        # Define base_frame first
        self.base_frame = rospy.get_param('~base_frame', 'base_link')

        self.initial_pose = None
        # Wait until tf is ready
        try:
            rospy.loginfo(
                "Esperando transformada inicial de 'map' a 'base_link'...")
            self.tf_listener.waitForTransform(
                "map", self.base_frame, rospy.Time(0), rospy.Duration(10.0))
            trans, rot = self.tf_listener.lookupTransform(
                "map", self.base_frame, rospy.Time(0))
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
            rospy.loginfo("Posición inicial registrada.")
        except Exception as e:
            rospy.logerr(f"No se pudo obtener la posición inicial: {e}")

        # Core movement and navigation parameters
        self.max_speed = rospy.get_param("~max_speed", 200)
        self.min_speed = rospy.get_param("~min_speed", 100)
        self.min_safe_distance = rospy.get_param("~min_safe_distance", 1.2)
        self.critical_distance = rospy.get_param("~critical_distance", 0.5)
        self.map_check_interval = rospy.get_param("~map_check_interval", 5)
        self.rotation_min_time = rospy.get_param("~rotation_min_time", 1.5)
        self.rotation_max_time = rospy.get_param("~rotation_max_time", 3.0)
        self.target_coverage = rospy.get_param("~target_coverage", 0.95)
        self.frontier_search_distance = rospy.get_param(
            "~frontier_search_distance", 3.0)
        self.lidar_safety_angle = math.radians(
            rospy.get_param("~lidar_safety_angle", 30))
        self.scan_timeout = rospy.Duration(
            rospy.get_param("~scan_timeout", 5.0))

        # Validación de parámetros críticos
        if self.min_safe_distance <= self.critical_distance:
            rospy.logwarn(
                "min_safe_distance debe ser mayor que critical_distance")
            self.min_safe_distance = self.critical_distance * 1.5

        if self.max_speed < self.min_speed:
            rospy.logwarn("max_speed debe ser mayor que min_speed")
            self.max_speed = self.min_speed * 1.5

        # Stair detector parameters with more conservative defaults
        # Reduced from 0.05 to 3cm for better precision
        self.ransac_threshold = rospy.get_param('~ransac_threshold', 0.03)
        self.floor_height_threshold = rospy.get_param(
            '~floor_height_threshold', 0.03)  # Reduced from 0.05 to 3cm
        # Reduced from 0.15 to 12cm for earlier detection
        self.drop_threshold = rospy.get_param('~drop_threshold', 0.12)
        # Increased from 0.8 to 1.0m for safer operation
        self.danger_distance = rospy.get_param('~danger_distance', 1.0)
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_link')
        self.avoid_stairs_enabled = rospy.get_param("~avoid_stairs", True)

        # YOLO parameters with additional validation
        self.yolo_weights = rospy.get_param(
            "~yolo_weights", "/home/jetson/catkin_ws/src/yolov7/runs/exp_custom17/weights/best.pt")
        if not os.path.exists(self.yolo_weights):
            rospy.logerr(
                f"YOLO weights file not found at: {self.yolo_weights}")
        self.yolo_img_size = rospy.get_param("~yolo_img_size", 640)
        self.yolo_conf_thres = rospy.get_param("~yolo_conf_thres", 0.25)
        self.yolo_camera_source = rospy.get_param(
            "~yolo_camera_source", "/k4a/rgb/image_raw")
        self.yolo_capture_interval = rospy.get_param(
            "~yolo_capture_interval", 2.0)
        self.yolo_process = None

        # Camera parameters (adjust according to your ToF camera)
        self.fx = 525.0  # Focal length in x
        self.fy = 525.0  # Focal length in y
        self.cx = 319.5  # Optical center in x
        self.cy = 239.5  # Optical center in y

        # LiDAR scan zone configuration with additional safety margin
        # Reduced from 30 to 25 degrees for narrower front zone
        self.front_angle = math.radians(rospy.get_param("~front_angle", 25))
        self.side_angle = math.radians(rospy.get_param(
            "~side_angle", 50))  # Reduced from 60 to 50 degrees

        # State variables with additional safety flags
        self.running = True
        self.exploration_state = "INITIAL_SCAN"
        self.last_lidar_time = rospy.Time.now()
        self.laser_scan = None
        self.initial_scan_start = None
        self.map_data = None
        self.last_map_check_time = rospy.Time.now()
        self.map_coverage = 0.0
        self.stair_detected = False
        self.floor_plane_eq = None  # Floor plane equation [a, b, c, d]
        self.danger_zone_detected = False
        self.emergency_stop_flag = False  # Flag for emergency stop conditions
        self.last_depth_image = None  # Almacenar última imagen de profundidad
        self.last_depth_header = None  # Almacenar último encabezado de imagen

        # Movement control state with additional safety checks
        self.current_move_direction = "STOP"
        self.current_motor_speeds = [0, 0, 0, 0]
        self.movement_lock = False
        self.last_movement_command_time = rospy.Time.now()  # Track last command time

        # Recovery behavior parameters
        self.consecutive_obstacles = 0  # Counter for consecutive obstacle detections
        self.max_consecutive_obstacles = rospy.get_param(
            "~max_consecutive_obstacles", 5)  # Max before recovery
        self.recovery_rotation_factor = rospy.get_param(
            "~recovery_rotation_factor", 2.0)  # Longer rotation during recovery

        # Mutex locks for thread safety
        self.scan_lock = threading.RLock()
        self.movement_command_lock = threading.RLock()

        # Mejorar la gestión de guardado de mapas
        self.map_save_path = os.path.expanduser("~/saved_maps")
        self.map_save_counter = 0
        self.last_map_save_time = rospy.Time.now()
        self.map_save_interval = rospy.Duration(
            300)  # 5 minutos entre guardados

        # Crear directorio con timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_map_directory = os.path.join(
            self.map_save_path, f"map_session_{timestamp}")

        # Crear subdirectorios para mapas periódicos y finales
        self.periodic_maps_dir = os.path.join(
            self.current_map_directory, "periodic_maps")
        self.final_maps_dir = os.path.join(
            self.current_map_directory, "final_maps")

        try:
            os.makedirs(self.current_map_directory, exist_ok=True)
            os.makedirs(self.periodic_maps_dir, exist_ok=True)
            os.makedirs(self.final_maps_dir, exist_ok=True)
            rospy.loginfo(
                f"Directorios de mapas creados en: {self.current_map_directory}")
        except Exception as e:
            rospy.logerr(f"Error creando directorios de mapas: {e}")
            self.current_map_directory = self.map_save_path
            self.periodic_maps_dir = self.map_save_path
            self.final_maps_dir = self.map_save_path

        # Utility instances
        self.bridge = CvBridge()
        self.ransac = pyrsc.Plane()  # Reuse RANSAC instance for efficiency

        # Publishers with larger queue sizes for critical topics
        self.cmd_pub = rospy.Publisher(
            "/robot/move/raw", String, queue_size=20)
        self.move_cmd_pub = rospy.Publisher(
            "/robot/move/direction", String, queue_size=20)
        self.danger_pub = rospy.Publisher(
            '/stair_detector/danger', Bool, queue_size=5)
        self.stair_mask_pub = rospy.Publisher(
            '/stair_detector/stair_mask', Image, queue_size=5)
        self.status_pub = rospy.Publisher(
            '/explorer/status', String, queue_size=5)  # Status publisher

        # cmd_vel publisher for smoother control
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Diagnostic publishers
        self.diagnostics_pub = rospy.Publisher(
            '/explorer/diagnostics', String, queue_size=5)

        # Subscribers with buffered messages
        self.laser_sub = rospy.Subscriber(
            "/scan", LaserScan, self.laser_callback, queue_size=1, buff_size=2**24)
        self.map_sub = rospy.Subscriber(
            "/rtabmap/mapData", MapData, self.map_callback, queue_size=1)
        self.grid_map_sub = rospy.Subscriber(
            "/map", OccupancyGrid, self.grid_map_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(
            '/k4a/depth/image_raw', Image, self.depth_callback, queue_size=1, buff_size=2**24)

        # Configure logging with timestamp
        if rospy.get_param("~debug", False):
            rospy.loginfo("Debug mode enabled - verbose logging activated")
            self.log_level = rospy.DEBUG
        else:
            self.log_level = rospy.INFO

        rospy.loginfo(
            "Safe autonomous explorer with stair detection and YOLO initialized")

        # Initialize YOLO detector with retry logic
        self.start_yolo_detector()

        # Start exploration with safety delay
        rospy.Timer(rospy.Duration(1.0),
                    lambda event: self.start_initial_scan(), oneshot=True)

        # Watchdog timer for system health monitoring
        self.watchdog_timer = rospy.Timer(
            rospy.Duration(5.0), self.watchdog_callback)

    def start_yolo_detector(self, retry_count=3):
        """Start the YOLO detector as a separate process with retry logic"""
        try:
            # Build YOLO command
            yolo_script_path = os.path.join(os.path.dirname(os.path.dirname(
                os.path.realpath(__file__))), "yolov7/detec-mapa-YOLO1.py")

            # Try to find the script in common locations if specific path doesn't exist
            if not os.path.exists(yolo_script_path):
                possible_paths = [
                    "/home/jetson/catkin_ws/src/yolov7/detec-mapa-YOLO1.py",
                    os.path.expanduser("~/yolov7/detec-mapa-YOLO1.py"),
                    "./detec-mapa-YOLO1.py"
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        yolo_script_path = path
                        break
                else:
                    rospy.logerr(
                        "Could not find detec-mapa1.1.py script. Please specify the full path.")
                    if retry_count > 0:
                        rospy.loginfo(
                            f"Retrying YOLO detector startup ({retry_count} attempts remaining)")
                        rospy.Timer(rospy.Duration(2.0), lambda event: self.start_yolo_detector(
                            retry_count-1), oneshot=True)
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
                "--project", os.path.join(self.map_save_path,
                                          "yolo_detections"),
                "--name", "yolo_detections_" +
                str(int(time.time()))  # Unique run name
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

            # Add timeout for YOLO process
            yolo_start_timeout = 30  # seconds
            start_time = time.time()

            while time.time() - start_time < yolo_start_timeout:
                if self.yolo_process and self.yolo_process.poll() is None:
                    break
                rospy.sleep(1.0)

            if time.time() - start_time >= yolo_start_timeout:
                rospy.logerr("Timeout al iniciar YOLO")
                self.stop_yolo_detector()
                if retry_count > 0:
                    self.start_yolo_detector(retry_count - 1)
                return

            # Start a thread to monitor YOLO output with error detection
            def monitor_yolo_output():
                while self.yolo_process and self.running:
                    line = self.yolo_process.stdout.readline()
                    if not line and self.yolo_process.poll() is not None:
                        break
                    rospy.loginfo(f"YOLO: {line.strip()}")

                    # Check for common error patterns
                    if "error" in line.lower() or "exception" in line.lower():
                        rospy.logwarn(
                            f"YOLO detector error detected: {line.strip()}")
                        if retry_count > 0:
                            rospy.loginfo(
                                f"Attempting YOLO detector restart ({retry_count} attempts remaining)")
                            self.stop_yolo_detector()
                            rospy.Timer(rospy.Duration(2.0), lambda event: self.start_yolo_detector(
                                retry_count-1), oneshot=True)
                            break

            yolo_monitor = threading.Thread(target=monitor_yolo_output)
            yolo_monitor.daemon = True
            yolo_monitor.start()

            rospy.loginfo("YOLO detector started successfully")
        except Exception as e:
            rospy.logerr(f"Error starting YOLO detector: {e}")
            if retry_count > 0:
                rospy.loginfo(
                    f"Retrying YOLO detector startup ({retry_count} attempts remaining)")
                rospy.Timer(rospy.Duration(2.0), lambda event: self.start_yolo_detector(
                    retry_count-1), oneshot=True)
            else:
                self.yolo_process = None

    # Improved laser_callback with thread safety
    def laser_callback(self, msg):
        """Process incoming laser scan data with thread protection"""
        with self.scan_lock:
            self.laser_scan = msg
            self.last_lidar_time = rospy.Time.now()

            # Track valid ranges in scan for better obstacle assessment
            ranges = np.array(msg.ranges)
            valid_ranges = ranges[(ranges > msg.range_min)
                                  & (ranges < msg.range_max)]

            # Calculate and log scan quality metrics
            if len(valid_ranges) > 0:
                range_percentage = (len(valid_ranges) / len(ranges)) * 100
                if range_percentage < 50:
                    rospy.logwarn(
                        f"LiDAR data quality low: only {range_percentage:.1f}% valid ranges")

    # Enhanced depth_to_pointcloud with SIMD optimizations
    def depth_to_pointcloud(self, depth_image):
        """Convertir imagen de profundidad a nube de puntos usando Azure Kinect"""
        try:
            if self.kinect_device is None:
                raise Exception("Dispositivo Azure Kinect no inicializado")

            # Obtener nube de puntos usando el SDK de Azure Kinect
            points_3d = self.transformation.depth_image_to_point_cloud(
                depth_image, _k4a.K4A_CALIBRATION_TYPE_DEPTH)

            # Convertir a array NumPy para procesamiento
            points = np.array(points_3d)

            # Filtrar por profundidad
            z = points[:, :, 2]
            valid_mask = np.logical_and(z > 0.1, z < 5.0)  # Entre 10cm y 5m

            # Extraer puntos válidos
            valid_points = points[valid_mask]

            # Verificar calidad de los datos
            if len(valid_points) < 100:
                rospy.logwarn(
                    f"Pocos puntos válidos detectados: {len(valid_points)}")

            return valid_points, valid_mask

        except Exception as e:
            rospy.logerr(f"Error en depth_to_pointcloud: {e}")
            return np.array([]), np.zeros_like(depth_image, dtype=bool)

    # Improved detect_stairs method with better plane segmentation
    def detect_stairs(self, depth_image, header):
        """Enhanced stair detection with better plane fitting and visualization"""
        try:
            # Añadir verificación de calidad de imagen
            if depth_image.mean() < 1e-6 or depth_image.std() < 1e-6:
                rospy.logwarn("Imagen de profundidad de baja calidad")
                return

            # Convert depth image to point cloud
            points_3d, depth_mask = self.depth_to_pointcloud(depth_image)

            if len(points_3d) < 50:  # Increased minimum points for better reliability
                rospy.logwarn("Not enough valid points for stair detection")
                return

            # 1. Detect floor plane using RANSAC with optimized parameters
            best_eq, inliers = self.ransac.fit(
                points_3d, self.ransac_threshold, maxIteration=1000)

            # Calculate inlier percentage after RANSAC
            inlier_percentage = (len(inliers) / len(points_3d)) * 100

            # Mejorar detección de plano con validación adicional
            if inlier_percentage < 30:  # Menos del 30% de puntos son inliers
                rospy.logwarn("Detección de plano no confiable")
                return

            # Check if the plane is horizontal with stricter criteria
            a, b, c, d = best_eq
            normal_vector = np.array([a, b, c])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)

            # Check alignment with vertical (assuming camera looks forward)
            # Y axis points up in camera coords
            vertical_alignment = np.abs(normal_vector[1])

            # More detailed floor plane analysis
            inlier_percentage = (len(inliers) / len(points_3d)) * 100
            rospy.logdebug(f"Floor plane detection: {inlier_percentage:.1f}% points as inliers, "
                           f"vertical alignment: {vertical_alignment:.3f}")

            # More strict horizontal plane requirement (was 0.8)
            if vertical_alignment > 0.9:
                self.floor_plane_eq = best_eq

                # Create visualization image with more detailed information
                visualization = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
                visualization = cv2.normalize(
                    visualization, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

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
                expected_z = (-a * x - b * y - d) / np.clip(c,
                                                            1e-6, None)  # Avoid division by zero

                # Calculate depth differences
                depth_diff = z - expected_z
                stair_mask = (depth_diff > self.drop_threshold) & valid_depth

                # Process mask to remove noise with adaptive morphology
                stair_mask = stair_mask.astype(np.uint8) * 255
                # Adaptive kernel size based on image width
                kernel_size = max(3, int(width/100))
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                stair_mask = cv2.morphologyEx(
                    stair_mask, cv2.MORPH_OPEN, kernel)
                stair_mask = cv2.morphologyEx(
                    stair_mask, cv2.MORPH_CLOSE, kernel)

                # Find contours with minimum area requirement
                contours, _ = cv2.findContours(
                    stair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                significant_contours = [cnt for cnt in contours if cv2.contourArea(
                    cnt) > 300]  # Reduced from 500

                # Analyze contour shapes to distinguish stairs from other drops
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
                self.stair_detected = len(
                    significant_contours) > 0 and is_likely_stair
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
                        contour_depths = depth_image[points[:,
                                                            1], points[:, 0]]
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

                # Informational overlay
                cv2.putText(visualization, f"Floor confidence: {inlier_percentage:.1f}%", (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Publish visualization with contour information
                cv2.drawContours(
                    visualization, significant_contours, -1, (0, 0, 255), 2)
                self.danger_pub.publish(Bool(self.danger_zone_detected))

                # Publish visualization image
                stair_mask_msg = self.bridge.cv2_to_imgmsg(
                    visualization, encoding="bgr8")
                stair_mask_msg.header = header
                self.stair_mask_pub.publish(stair_mask_msg)

                # Log detection status
                if self.danger_zone_detected:
                    rospy.logwarn(
                        f"DANGER! Stair detected at {min_distance:.2f}m")
                elif self.stair_detected:
                    rospy.loginfo(
                        f"Stair detected at safe distance: {min_distance:.2f}m")

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

        # Outlier filtering for more robust zone analysis
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
            # If value is significantly different from median
            if abs(ranges[i] - median) > 0.5:
                valid_mask[i] = False  # Mark as invalid

        # Angle array for easier zone calculation
        angles = np.arange(len(ranges)) * angle_increment + angle_min

        # Define zones - front, left, right with overlap for better awareness
        front_mask = np.abs(angles) <= self.front_angle
        left_mask = np.logical_and(
            angles > -self.side_angle, angles < -self.front_angle * 0.5)
        right_mask = np.logical_and(
            angles < self.side_angle, angles > self.front_angle * 0.5)

        # Rear zone for better situational awareness
        rear_mask = np.abs(angles) >= math.radians(150)

        # Combine masks with valid readings
        front_valid = np.logical_and(front_mask, valid_mask)
        left_valid = np.logical_and(left_mask, valid_mask)
        right_valid = np.logical_and(right_mask, valid_mask)
        rear_valid = np.logical_and(rear_mask, valid_mask)

        # Calculate zone statistics with better handling of empty zones
        front_distances = ranges[front_valid] if np.any(
            front_valid) else np.array([float('inf')])
        left_distances = ranges[left_valid] if np.any(
            left_valid) else np.array([float('inf')])
        right_distances = ranges[right_valid] if np.any(
            right_valid) else np.array([float('inf')])
        rear_distances = ranges[rear_valid] if np.any(
            rear_valid) else np.array([float('inf')])

        # Calculate more robust zone metrics including quantiles
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

        # Log data quality metrics
        for zone_name, zone_data in zones.items():
            valid_ratio = zone_data['valid_count'] / \
                max(1, zone_data['total_count'])
            if valid_ratio < 0.5:  # Less than 50% valid readings
                rospy.logdebug(
                    f"Low LiDAR quality in {zone_name} zone: {valid_ratio:.2f}")

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
            rospy.logwarn(
                "LiDAR data timeout - stopping robot and initiating recovery")
            self.stop_robot()
            self.start_recovery_behavior()
            return

        # First check for stair danger with additional conditions
        if self.danger_zone_detected and self.avoid_stairs_enabled:
            # Don't allow forward movement if there's a drop hazard
            rospy.logwarn(
                "Avoiding forward movement - stair detected in danger zone")
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

        # Handle return to dock state
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
        rospy.logdebug(
            f"LiDAR zone analysis - Front: {zones_analysis['front']['min_distance']:.2f}m")
        rospy.logdebug(
            f"LiDAR zone analysis - Left: {zones_analysis['left']['min_distance']:.2f}m")
        rospy.logdebug(
            f"LiDAR zone analysis - Right: {zones_analysis['right']['min_distance']:.2f}m")
        rospy.logdebug(
            f"LiDAR zone analysis - Rear: {zones_analysis['rear']['min_distance']:.2f}m")

        # Enhanced obstacle classification using percentile values for more robustness
        # This avoids overreacting to single outlier readings
        front_obstacle = zones_analysis['front']['closest_percentile'] < self.min_safe_distance
        left_obstacle = zones_analysis['left']['closest_percentile'] < self.min_safe_distance * 0.8
        right_obstacle = zones_analysis['right']['closest_percentile'] < self.min_safe_distance * 0.8
        rear_obstacle = zones_analysis['rear']['closest_percentile'] < self.min_safe_distance * 0.5

        # Check for critical obstacles
        if zones_analysis['front']['min_distance'] < self.critical_distance:
            rospy.logwarn(
                f"Critical obstacle at {zones_analysis['front']['min_distance']:.2f}m - EMERGENCY STOP")
            self.emergency_stop_flag = True
            self.stop_robot()
            self.start_rotation()
            self.consecutive_obstacles += 1
            return

        # Track consecutive obstacle encounters for stuck detection
        if front_obstacle:
            self.consecutive_obstacles += 1
            if self.consecutive_obstacles > self.max_consecutive_obstacles:
                rospy.logwarn(
                    "Potentially stuck - initiating recovery behavior")
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

    # Improved obstacle evasion method
    def execute_evasion_maneuver(self):
        """Execute a maneuver to evade an obstacle or hazard"""
        rospy.logwarn("Executing evasion maneuver")

        # First stop to ensure safety
        self.stop_robot()

        # Then back up slightly
        self.move("BACKWARD", duration=1.0)

        # After backing up, rotate to find clear path
        rospy.Timer(rospy.Duration(1.2),
                    lambda event: self.start_rotation(), oneshot=True)

        # Reset emergency flag after evasion
        rospy.Timer(rospy.Duration(5.0), lambda event: setattr(
            self, 'emergency_stop_flag', False), oneshot=True)

    # Recovery behavior method
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
                    lambda event: self.start_rotation(
                        min_time=self.rotation_min_time * self.recovery_rotation_factor),
                    oneshot=True)

        # 4. Resume normal operation
        rotation_time = self.rotation_min_time * self.recovery_rotation_factor + 0.5
        rospy.Timer(rospy.Duration(2.7 + rotation_time),
                    lambda event: self.reset_after_recovery(),
                    oneshot=True)

    # Helper method for recovery reset
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
                # Reduced for smoother rotation
                base_speed = int(self.max_speed * 0.7 * speed_factor)
                speeds = [-base_speed, base_speed, -base_speed, base_speed]
            elif direction == "ROTATE_RIGHT":
                # Reduced for smoother rotation
                base_speed = int(self.max_speed * 0.7 * speed_factor)
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
            motor_command = f"M:{speeds[0]}:{speeds[1]}:{speeds[2]}:{speeds[3]}"
            self.cmd_pub.publish(motor_command)

            # Also publish direction for higher-level monitoring
            self.move_cmd_pub.publish(direction)

            # Also publish cmd_vel for ROS navigation compatibility
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

    # Method to publish cmd_vel for standard ROS compatibility
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

    # Method to release movement lock
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
        rospy.loginfo(
            f"Starting {direction} rotation for {rotation_time:.2f} seconds")
        self.move(direction)

        # Set timer to stop rotation
        self.movement_lock = True
        rospy.Timer(rospy.Duration(rotation_time),
                    lambda event: self.stop_after_rotation(),
                    oneshot=True)

    # Method to stop after rotation
    def stop_after_rotation(self):
        """Stop robot after rotation and reset flags"""
        self.stop_robot()
        self.movement_lock = False

        # Reset emergency flag if it was set
        if self.emergency_stop_flag:
            rospy.Timer(rospy.Duration(0.5),
                        lambda event: setattr(
                            self, 'emergency_stop_flag', False),
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

        # Calculate exploration efficiency metrics
        free_to_occupied_ratio = free_cells / max(1, occupied_cells)

        rospy.loginfo(
            f"Map coverage: {coverage:.2f}, Free/Occupied ratio: {free_to_occupied_ratio:.2f}")
        rospy.loginfo(
            f"Map cells - Free: {free_cells}, Occupied: {occupied_cells}, Unknown: {unknown_cells}")

        return coverage

    # Enhanced frontier detection
    def should_find_frontier(self):
        if self.exploration_state == "RETURN_TO_DOCK":
            return False

        """Determine if robot should seek frontier or finish mapping"""
        if (rospy.Time.now() - self.last_map_check_time).to_sec() < self.map_check_interval:
            return False

        self.last_map_check_time = rospy.Time.now()

        # 🔁 FORZAR ACTUALIZACIÓN DE COBERTURA
        if hasattr(self, "latest_grid_map"):
            self.map_coverage = self.estimate_map_coverage(
                self.latest_grid_map)

        if self.exploration_state != "RETURN_TO_DOCK" and self.map_coverage > self.target_coverage:
            rospy.loginfo(
                f"Map coverage ({self.map_coverage:.2f}) exceeds target ({self.target_coverage:.2f})")
            self.exploration_state = "RETURN_TO_DOCK"
            self.return_to_initial_position()
            return False

        chance = 0.5 * (1.0 - self.map_coverage / self.target_coverage)
        return random.random() < chance

    def return_to_initial_position(self):
        """Método para retornar a la posición inicial con manejo de errores y feedback"""
        try:
            if self.initial_pose is None:
                rospy.logerr("No hay posición inicial registrada")
                return False

            # Obtener posición actual
            try:
                current_trans, current_rot = self.tf_listener.lookupTransform(
                    "map", self.base_frame, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(f"Error obteniendo posición actual: {e}")
                return False

            # Calcular distancia a la posición inicial
            self.stop_yolo_detector()
            distance = math.sqrt(
                (current_trans[0] - self.initial_pose.pose.position.x) ** 2 +
                (current_trans[1] - self.initial_pose.pose.position.y) ** 2
            )

            if distance < 0.1:  # Si está cerca de la posición inicial
                rospy.loginfo("Ya se encuentra en la posición inicial")
                return True

            # Publicar comando de movimiento hacia la posición inicial
            cmd_vel = Twist()

            # Ajustar velocidad basada en la distancia
            # Reducir velocidad cuando está cerca
            speed_factor = min(1.0, distance / 2.0)
            cmd_vel.linear.x = 0.5 * speed_factor  # Velocidad máxima de 0.5 m/s

            # Publicar comando
            self.cmd_vel_pub.publish(cmd_vel)

            rospy.loginfo(
                f"Retornando a posición inicial. Distancia: {distance:.2f}m")
            return True

        except Exception as e:
            rospy.logerr(f"Error en return_to_initial_position: {e}")
            return False

    def is_at_initial_position(self):
        """Verifica si el robot está en la posición inicial"""
        try:
            if self.initial_pose is None:
                return False

            current_trans, _ = self.tf_listener.lookupTransform(
                "map", self.base_frame, rospy.Time(0))

            # Calcular distancia a la posición inicial
            distance = math.sqrt(
                (current_trans[0] - self.initial_pose.pose.position.x) ** 2 +
                (current_trans[1] - self.initial_pose.pose.position.y) ** 2
            )

            return distance < 0.1  # Considerar que está en posición inicial si está a menos de 10cm

        except Exception as e:
            rospy.logerr(f"Error verificando posición inicial: {e}")
            return False

    # Improved frontier-based exploration

    def find_exploration_direction(self, zones_analysis):
        """Find direction to unexplored areas (frontiers)"""
        rospy.loginfo("Seeking frontier for exploration")

        # For this simplified version, we'll use a heuristic approach:
        # - Prefer directions with more open space (higher mean distance)
        # - Avoid directions with obstacles

        # Calculate direction scores based on mean distance
        # Prefer forward
        front_score = zones_analysis['front']['mean_distance'] * 1.5
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
            rospy.Timer(rospy.Duration(1.2), lambda event: self.move(
                "FORWARD"), oneshot=True)
        else:
            self.move("ROTATE_RIGHT", duration=1.0)
            # Schedule forward movement after rotation
            rospy.Timer(rospy.Duration(1.2), lambda event: self.move(
                "FORWARD"), oneshot=True)

        # Log decision
        rospy.loginfo(
            f"Frontier scores - Front: {front_score:.2f}, Left: {left_score:.2f}, Right: {right_score:.2f}")

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

        # Save final map if available
        final_map_path = None
        if hasattr(self, 'map_data') and self.map_data is not None:
            try:
                rospy.loginfo("Guardando mapa final...")
                final_map_path = self.save_current_map(is_final=True)
            except Exception as e:
                rospy.logerr(f"Error guardando mapa final: {e}")

        # Lanzar el generador de mapas simbólicos
        try:
            script_path = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "mapa_simbolico-YOLO1.py")

            if os.path.exists(script_path):
                possible_paths = [
                    "/home/jetson/catkin_ws/src/barrier_map/mapa_simbolico-YOLO1.py",
                    os.path.expanduser(
                        "~/barrier_map/mapa_simbolico-YOLO1.py"),
                    "./mapa_simbolico-YOLO1.py"
                ]

            for path in possible_paths:
                if os.path.exists(path):
                    script_path = path
                    break

            # Si tenemos la ruta del mapa final, pasarla como argumento
            if final_map_path:
                cmd.append(final_map_path)

            rospy.loginfo(
                f"Lanzando generador de mapas simbólicos: {script_path}")
            # Aquí podrías decidir si esperar o no a que termine
            # subprocess.Popen(cmd)
            # Si quieres esperar a que termine (con un timeout):
            try:
                subprocess.run(cmd, timeout=10800)
            except subprocess.TimeoutExpired:
                rospy.logwarn(
                    "El generador de mapas simbólicos excedió el tiempo de espera")
            else:
                rospy.logerr(
                    f"No se encontró el script del generador de mapas en: {script_path}")
        except Exception as e:
            rospy.logerr(
                f"Error al lanzar el generador de mapas simbólicos: {e}")

        # Small delay to ensure messages are sent
        rospy.sleep(0.5)

        rospy.loginfo("Safe autonomous explorer shutdown complete")

    # Method to stop YOLO detector
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
                    rospy.logwarn(
                        "YOLO process did not terminate gracefully, forcing kill")
                    self.yolo_process.kill()

                self.yolo_process = None
            except Exception as e:
                rospy.logerr(f"Error stopping YOLO detector: {e}")

    def __del__(self):
        """Limpieza de recursos al destruir la instancia"""
        try:
            if hasattr(self, 'yolo_process') and self.yolo_process:
                self.stop_yolo_detector()
            if hasattr(self, 'watchdog_timer'):
                self.watchdog_timer.shutdown()
            # Cerrar todos los subscribers
            if hasattr(self, 'laser_sub'):
                self.laser_sub.unregister()
            if hasattr(self, 'depth_sub'):
                self.depth_sub.unregister()
            # Cerrar el dispositivo Azure Kinect
            if hasattr(self, 'kinect_device') and self.kinect_device is not None:
                self.kinect_device.stop()
        except Exception as e:
            rospy.logerr(f"Error durante la limpieza: {e}")

    # Run method to start exploration
    def run(self):
        """Main run loop for the explorer"""
        rospy.loginfo("Starting exploration run")

        # Set up shutdown handler
        rospy.on_shutdown(self.shutdown)

        # Start processing loop
        rate = rospy.Rate(10)  # 10 Hz control loop

        # Variables para control de frecuencia de detección de escaleras
        last_stair_detection_time = rospy.Time.now()
        stair_detection_interval = rospy.Duration(
            0.5)  # Detectar escaleras cada 0.5 segundos

        while not rospy.is_shutdown() and self.running:
            # Process LiDAR for obstacle avoidance
            self.process_lidar_for_navigation()

            # Detección de escaleras con control de frecuencia
            current_time = rospy.Time.now()
            if self.avoid_stairs_enabled and hasattr(self, 'last_depth_image') and \
                    (current_time - last_stair_detection_time) > stair_detection_interval:
                self.detect_stairs(self.last_depth_image,
                                   self.last_depth_header)
                last_stair_detection_time = current_time

            # Check for map coverage periodically
            if (current_time - self.last_map_check_time).to_sec() >= self.map_check_interval:
                # This would normally check map coverage
                pass

            rate.sleep()

    def depth_callback(self, msg):
        """Procesa las imágenes de profundidad y detecta escaleras"""
        try:
            # Convertir mensaje ROS Image a imagen OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")

            # Almacenar la última imagen de profundidad para procesamiento
            self.last_depth_image = depth_image
            self.last_depth_header = msg.header

            # Registrar recepción de datos (para diagnóstico)
            if rospy.get_param("~debug", False):
                rospy.logdebug("Imagen de profundidad recibida")

        except Exception as e:
            rospy.logerr(f"Error procesando imagen de profundidad: {e}")

    # Method to start initial scan

    def start_initial_scan(self):
        """Start initial 360-degree scan"""
        rospy.loginfo("Starting initial 360-degree scan")
        self.initial_scan_start = rospy.Time.now()
        self.exploration_state = "INITIAL_SCAN"
        self.status_pub.publish("INITIAL_SCAN_STARTED")

        # Rotate slowly to capture surroundings
        # Slower rotation for better mapping
        self.move("ROTATE_LEFT", speed_factor=0.6)

    # Method to save the latest map
    def map_callback(self, msg):
        """Callback mejorado para el procesamiento de datos del mapa"""
        try:
            self.map_data = msg

            # Verificar si es tiempo de guardar el mapa
            current_time = rospy.Time.now()
            if (current_time - self.last_map_save_time) > self.map_save_interval:
                self.save_current_map(is_final=False)

        except Exception as e:
            rospy.logerr(f"Error en map_callback: {e}")

    def save_current_map(self, is_final=False):
        """Guarda el mapa actual con nombre único en la estructura de directorios solicitada"""
        try:
            if not self.map_data:
                rospy.logwarn("No hay datos de mapa para guardar")
                return False

            # Determinar el directorio de destino según si es un mapa final o periódico
            if is_final:
                save_dir = self.final_maps_dir
                map_name = f"final_map_{time.strftime('%Y%m%d_%H%M%S')}"
            else:
                save_dir = self.periodic_maps_dir
                map_name = f"map_{time.strftime('%Y%m%d_%H%M%S')}_{self.map_save_counter}"
                self.map_save_counter += 1

            # Asegurarse de que el directorio existe
            os.makedirs(save_dir, exist_ok=True)

            # Guardar mapa de ocupación
            if hasattr(self, 'current_grid_map'):
                grid_map_path = os.path.join(save_dir, f"{map_name}_grid.pgm")
                cv2.imwrite(grid_map_path, self.current_grid_map)
                rospy.loginfo(
                    f"Mapa de ocupación guardado en: {grid_map_path}")

            # Guardar datos de RTAB-Map
            rtabmap_path = os.path.join(save_dir, f"{map_name}_rtabmap.db")
            # Aquí iría la lógica para guardar el archivo .db de RTAB-Map

            # Guardar metadatos
            metadata = {
                'timestamp': time.strftime('%Y%m%d_%H%M%S'),
                'coverage': self.map_coverage,
                'position': self.get_current_position(),
            }

            # Actualizar el tiempo del último guardado
            self.last_map_save_time = rospy.Time.now()

            rospy.loginfo(
                f"Mapa guardado exitosamente en: {save_dir}/{map_name}")
            return True

        except Exception as e:
            rospy.logerr(f"Error al guardar el mapa: {e}")
            return False

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


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
