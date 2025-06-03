#!/usr/bin/env python3

import rospy
import numpy as np
import os
import signal
import subprocess
import sys
import time
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String
from rtabmap_msgs.msg import MapData
from nav_msgs.msg import OccupancyGrid
import tf
import random
import math
from geometry_msgs.msg import PoseStamped
from nav_msgs.srv import GetMap
import cv2
from cv_bridge import CvBridge

class LidarToFExplorerWithYOLO:
    def __init__(self):
        rospy.init_node("lidar_tof_explorer", anonymous=True)
       
        # Parameters - Get from ROS parameter server instead of hardcoding
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
        
        # ToF Camera parameters
        self.tof_min_valid = rospy.get_param("~tof_min_valid", 100)   # 0.1m in mm
        self.tof_max_valid = rospy.get_param("~tof_max_valid", 10000) # 10m in mm
        self.tof_floor_threshold = rospy.get_param("~tof_floor_threshold", 500) # mm difference to detect stairs
        self.tof_roi_ratio = rospy.get_param("~tof_roi_ratio", 0.3) # Center ROI ratio
        
        # YOLO parameters
        self.yolo_weights = rospy.get_param("~yolo_weights", "/home/jetson/catkin_ws/src/yolov7/runs/exp_custom17/weights/best.pt")
        self.yolo_img_size = rospy.get_param("~yolo_img_size", 640)
        self.yolo_conf_thres = rospy.get_param("~yolo_conf_thres", 0.25)
        self.yolo_camera_source = rospy.get_param("~yolo_camera_source", "0")
        self.yolo_capture_interval = rospy.get_param("~yolo_capture_interval", 2.0)
        self.yolo_process = None
        
        # Zone scanning configuration
        self.front_angle = math.radians(rospy.get_param("~front_angle", 30))
        self.side_angle = math.radians(rospy.get_param("~side_angle", 60))
        
        # State to avoid repeated commands
        self.current_move_direction = "STOP"
        self.current_motor_speeds = [0, 0, 0, 0]
       
        # State variables
        self.running = True
        self.exploration_state = "INITIAL_SCAN"
        self.last_lidar_time = rospy.Time.now()
        self.last_tof_time = rospy.Time.now()
        self.laser_scan = None
        self.tof_image = None
        self.tf_listener = tf.TransformListener()
        self.bridge = CvBridge()
        self.initial_scan_start = None
        self.map_data = None
        self.map_save_path = os.path.expanduser("~/saved_maps")
        self.last_map_check_time = rospy.Time.now()
        self.map_coverage = 0.0
        self.floor_detected = True  # Assume floor is present until proven otherwise
        
        # Movement lock for callback concurrency
        self.movement_lock = False
        
        # Ensure map directory exists
        os.makedirs(self.map_save_path, exist_ok=True)
       
        # Publishers
        self.cmd_pub = rospy.Publisher("/robot/move/raw", String, queue_size=10)
        self.move_cmd_pub = rospy.Publisher("/robot/move/direction", String, queue_size=10)
       
        # Subscribers
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback, queue_size=1)
        self.tof_sub = rospy.Subscriber("/k4a/depth/image_resized", Image, self.tof_callback, queue_size=1)
        self.map_sub = rospy.Subscriber("/rtabmap/mapData", MapData, self.map_callback, queue_size=1)
        self.grid_map_sub = rospy.Subscriber("/map", OccupancyGrid, self.grid_map_callback, queue_size=1)
       
        rospy.loginfo("Autonomous mapping explorer with LiDAR and ToF initialized")
        
        # Configure log levels
        if rospy.get_param("~debug", False):
            rospy.loginfo("Debug mode enabled - verbose logging activated")
            self.log_level = rospy.DEBUG
        else:
            self.log_level = rospy.INFO
            
        # Start YOLO detector
        self.start_yolo_detector()
        
        # Start exploration
        self.start_initial_scan()

    def tof_callback(self, msg):
        """Process ToF depth image data"""
        if self.movement_lock:
            rospy.logdebug("Movement in progress, skipping ToF processing")
            return
            
        try:
            self.tof_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            self.last_tof_time = rospy.Time.now()
            
            # Check for floor/stair detection
            self.check_floor_condition()
            
        except Exception as e:
            rospy.logerr(f"Error processing ToF image: {e}")

    def check_floor_condition(self):
        """Check ToF data for floor/stair conditions"""
        if self.tof_image is None:
            return
            
        height, width = self.tof_image.shape
        roi_height = int(height * self.tof_roi_ratio)
        roi_width = int(width * self.tof_roi_ratio)
        
        # Define ROIs - bottom center for floor, top center for stairs
        floor_roi = self.tof_image[height-roi_height:, width//2-roi_width//2:width//2+roi_width//2]
        stair_roi = self.tof_image[:roi_height, width//2-roi_width//2:width//2+roi_width//2]
        
        # Create valid masks
        floor_mask = (floor_roi > self.tof_min_valid) & (floor_roi < self.tof_max_valid)
        stair_mask = (stair_roi > self.tof_min_valid) & (stair_roi < self.tof_max_valid)
        
        # Calculate average distances
        floor_avg = np.mean(floor_roi[floor_mask]) if np.sum(floor_mask) > 0 else 0
        stair_avg = np.mean(stair_roi[stair_mask]) if np.sum(stair_mask) > 0 else 0
        
        # Check for sudden drop (stairs)
        if floor_avg > 0 and stair_avg > 0:
            height_diff = abs(floor_avg - stair_avg)
            if height_diff > self.tof_floor_threshold:
                rospy.logwarn(f"STAIR DETECTED! Height difference: {height_diff}mm")
                self.floor_detected = False
                self.stop_robot()
                return
                
        self.floor_detected = True

    def process_tof_for_navigation(self):
        """Analyze ToF data for obstacle detection"""
        if self.tof_image is None:
            return
            
        height, width = self.tof_image.shape
        roi_height = int(height * self.tof_roi_ratio)
        roi_width = int(width * self.tof_roi_ratio)
        
        # Center ROI for obstacle detection
        center_roi = self.tof_image[height//2-roi_height//2:height//2+roi_height//2, 
                                  width//2-roi_width//2:width//2+roi_width//2]
        
        # Create valid mask
        valid_mask = (center_roi > self.tof_min_valid) & (center_roi < self.tof_max_valid)
        
        # Check for obstacles
        if np.sum(valid_mask) > 0:
            min_dist = np.min(center_roi[valid_mask]) / 1000.0  # Convert mm to meters
            
            if min_dist < self.critical_distance:
                rospy.logwarn(f"ToF CRITICAL OBSTACLE at {min_dist:.2f}m - EMERGENCY STOP")
                self.stop_robot()
                self.start_rotation()
                return True  # Obstacle detected
                
        return False  # No obstacle detected

    def laser_callback(self, msg):
        """Process LiDAR scan data with protection against command accumulation"""
        if self.movement_lock:
            rospy.logdebug("Movement in progress, skipping LiDAR processing")
            return
            
        self.laser_scan = msg
        self.last_lidar_time = rospy.Time.now()
        self.process_sensors_for_navigation()

    def process_sensors_for_navigation(self):
        """Combine data from all sensors for navigation decisions"""
        # First check if sensors are responding
        if self.check_sensor_timeouts():
            return
            
        # Check floor condition from ToF
        if not self.floor_detected:
            rospy.logwarn("Floor not detected - avoiding movement")
            self.stop_robot()
            return
            
        # Check for obstacles with ToF (low obstacles)
        tof_obstacle = self.process_tof_for_navigation()
        if tof_obstacle:
            return
            
        # Then process LiDAR data
        self.process_lidar_for_navigation()

    def check_sensor_timeouts(self):
        """Check if sensor data is too old"""
        lidar_timeout = (rospy.Time.now() - self.last_lidar_time) > self.scan_timeout
        tof_timeout = (rospy.Time.now() - self.last_tof_time) > self.scan_timeout
        
        if lidar_timeout or tof_timeout:
            rospy.logwarn(f"Sensor timeout - LiDAR: {lidar_timeout}, ToF: {tof_timeout}")
            self.stop_robot()
            self.start_recovery_behavior()
            return True
        return False

    def process_lidar_for_navigation(self):
        """Analyze LiDAR data by zones for obstacle avoidance and navigation"""
        if self.laser_scan is None:
            return
            
        # Handle initial scan state
        if self.exploration_state == "INITIAL_SCAN":
            if (rospy.Time.now() - self.initial_scan_start) > rospy.Duration(10):
                self.stop_robot()
                self.exploration_state = "SCANNING"
                rospy.loginfo("Initial scan complete, beginning exploration")
            return
            
        # Analyze LiDAR data by zones (front, left, right)
        zones_analysis = self.analyze_lidar_zones()
        
        # Log zone analysis for debugging
        rospy.logdebug(f"LiDAR zone analysis: {zones_analysis}")
        
        # Check if front zone has obstacles
        if zones_analysis['front']['min_distance'] < self.critical_distance:
            rospy.logwarn(f"Critical obstacle ahead at {zones_analysis['front']['min_distance']:.2f}m - EMERGENCY STOP")
            self.stop_robot()
            self.start_rotation()
            return
        elif zones_analysis['front']['min_distance'] < self.min_safe_distance:
            rospy.loginfo(f"Close obstacle ahead at {zones_analysis['front']['min_distance']:.2f}m - adjusting path")
            
            # Choose rotation direction based on which side has more space
            if zones_analysis['left']['min_distance'] > zones_analysis['right']['min_distance']:
                self.move("ROTATE_LEFT")
            else:
                self.move("ROTATE_RIGHT")
            return
                
        # If no obstacles detected in front, move forward
        if self.exploration_state == "SCANNING":
            # Check if we need to look for frontiers
            if self.should_find_frontier():
                self.find_exploration_direction(zones_analysis)
            else:
                self.move("FORWARD")
                
      def analyze_lidar_zones(self):
        """Analyze LiDAR data by zones (front, left, right)"""
        angle_min = self.laser_scan.angle_min
        angle_increment = self.laser_scan.angle_increment
        num_readings = len(self.laser_scan.ranges)
        center_idx = num_readings // 2
        
        # Calculate indices for each zone
        # Front zone
        front_range = int(self.front_angle / angle_increment)
        front_start = center_idx - front_range
        front_end = center_idx + front_range
        
        # Left zone (from front edge to side angle)
        left_start = front_start - int(self.side_angle / angle_increment)
        left_end = front_start
        
        # Right zone (from front edge to side angle)
        right_start = front_end
        right_end = front_end + int(self.side_angle / angle_increment)
        
        # Ensure indices are within range
        front_start = max(0, front_start)
        front_end = min(num_readings, front_end)
        left_start = max(0, left_start)
        right_end = min(num_readings, right_end)
        
        # Function to analyze a range of readings
        def analyze_range(start, end):
            ranges = self.laser_scan.ranges[start:end]
            valid_ranges = [r for r in ranges if self.laser_scan.range_min < r < self.laser_scan.range_max]
            
            if not valid_ranges:
                return {
                    'min_distance': float('inf'),
                    'max_distance': 0,
                    'avg_distance': 0,
                    'valid_readings': 0
                }
                
            return {
                'min_distance': min(valid_ranges),
                'max_distance': max(valid_ranges),
                'avg_distance': sum(valid_ranges) / len(valid_ranges),
                'valid_readings': len(valid_ranges)
            }
            
        # Analyze each zone
        zones = {
            'front': analyze_range(front_start, front_end),
            'left': analyze_range(left_start, left_end),
            'right': analyze_range(right_start, right_end)
        }
        
        return zones

    def should_find_frontier(self):
        """Determine if we should search for a frontier based on time or conditions"""
        return (rospy.Time.now() - self.last_map_check_time).to_sec() > self.map_check_interval

    def find_exploration_direction(self, zones_analysis):
        """Find the best direction to explore based on LiDAR zones"""
        self.last_map_check_time = rospy.Time.now()
        
        # Simple approach: go where there's most open space
        left_space = zones_analysis['left']['avg_distance']
        right_space = zones_analysis['right']['avg_distance']
        front_space = zones_analysis['front']['avg_distance']
        
        rospy.loginfo(f"Exploration spaces - Front: {front_space:.2f}m, Left: {left_space:.2f}m, Right: {right_space:.2f}m")
        
        # If front has good space, keep going forward
        if front_space > self.min_safe_distance * 2:
            self.move("FORWARD")
        # Otherwise go in direction with most space
        elif left_space > right_space:
            self.move("FORWARD_LEFT")
        else:
            self.move("FORWARD_RIGHT")

    def map_callback(self, msg):
        """Process RTAB-Map data"""
        rospy.logdebug("Received RTAB-Map data")

    def grid_map_callback(self, msg):
        """Process and store the occupancy grid map"""
        self.map_data = msg
        rospy.loginfo("OccupancyGrid map received")
        
        # Calculate map coverage
        self.calculate_map_coverage()
        
        # Check if we need to save the map
        if (rospy.Time.now() - self.last_map_check_time) > rospy.Duration(self.map_check_interval):
            self.last_map_check_time = rospy.Time.now()
            self.save_map()
            
            # Check if we've reached target coverage
            if self.map_coverage >= self.target_coverage:
                rospy.loginfo(f"Target coverage of {self.target_coverage*100}% reached. Exploration complete.")
                self.complete_exploration()

    def calculate_map_coverage(self):
        """Calculate the percentage of explored area in the map"""
        if self.map_data is None:
            return
            
        try:
            # Count cells that are known (not -1)
            known_cells = sum(1 for cell in self.map_data.data if cell != -1)
            total_cells = len(self.map_data.data)
            self.map_coverage = known_cells / total_cells if total_cells > 0 else 0
            rospy.loginfo(f"Current map coverage: {self.map_coverage*100:.2f}%")
        except Exception as e:
            rospy.logerr(f"Error calculating map coverage: {e}")

    def save_map(self):
        """Save the occupancy grid map as .pgm and .yaml files"""
        if self.map_data is None:
            rospy.logwarn("No map data to save")
            return
       
        try:
            # Convert OccupancyGrid to numpy array
            width = self.map_data.info.width
            height = self.map_data.info.height
            map_img = np.array(self.map_data.data, dtype=np.int8).reshape((height, width))
           
            # Convert to grayscale (0: free, 100: occupied, -1: unknown)
            unknown_mask = (map_img == -1)
            map_img = np.clip(map_img, 0, 100)  # Clip occupied values
            map_img = (map_img * 2.55).astype(np.uint8)  # Scale to 0-255
            map_img[unknown_mask] = 205  # Gray for unknown
           
            # Save as .pgm
            timestamp = rospy.Time.now().to_sec()
            map_filename = os.path.join(self.map_save_path, f"exploration_map_{timestamp:.0f}.pgm")
            cv2.imwrite(map_filename, map_img)
           
            # Save .yaml metadata
            yaml_filename = os.path.join(self.map_save_path, f"exploration_map_{timestamp:.0f}.yaml")
            with open(yaml_filename, "w") as f:
                f.write(f"image: exploration_map_{timestamp:.0f}.pgm\n")
                f.write(f"resolution: {self.map_data.info.resolution}\n")
                f.write(f"origin: [{self.map_data.info.origin.position.x}, {self.map_data.info.origin.position.y}, 0]\n")
                f.write(f"negate: 0\n")
                f.write(f"occupied_thresh: 0.65\n")
                f.write(f"free_thresh: 0.196\n")
           
            rospy.loginfo(f"Map saved to {map_filename}")
        except Exception as e:
            rospy.logerr(f"Failed to save map: {e}")

    def get_robot_position(self):
        """Get the current robot position from TF"""
        try:
            self.tf_listener.waitForTransform("/map", "/base_link", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("/map", "/base_link", rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error: {e}")
            return None, None

    def start_initial_scan(self):
        """Begin the initial rotation scan"""
        self.exploration_state = "INITIAL_SCAN"
        self.initial_scan_start = rospy.Time.now()
        self.move("ROTATE_LEFT")
        rospy.loginfo("Starting initial scan rotation")

    def move(self, direction, speed_factor=1.0):
        """Send movement command to the robot only if different from current state"""
        if not self.running:
            return
            
        # Set lock to prevent concurrent movement commands
        self.movement_lock = True
           
        # Calculate speeds based on direction and speed factor
        current_max = int(self.max_speed * speed_factor)
        current_min = int(self.min_speed * speed_factor)
       
        if direction == "FORWARD":
            speeds = [current_max, current_max, current_max, current_max]
        elif direction == "FORWARD_LEFT":
            speeds = [current_min, current_max, current_min, current_max]
        elif direction == "FORWARD_RIGHT":
            speeds = [current_max, current_min, current_max, current_min]
        elif direction == "ROTATE_LEFT":
            speeds = [-current_max, current_max, -current_max, current_max]
        elif direction == "ROTATE_RIGHT":
            speeds = [current_max, -current_max, current_max, -current_max]
        elif direction == "STOP":
            speeds = [0, 0, 0, 0]
        else:
            speeds = [0, 0, 0, 0]
            direction = "STOP"
       
        # Only publish if command is different from current
        if direction != self.current_move_direction or speeds != self.current_motor_speeds:
            # Update state
            self.current_move_direction = direction
            self.current_motor_speeds = speeds
            
            # First send raw command
            cmd = f"M:{speeds[0]}:{speeds[1]}:{speeds[2]}:{speeds[3]}"
            self.cmd_pub.publish(String(cmd))
            
            # Then update direction state
            self.move_cmd_pub.publish(String(direction))
            
            rospy.loginfo(f"Published new movement command: {direction} - {speeds}")
        else:
            rospy.logdebug(f"Skipping redundant movement command: {direction}")
            
        # Release lock
        self.movement_lock = False

    def stop_robot(self):
        """Stop all motors if not already stopped"""
        self.move("STOP")

    def start_rotation(self):
        """Begin rotation to find a clear path"""
        direction = "ROTATE_LEFT" if random.random() > 0.5 else "ROTATE_RIGHT"
        self.move(direction)
        rospy.Timer(rospy.Duration(random.uniform(self.rotation_min_time, self.rotation_max_time)), 
                   self.stop_rotation, oneshot=True)

    def stop_rotation(self, event=None):
        """End rotation and resume scanning"""
        self.stop_robot()
        self.exploration_state = "SCANNING"

    def start_recovery_behavior(self):
        """Start recovery behavior after sensor timeout"""
        rospy.logwarn("Starting recovery behavior due to sensor timeout")
        self.stop_robot()
        
        # Wait a bit then try rotating to get new sensor data
        rospy.Timer(rospy.Duration(3.0), self.execute_recovery, oneshot=True)

    def execute_recovery(self, event=None):
        """Execute recovery action"""
        rospy.loginfo("Executing recovery - rotating to find valid sensor data")
        self.start_rotation()

    def complete_exploration(self):
        """Actions to take when exploration is complete"""
        self.stop_robot()
        self.exploration_state = "COMPLETE"
        self.save_map()  # Save final map
        rospy.loginfo("Exploration complete! Final map saved.")

        # Detener la detección de YOLO
        self.stop_yolo_detector()

        # Ejecutar el script de mapa simbólico
        rospy.loginfo("Launching symbolic map generator...")
        try:
            script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mapa_simbolico.py")
            if not os.path.exists(script_path):
                # Buscar en ubicaciones alternativas
                possible_paths = [
                    "/home/jetson/catkin_ws/src/barrier_map/scripts/mapa_simbolico.py",
                    os.path.expanduser("~/scripts/mapa_simbolico.py"),
                    "./mapa_simbolico.py"
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        script_path = path
                        break

            if os.path.exists(script_path):
                subprocess.Popen(["python3", script_path])
                rospy.loginfo(f"Symbolic map generator launched from {script_path}")
            else:
                rospy.logerr("Could not find mapa_simbolico.py script")
        except Exception as e:
            rospy.logerr(f"Error launching symbolic map generator: {e}")

    def run(self):
        """Main control loop using proper ROS rate control"""
        rate = rospy.Rate(10)  # 10Hz control loop
        
        try:
            while not rospy.is_shutdown() and self.running:
                # Periodically check map coverage
                if (rospy.Time.now() - self.last_map_check_time) > rospy.Duration(self.map_check_interval):
                    self.last_map_check_time = rospy.Time.now()
                    self.calculate_map_coverage()
                
                # Sleep to maintain rate
                rate.sleep()
        except rospy.ROSInterruptException:
            self.shutdown()
        finally:
            self.shutdown()
            
    def shutdown(self):
        """Clean shutdown"""
        self.running = False
        self.stop_robot()
        
        # Detener la detección de YOLO
        self.stop_yolo_detector()
        
        rospy.loginfo("Exploration node shutdown")


if __name__ == "__main__":
    try:
        explorer = LidarExplorerWithYOLO()
        explorer.run()
    except rospy.ROSInterruptException:
        pass
