#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import os
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from rtabmap_msgs.msg import MapData
from nav_msgs.msg import OccupancyGrid, MapMetaData
from cv_bridge import CvBridge
import tf
import random
import math
from geometry_msgs.msg import PoseStamped
from nav_msgs.srv import GetMap

class DepthExplorer:
    def __init__(self):
        rospy.init_node("depth_explorer", anonymous=True)
       
        # Parameters
        self.max_speed = 200
        self.min_speed = 100
        self.min_safe_distance = rospy.get_param("~min_safe_distance", 1.2)
        self.critical_distance = rospy.get_param("~critical_distance", 0.5)
        self.map_check_interval = rospy.get_param("~map_check_interval", 5)
        self.rotation_min_time = rospy.get_param("~rotation_min_time", 1.5)
        self.rotation_max_time = rospy.get_param("~rotation_max_time", 3.0)
        self.target_coverage = rospy.get_param("~target_coverage", 0.95)
        self.frontier_search_distance = rospy.get_param("~frontier_search_distance", 3.0)
        self.lidar_safety_angle = math.radians(30)  # 30 degrees cone in front
       
        # State variables
        self.running = True
        self.exploration_state = "INITIAL_SCAN"
        self.bridge = CvBridge()
        self.last_depth_time = rospy.Time.now()
        self.depth_image = None
        self.laser_scan = None
        self.tf_listener = tf.TransformListener()
        self.initial_scan_start = None
        self.map_data = None  # Store the occupancy grid map
        self.map_save_path = os.path.expanduser("~/saved_maps")  # Directory to save maps
        os.makedirs(self.map_save_path, exist_ok=True)  # Create directory if it doesn't exist
       
        # Publishers
        self.cmd_pub = rospy.Publisher("/robot/move/raw", String, queue_size=10)
        self.move_cmd_pub = rospy.Publisher("/robot/move/direction", String, queue_size=10)
       
        # Subscribers
        self.depth_sub = rospy.Subscriber("/k4a/depth/image_resized", Image, self.depth_callback)
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback)
        self.map_sub = rospy.Subscriber("/rtabmap/mapData", MapData, self.map_callback)
        self.grid_map_sub = rospy.Subscriber("/map", OccupancyGrid, self.grid_map_callback)
       
        rospy.loginfo("Autonomous mapping explorer with LiDAR initialized")
        self.start_initial_scan()

    def laser_callback(self, msg):
        """Process LiDAR scan data"""
        self.laser_scan = msg
        self.process_lidar_for_navigation()

    def process_lidar_for_navigation(self):
        """Analyze LiDAR data for obstacle avoidance"""
        if self.laser_scan is None or self.exploration_state != "SCANNING":
            return
           
        # Get the angle ranges we care about (front cone)
        angle_min = self.laser_scan.angle_min
        angle_increment = self.laser_scan.angle_increment
        num_readings = len(self.laser_scan.ranges)
       
        # Calculate indices for the front cone
        center_idx = num_readings // 2
        angle_range = int(self.lidar_safety_angle / angle_increment)
        start_idx = max(0, center_idx - angle_range)
        end_idx = min(num_readings, center_idx + angle_range)
       
        # Get the ranges in the front cone
        front_ranges = self.laser_scan.ranges[start_idx:end_idx]
        valid_ranges = [r for r in front_ranges if self.laser_scan.range_min < r < self.laser_scan.range_max]
       
        if valid_ranges:
            min_distance = min(valid_ranges)
            if min_distance < self.critical_distance:
                rospy.logwarn(f"LiDAR detected critical obstacle at {min_distance:.2f}m - EMERGENCY STOP")
                self.stop_robot()
                self.start_rotation()
                return
            elif min_distance < self.min_safe_distance:
                rospy.loginfo(f"LiDAR detected close obstacle at {min_distance:.2f}m - adjusting path")
                self.start_rotation()
                return

    def map_callback(self, msg):
        """Process RTAB-Map data (optional for advanced mapping)"""
        pass  # Add RTAB-Map processing if needed

    def grid_map_callback(self, msg):
        """Process and store the occupancy grid map"""
        self.map_data = msg
        rospy.loginfo("OccupancyGrid map received and stored")
       
        # Example: Save map periodically or based on conditions
        if self.exploration_state == "SCANNING":
            self.save_map()

    def save_map(self):
        """Save the occupancy grid map as .pgm and .yaml files"""
        if self.map_data is None:
            rospy.logwarn("No map data to save")
            return
       
        try:
            # Convert OccupancyGrid to OpenCV image
            width = self.map_data.info.width
            height = self.map_data.info.height
            map_img = np.array(self.map_data.data, dtype=np.int8).reshape((height, width))
           
            # Convert to grayscale (0: free, 100: occupied, -1: unknown)
            map_img = np.clip(map_img, 0, 100)  # Treat unknowns as free space
            map_img = (map_img * 2.55).astype(np.uint8)  # Scale to 0-255
           
            # Save as .pgm
            map_filename = os.path.join(self.map_save_path, "exploration_map.pgm")
            cv2.imwrite(map_filename, map_img)
           
            # Save .yaml metadata
            yaml_filename = os.path.join(self.map_save_path, "exploration_map.yaml")
            with open(yaml_filename, "w") as f:
                f.write(f"image: exploration_map.pgm\n")
                f.write(f"resolution: {self.map_data.info.resolution}\n")
                f.write(f"origin: [{self.map_data.info.origin.position.x}, {self.map_data.info.origin.position.y}, 0]\n")
                f.write(f"negate: 0\n")
                f.write(f"occupied_thresh: 0.65\n")
                f.write(f"free_thresh: 0.196\n")
           
            rospy.loginfo(f"Map saved to {self.map_save_path}")
        except Exception as e:
            rospy.logerr(f"Failed to save map: {e}")

    def start_initial_scan(self):
        """Begin the initial rotation scan"""
        self.exploration_state = "INITIAL_SCAN"
        self.initial_scan_start = rospy.Time.now()
        self.move("ROTATE_LEFT")
        rospy.loginfo("Starting initial scan rotation")

    def depth_callback(self, msg):
        """Process depth image for navigation decisions"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            self.last_depth_time = rospy.Time.now()
           
            if self.exploration_state == "INITIAL_SCAN":
                if (rospy.Time.now() - self.initial_scan_start) > rospy.Duration(10):
                    self.stop_robot()
                    self.exploration_state = "SCANNING"
                    rospy.loginfo("Initial scan complete, beginning exploration")
            elif self.exploration_state == "SCANNING":
                self.process_depth_for_navigation()
               
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")

    def process_depth_for_navigation(self):
        """Analyze depth image to make navigation decisions"""
        if self.depth_image is None:
            return
           
        height, width = self.depth_image.shape
        center_roi = self.depth_image[height//3:2*height//3, width//3:2*width//3]
       
        # Create depth masks (valid values between 0.1m and 10m)
        min_valid = 100   # 0.1m in mm
        max_valid = 10000 # 10m in mm
        center_mask = (center_roi > min_valid) & (center_roi < max_valid)
       
        # Emergency stop check
        if np.sum(center_mask) > 0:
            min_center_dist = np.min(center_roi[center_mask]) / 1000.0
            if min_center_dist < self.critical_distance:
                rospy.logwarn(f"CRITICAL OBSTACLE at {min_center_dist:.2f}m - EMERGENCY STOP")
                self.stop_robot()
                self.start_rotation()
                return
       
        # Normal movement (only if LiDAR didn't detect obstacles)
        if self.laser_scan is None or not self.is_lidar_blocked():
            self.move("FORWARD")

    def is_lidar_blocked(self):
        """Check if LiDAR detects obstacles in the path"""
        if self.laser_scan is None:
            return False
           
        angle_min = self.laser_scan.angle_min
        angle_increment = self.laser_scan.angle_increment
        num_readings = len(self.laser_scan.ranges)
       
        center_idx = num_readings // 2
        angle_range = int(self.lidar_safety_angle / angle_increment)
        start_idx = max(0, center_idx - angle_range)
        end_idx = min(num_readings, center_idx + angle_range)
       
        front_ranges = self.laser_scan.ranges[start_idx:end_idx]
        valid_ranges = [r for r in front_ranges if self.laser_scan.range_min < r < self.laser_scan.range_max]
       
        if valid_ranges:
            min_distance = min(valid_ranges)
            return min_distance < self.min_safe_distance
        return False


    def move(self, direction, speed_factor=1.0):
        """Send movement command to the robot"""
        if not self.running:
            return
           
        self.move_cmd_pub.publish(String(direction))
        current_max = int(self.max_speed * speed_factor)
        current_min = int(self.min_speed * speed_factor)
       
        if direction == "FORWARD":
            speeds = [current_max, current_max, current_max, current_max]
        elif direction == "FORWARD_LEFT":
            speeds = [current_min, current_max, current_min, current_max]
        elif direction == "FORWARD_RIGHT":
            speeds = [current_max, current_min, current_max, current_min]
        elif direction == "ROTATE_LEFT":
            speeds = [-current_max, -current_max, current_max, current_max]
        elif direction == "ROTATE_RIGHT":
            speeds = [current_max, current_max, -current_max, -current_max]
        else:
            speeds = [0, 0, 0, 0]
       
        cmd = f"M:{speeds[0]}:{speeds[1]}:{speeds[2]}:{speeds[3]}"
        self.cmd_pub.publish(String(cmd))
        rospy.loginfo(f"Published command: {cmd}")

    def stop_robot(self):
        """Stop all motors"""
        self.cmd_pub.publish(String("M:0:0:0:0"))
        rospy.loginfo("Published stop command")

    def start_rotation(self):
        """Begin rotation to find a clear path"""
        direction = "ROTATE_LEFT" if random.random() > 0.5 else "ROTATE_RIGHT"
        self.move(direction)
        rospy.Timer(rospy.Duration(random.uniform(self.rotation_min_time, self.rotation_max_time)), self.stop_rotation, oneshot=True)

    def stop_rotation(self, event=None):
        """End rotation and resume scanning"""
        self.stop_robot()
        self.exploration_state = "SCANNING"

    def run(self):
        """Main control loop"""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.running:
            rate.sleep()

    def shutdown(self):
        """Clean shutdown"""
        self.running = False
        self.stop_robot()
        rospy.loginfo("Exploration node shutdown")


if __name__ == "__main__":
    explorer = DepthExplorer()
    try:
        explorer.run()
    except rospy.ROSInterruptException:
        explorer.shutdown()
