#!/usr/bin/env python3

import rospy
import numpy as np
import os
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rtabmap_msgs.msg import MapData
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import tf
import random
import math
from geometry_msgs.msg import PoseStamped
from nav_msgs.srv import GetMap

class LidarExplorer:
    def __init__(self):
        rospy.init_node("lidar_explorer", anonymous=True)
       
        # Get parameters from the ROS parameter server for consistent configuration
        # Use global parameter namespace for shared parameters
        self.max_speed = rospy.get_param("/robot/motor/max_speed", 200)
        self.min_speed = rospy.get_param("/robot/motor/min_speed", 100)
        self.min_safe_distance = rospy.get_param("~min_safe_distance", 1.2)
        self.critical_distance = rospy.get_param("~critical_distance", 0.5)
        self.map_check_interval = rospy.get_param("~map_check_interval", 5)
        self.rotation_min_time = rospy.get_param("~rotation_min_time", 1.5)
        self.rotation_max_time = rospy.get_param("~rotation_max_time", 3.0)
        self.target_coverage = rospy.get_param("~target_coverage", 0.95)
        self.frontier_search_distance = rospy.get_param("~frontier_search_distance", 3.0)
        self.lidar_safety_angle = math.radians(30)  # 30 degrees cone in front
        self.scan_timeout = rospy.Duration(5.0)  # Timeout for LiDAR data
       
        # State variables
        self.running = True
        self.exploration_state = "INITIAL_SCAN"
        self.last_lidar_time = rospy.Time.now()
        self.laser_scan = None
        self.tf_listener = tf.TransformListener()
        self.initial_scan_start = None
        self.map_data = None  # Store the occupancy grid map
        self.map_save_path = os.path.expanduser("~/saved_maps")  # Directory to save maps
        os.makedirs(self.map_save_path, exist_ok=True)  # Create directory if it doesn't exist
        self.last_command_status = "NONE"  # Track command status
       
        # Publishers
        self.cmd_pub = rospy.Publisher("/robot/move/raw", String, queue_size=10)
        self.move_cmd_pub = rospy.Publisher("/robot/move/direction", String, queue_size=10)
       
        # Subscribers
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback)
        self.map_sub = rospy.Subscriber("/rtabmap/mapData", MapData, self.map_callback)
        self.grid_map_sub = rospy.Subscriber("/map", OccupancyGrid, self.grid_map_callback)
        # New subscriber to monitor motor command status
        self.motor_status_sub = rospy.Subscriber("/robot/motor/status", String, self.motor_status_callback)
       
        rospy.loginfo("Autonomous mapping explorer with LiDAR initialized")
        self.start_initial_scan()

    def motor_status_callback(self, msg):
        """Process motor command status feedback"""
        self.last_command_status = msg.data
        rospy.loginfo(f"Motor command status: {self.last_command_status}")

        # Handle errors or command failures
        if self.last_command_status == "COMMAND_FAILED":
            rospy.logwarn("Command failed to execute - retrying")
            # Implement retry logic if needed

    def laser_callback(self, msg):
        """Process LiDAR scan data"""
        self.laser_scan = msg
        self.last_lidar_time = rospy.Time.now()
        self.process_lidar_for_navigation()

    def process_lidar_for_navigation(self):
        """Analyze LiDAR data for obstacle avoidance and navigation"""
        if self.laser_scan is None:
            return
            
        # Check if LiDAR data is too old
        if (rospy.Time.now() - self.last_lidar_time) > self.scan_timeout:
            rospy.logwarn("LiDAR data timeout - stopping robot")
            self.stop_robot()
            return
            
        # Handle initial scan state
        if self.exploration_state == "INITIAL_SCAN":
            if (rospy.Time.now() - self.initial_scan_start) > rospy.Duration(10):
                self.stop_robot()
                self.exploration_state = "SCANNING"
                rospy.loginfo("Initial scan complete, beginning exploration")
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
                
        # If no obstacles detected, move forward
        if self.exploration_state == "SCANNING":
            self.move("FORWARD")

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
            # Convert OccupancyGrid to numpy array
            width = self.map_data.info.width
            height = self.map_data.info.height
            map_img = np.array(self.map_data.data, dtype=np.int8).reshape((height, width))
           
            # Convert to grayscale (0: free, 100: occupied, -1: unknown)
            map_img = np.clip(map_img, 0, 100)  # Treat unknowns as free space
            map_img = (map_img * 2.55).astype(np.uint8)  # Scale to 0-255
           
            # Save as .pgm
            map_filename = os.path.join(self.map_save_path, "exploration_map.pgm")
            from cv2 import imwrite
            imwrite(map_filename, map_img)
           
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

    def move(self, direction, speed_factor=1.0):
        """Send movement command to the robot"""
        if not self.running:
            return
           
        self.move_cmd_pub.publish(String(direction))
        
        # Calculate speeds with validation to ensure they stay within limits
        current_max = min(int(self.max_speed * speed_factor), self.max_speed)
        current_min = min(int(self.min_speed * speed_factor), self.max_speed)
       
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
       
        # Validate speeds before sending to ensure they are within limits
        for i in range(len(speeds)):
            if abs(speeds[i]) > self.max_speed:
                rospy.logwarn(f"Speed value {speeds[i]} exceeds maximum {self.max_speed}")
                speeds[i] = self.max_speed if speeds[i] > 0 else -self.max_speed
       
        cmd = f"M:{speeds[0]}:{speeds[1]}:{speeds[2]}:{speeds[3]}"
        self.cmd_pub.publish(String(cmd))
        rospy.loginfo(f"Published command: {cmd}")
        
        # Add mechanism to check for command success
        self.check_command_status(cmd)

    def check_command_status(self, cmd, max_checks=10, check_interval=0.1):
        """Check if command was successfully sent"""
        # Implement a mechanism to wait for command status feedback
        # This is a simple implementation that could be expanded
        count = 0
        while count < max_checks and self.last_command_status != "COMMAND_SENT":
            rospy.sleep(check_interval)
            count += 1
            
        if self.last_command_status != "COMMAND_SENT":
            rospy.logwarn(f"No command status received for: {cmd}")

    def stop_robot(self):
        """Stop all motors"""
        cmd = "M:0:0:0:0"
        self.cmd_pub.publish(String(cmd))
        rospy.loginfo("Published stop command")
        
        # Make sure stop command was received
        self.check_command_status(cmd)

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
    explorer = LidarExplorer()
    try:
        explorer.run()
    except rospy.ROSInterruptException:
        explorer.shutdown()