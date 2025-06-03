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
import cv2  # Importación explícita de cv2

class LidarExplorer:
    def __init__(self):
        rospy.init_node("lidar_explorer", anonymous=True)
       
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
        self.scan_timeout = rospy.Duration(5.0)  # Timeout for LiDAR data
        self.recovery_timer = None  # Timer for recovery actions
       
        # State variables
        self.running = True
        self.exploration_state = "INITIAL_SCAN"
        self.last_lidar_time = rospy.Time.now()
        self.laser_scan = None
        self.tf_listener = tf.TransformListener()
        self.initial_scan_start = None
        self.map_data = None  # Store the occupancy grid map
        self.map_save_path = os.path.expanduser("~/saved_maps")  # Directory to save maps
        self.last_map_check_time = rospy.Time.now()
        self.map_coverage = 0.0  # Current map coverage percentage
        os.makedirs(self.map_save_path, exist_ok=True)  # Create directory if it doesn't exist
       
        # Publishers
        self.cmd_pub = rospy.Publisher("/robot/move/raw", String, queue_size=10)
        self.move_cmd_pub = rospy.Publisher("/robot/move/direction", String, queue_size=10)
       
        # Subscribers
        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.laser_callback)
        self.map_sub = rospy.Subscriber("/rtabmap/mapData", MapData, self.map_callback)
        self.grid_map_sub = rospy.Subscriber("/map", OccupancyGrid, self.grid_map_callback)
       
        rospy.loginfo("Autonomous mapping explorer with LiDAR initialized")
        self.start_initial_scan()

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
            self.start_recovery_behavior()
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
            # Check if we should search for a frontier
            if self.should_find_frontier():
                self.find_and_navigate_to_frontier()
            else:
                self.move("FORWARD")

    def should_find_frontier(self):
        """Determine if we should search for a frontier based on time or conditions"""
        # Every 20 seconds, look for frontiers
        return (rospy.Time.now() - self.last_map_check_time) > rospy.Duration(20)

    def find_and_navigate_to_frontier(self):
        """Find exploration frontiers and navigate towards them"""
        self.last_map_check_time = rospy.Time.now()
        
        if self.map_data is None:
            rospy.logwarn("No map data available for frontier detection")
            return

        # Very basic frontier detection - in real implementation this would be more sophisticated
        # A frontier is typically the boundary between explored and unexplored space
        try:
            # Convert map to CV image
            width = self.map_data.info.width
            height = self.map_data.info.height
            map_img = np.array(self.map_data.data, dtype=np.int8).reshape((height, width))
            
            # In a real implementation, you would:
            # 1. Find boundaries between known and unknown areas
            # 2. Cluster them into frontiers
            # 3. Select the best frontier based on distance, size, etc.
            
            # For now we'll just rotate occasionally to explore
            rospy.loginfo("Looking for new exploration directions")
            self.start_rotation()
            
        except Exception as e:
            rospy.logerr(f"Error in frontier detection: {e}")

    def map_callback(self, msg):
        """Process RTAB-Map data"""
        # Store data that might be useful for advanced mapping
        # In a real implementation, you'd process 3D map data here
        rospy.logdebug("Received RTAB-Map data")

    def grid_map_callback(self, msg):
        """Process and store the occupancy grid map"""
        self.map_data = msg
        rospy.loginfo("OccupancyGrid map received and stored")
        
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
            # Handle unknown values (-1) separately
            unknown_mask = (map_img == -1)
            map_img = np.clip(map_img, 0, 100)  # Clip occupied values
            map_img = (map_img * 2.55).astype(np.uint8)  # Scale to 0-255
            map_img[unknown_mask] = 205  # Gray for unknown (value chosen to be visible but distinct)
           
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
        """Send movement command to the robot"""
        if not self.running:
            return
           
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
        else:
            speeds = [0, 0, 0, 0]
       
        # First send raw command for immediate response
        cmd = f"M:{speeds[0]}:{speeds[1]}:{speeds[2]}:{speeds[3]}"
        self.cmd_pub.publish(String(cmd))
        
        # Then update direction state
        self.move_cmd_pub.publish(String(direction))
        
        rospy.loginfo(f"Published command: {cmd} ({direction})")

    def stop_robot(self):
        """Stop all motors"""
        self.cmd_pub.publish(String("M:0:0:0:0"))
        self.move_cmd_pub.publish(String("STOP"))
        rospy.loginfo("Published stop command")

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
        if self.recovery_timer is None or not self.recovery_timer.is_alive():
            self.recovery_timer = rospy.Timer(rospy.Duration(3.0), self.execute_recovery, oneshot=True)

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

    def run(self):
        """Main control loop"""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and self.running:
            # Periodically check map coverage
            if (rospy.Time.now() - self.last_map_check_time) > rospy.Duration(self.map_check_interval):
                self.last_map_check_time = rospy.Time.now()
                self.calculate_map_coverage()
            
            rate.sleep()

    def shutdown(self):
        """Clean shutdown"""
        self.running = False
        self.stop_robot()
        if self.recovery_timer is not None and self.recovery_timer.is_alive():
            self.recovery_timer.shutdown()
        rospy.loginfo("Exploration node shutdown")


if __name__ == "__main__":
    explorer = LidarExplorer()
    try:
        explorer.run()
    except rospy.ROSInterruptException:
        explorer.shutdown()