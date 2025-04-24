#!/usr/bin/env python3

import rospy
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import os
import sys
import math
from sklearn.cluster import DBSCAN  # Para agrupar detecciones similares


class ObjectMapGenerator:
    def __init__(self):
        rospy.init_node("object_map_generator", anonymous=True)

        # Parameters
        self.map_save_path = rospy.get_param("~map_save_path", "~/saved_maps")
        self.map_save_path = os.path.expanduser(self.map_save_path)
        self.yolo_detections_path = rospy.get_param("~yolo_detections_path", "~/saved_maps/yolo_detections")
        self.yolo_detections_path = os.path.expanduser(self.yolo_detections_path)
        self.output_path = rospy.get_param("~output_path", "~/annotated_maps")
        self.output_path = os.path.expanduser(self.output_path)
        self.object_marker_size = rospy.get_param("~object_marker_size", 0.5)  # in meters
        self.cluster_distance = rospy.get_param("~cluster_distance", 0.5)  # for DBSCAN clustering
        self.min_detections = rospy.get_param("~min_detections", 1)  # min detections to consider object valid

        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)

        # TF listener for coordinate transformations
        self.tf_listener = tf.TransformListener()

        # Wait for the final map
        rospy.loginfo("Waiting for final map data...")
        self.final_map = None
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)

        # Wait until we have the map
        timeout = rospy.get_param("~map_wait_timeout", 60)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown() and self.final_map is None:
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logerr(f"Timeout waiting for map data after {timeout} seconds")
                sys.exit(1)
            rospy.sleep(0.1)

        # Process the data
        self.generate_object_map()

    def map_callback(self, msg):
        """Store the final map data"""
        if self.final_map is None:  # Only store the first map we receive
            self.final_map = msg
            rospy.loginfo("Received final map data")

    def load_yolo_detections(self):
        """Load all YOLO detection metadata files"""
        detections = []
        try:
            detection_files = list(Path(self.yolo_detections_path).glob("metadata_*.json"))

            if not detection_files:
                rospy.logwarn(f"No YOLO detection files found in {self.yolo_detections_path}")
                return detections

            for file_path in detection_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        detections.append(data)
                except Exception as e:
                    rospy.logerr(f"Error loading detection file {file_path}: {e}")

            rospy.loginfo(f"Loaded {len(detections)} YOLO detection sets")
        except Exception as e:
            rospy.logerr(f"Error searching for detection files: {e}")

        return detections

    def transform_point_to_map(self, image_point, image_size, capture_time):
        """
        Transform a point from image coordinates to map coordinates
        Args:
            image_point: (x,y) in image pixels
            image_size: (width, height) of the image
            capture_time: ROS Time when the image was taken
        Returns:
            (x, y) in map coordinates or None if transform not available
        """
        try:
            # Get camera position in map frame at capture time
            self.tf_listener.waitForTransform(
                "/map", "/base_link", capture_time, rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform(
                "/map", "/base_link", capture_time)

            # Calculate camera orientation
            quaternion = rot
            euler = tf.transformations.euler_from_quaternion(quaternion)
            yaw = euler[2]  # Rotation around Z axis (assuming camera points forward)

            # Normalize image coordinates to [-1, 1]
            img_width, img_height = image_size
            x_normalized = (image_point[0] - img_width / 2) / (img_width / 2)
            y_normalized = (img_height / 2 - image_point[1]) / (img_height / 2)

            # Camera parameters (field of view) - these should be properly calibrated
            # for your specific camera
            fov_h = math.radians(60)  # horizontal field of view
            fov_v = math.radians(45)  # vertical field of view

            # Calculate angles
            angle_h = x_normalized * (fov_h / 2)
            angle_v = y_normalized * (fov_v / 2)

            # Estimate distance (this is simplified and needs proper depth estimation)
            # For this example, we'll use a fixed distance based on object size
            estimated_distance = 2.0  # meters, should be replaced with actual depth

            # Project to 3D space
            # Note: This is a simplified projection. For accurate results, use proper camera calibration
            # and perspective projection
            forward_distance = estimated_distance * math.cos(angle_h) * math.cos(angle_v)
            side_distance = estimated_distance * math.sin(angle_h)

            # Transform to map coordinates using robot pose and orientation
            map_x = trans[0] + forward_distance * math.cos(yaw) - side_distance * math.sin(yaw)
            map_y = trans[1] + forward_distance * math.sin(yaw) + side_distance * math.cos(yaw)

            return (map_x, map_y)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error during point transform: {e}")
            return None

    def cluster_object_detections(self, positions_by_class):
        """Cluster similar object detections to remove duplicates"""
        clustered_positions = {}

        for class_name, positions in positions_by_class.items():
            if len(positions) < 2:
                clustered_positions[class_name] = positions
                continue

            # Convert positions to numpy array for clustering
            positions_array = np.array(positions)

            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=self.cluster_distance, min_samples=1).fit(positions_array)
            labels = clustering.labels_

            # Group positions by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(positions[i])

            # Average positions within each cluster
            clustered_positions[class_name] = []
            for cluster_positions in clusters.values():
                if len(cluster_positions) >= self.min_detections:
                    avg_x = sum(p[0] for p in cluster_positions) / len(cluster_positions)
                    avg_y = sum(p[1] for p in cluster_positions) / len(cluster_positions)
                    clustered_positions[class_name].append((avg_x, avg_y))

        return clustered_positions

    def generate_object_map(self):
        """Main function to generate the object map"""
        try:
            # Load the final map
            map_img = self.process_map_image(self.final_map)

            # Load YOLO detections
            detections = self.load_yolo_detections()

            if not detections:
                rospy.logwarn("No detections found, generating empty object map")

            # Process each detection set
            object_positions = {}  # class: [(x,y), ...]

            for detection_set in detections:
                try:
                    # Parse timestamp from filename
                    timestamp_str = detection_set['timestamp']
                    capture_time = rospy.Time.from_sec(
                        datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S.%f").timestamp())

                    image_size = detection_set['camera_parameters']['resolution']

                    for detection in detection_set['detections']:
                        class_name = detection['class']
                        center_px = detection['center']
                        confidence = detection.get('confidence', 0)

                        # Skip low confidence detections
                        if confidence < 0.4:  # adjustable threshold
                            continue

                        # Transform to map coordinates
                        map_point = self.transform_point_to_map(
                            center_px, image_size, capture_time)

                        if map_point is not None:
                            if class_name not in object_positions:
                                object_positions[class_name] = []
                            object_positions[class_name].append(map_point)

                except Exception as e:
                    rospy.logerr(f"Error processing detection set: {e}")
                    continue

            # Cluster object detections to remove duplicates
            clustered_objects = self.cluster_object_detections(object_positions)

            # Create the annotated map
            self.create_annotated_map(map_img, clustered_objects)

        except Exception as e:
            rospy.logerr(f"Error in generate_object_map: {e}")

    def process_map_image(self, map_msg):
        """Convert OccupancyGrid to a displayable image"""
        width = map_msg.info.width
        height = map_msg.info.height

        # Convert map data to numpy array
        map_data = np.array(map_msg.data, dtype=np.int8).reshape((height, width))

        # Create color image
        map_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Color mapping:
        # -1: unknown (gray)
        # 0: free (white)
        # 100: occupied (black)
        map_img[map_data == -1] = [128, 128, 128]  # Gray for unknown
        map_img[map_data == 0] = [255, 255, 255]  # White for free
        map_img[map_data == 100] = [0, 0, 0]  # Black for occupied

        return map_img

    def create_annotated_map(self, map_img, object_positions):
        """Create the final annotated map with object markers"""
        # Create a copy of the map for annotation
        annotated_map = map_img.copy()

        # Define colors for different object classes
        class_colors = {
            'person': (255, 0, 0),  # Red
            'car': (0, 0, 255),  # Blue
            'chair': (0, 255, 0),  # Green
            'dog': (255, 165, 0),  # Orange
            'default': (255, 0, 255)  # Magenta for unknown classes
        }

        # Draw each object class
        for class_name, positions in object_positions.items():
            color = class_colors.get(class_name.lower(), class_colors['default'])

            for (x, y) in positions:
                try:
                    # Convert map coordinates to pixel coordinates
                    map_info = self.final_map.info
                    px = int((x - map_info.origin.position.x) / map_info.resolution)
                    py = int((y - map_info.origin.position.y) / map_info.resolution)

                    # Check if within image bounds
                    if 0 <= px < map_img.shape[1] and 0 <= py < map_img.shape[0]:
                        # Draw a circle for the object
                        radius = int(self.object_marker_size / map_info.resolution)
                        cv2.circle(annotated_map, (px, py), radius, color, -1)

                        # Add class label
                        font_scale = 0.5
                        thickness = 1
                        text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                        text_x = px - text_size[0] // 2
                        text_y = py - radius - 5

                        # Ensure text is within image bounds
                        if 0 <= text_x < map_img.shape[1] and 0 <= text_y < map_img.shape[0]:
                            cv2.putText(annotated_map, class_name, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
                            cv2.putText(annotated_map, class_name, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                except Exception as e:
                    rospy.logerr(f"Error drawing object on map: {e}")

        # Save the annotated map
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_path, f"object_map_{timestamp}.png")
        cv2.imwrite(output_file, annotated_map)

        rospy.loginfo(f"Saved annotated object map to {output_file}")

        # Also save the object positions as JSON
        object_data = {
            'map_resolution': self.final_map.info.resolution,
            'map_origin': {
                'x': self.final_map.info.origin.position.x,
                'y': self.final_map.info.origin.position.y
            },
            'objects': {class_name: [list(pos) for pos in positions]
                        for class_name, positions in object_positions.items()}
        }

        json_file = os.path.join(self.output_path, f"object_positions_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(object_data, f, indent=2)

        rospy.loginfo(f"Saved object position data to {json_file}")

        # Generate a visualization with matplotlib for better quality
        self.generate_matplotlib_visualization(object_positions, timestamp)

    def generate_matplotlib_visualization(self, object_positions, timestamp):
        """Generate a high-quality visualization using matplotlib"""
        try:
            # Create a figure
            fig, ax = plt.subplots(figsize=(12, 12))

            # Get map dimensions
            map_width = self.final_map.info.width * self.final_map.info.resolution
            map_height = self.final_map.info.height * self.final_map.info.resolution
            origin_x = self.final_map.info.origin.position.x
            origin_y = self.final_map.info.origin.position.y

            # Set plot limits
            ax.set_xlim(origin_x, origin_x + map_width)
            ax.set_ylim(origin_y, origin_y + map_height)

            # Plot objects
            for class_name, positions in object_positions.items():
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]

                # Different colors for different classes
                if class_name.lower() == 'person':
                    color = 'red'
                elif class_name.lower() == 'car':
                    color = 'blue'
                elif class_name.lower() == 'chair':
                    color = 'green'
                elif class_name.lower() == 'dog':
                    color = 'orange'
                else:
                    color = 'magenta'

                # Plot each position
                for x, y in zip(x_coords, y_coords):
                    circle = Circle((x, y), self.object_marker_size / 2, color=color, alpha=0.7)
                    ax.add_patch(circle)
                    ax.text(x, y + self.object_marker_size / 2 + 0.1, class_name,
                            ha='center', va='bottom', fontsize=9)

            # Add title and labels
            ax.set_title('Object Map Visualization')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.grid(True)

            # Save the figure
            output_file = os.path.join(self.output_path, f"object_map_viz_{timestamp}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            rospy.loginfo(f"Saved matplotlib visualization to {output_file}")
        except Exception as e:
            rospy.logerr(f"Error generating matplotlib visualization: {e}")


def main():
    try:
        generator = ObjectMapGenerator()
        rospy.loginfo("Object map generation complete.")
    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception in object map generator.")
    except Exception as e:
        rospy.logerr(f"Error in object map generator: {e}")


if __name__ == '__main__':
    main()