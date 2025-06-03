#!/usr/bin/env python3
import rospy
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class DepthResizer:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = None
        self.processing = False
       
        # Configurable parameters with safer defaults
        self.target_width = rospy.get_param('~width', 320)  # Reduced further
        self.target_height = rospy.get_param('~height', 180)
        self.max_rate = rospy.get_param('~rate', 3.0)  # Lower rate
        self.input_topic = rospy.get_param('~input_topic', '/k4a/depth/image_raw')
        self.output_topic = rospy.get_param('~output_topic', '/k4a/depth/image_resized')
       
        rospy.loginfo(f"Depth Resizer configured for: {self.target_width}x{self.target_height} at {self.max_rate}Hz")

    def image_callback(self, msg):
        if self.processing:
            return
           
        self.processing = True
        try:
            # Rate limiting
            if hasattr(self, 'last_time'):
                if (rospy.Time.now() - self.last_time).to_sec() < (1.0/self.max_rate):
                    return
           
            # Convert depth image
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            except CvBridgeError as e:
                rospy.logerr(f"CV Bridge error: {e}")
                return
               
            # Validate image
            if cv_image is None:
                rospy.logwarn("Received empty image")
                return
               
            # Resize with nearest neighbor for depth data
            resized_image = cv2.resize(
                cv_image,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_NEAREST
            )
           
            # Convert back to ROS message
            try:
                resized_msg = self.bridge.cv2_to_imgmsg(resized_image, encoding="passthrough")
                resized_msg.header = msg.header  # Preserve original header
                self.pub.publish(resized_msg)
            except CvBridgeError as e:
                rospy.logerr(f"CV Bridge output error: {e}")
           
            self.last_time = rospy.Time.now()
           
        except Exception as e:
            rospy.logerr(f"Unexpected error: {str(e)}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            rospy.logerr(f"Exception at line {exc_tb.tb_lineno}")
        finally:
            self.processing = False

    def run(self):
        rospy.init_node('depth_resizer_node', anonymous=True)
       
        # Wait for the input topic to become available
        rospy.loginfo(f"Waiting for topic {self.input_topic}...")
        try:
            rospy.wait_for_message(self.input_topic, Image, timeout=30.0)
        except rospy.ROSException:
            rospy.logerr(f"Timeout waiting for topic {self.input_topic}")
            sys.exit(1)
           
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        rospy.Subscriber(self.input_topic, Image, self.image_callback, queue_size=1)
       
        rospy.loginfo("Depth resizer node ready")
        rospy.spin()

if __name__ == '__main__':
    try:
        node = DepthResizer()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Node initialization failed: {str(e)}")
        sys.exit(1)

