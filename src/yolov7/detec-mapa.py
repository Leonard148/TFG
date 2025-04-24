#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import json
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
import tf
import yaml
import torch  # Para YOLO con PyTorch
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

class ObjectDetectionMapper:
    def __init__(self):
        rospy.init_node('object_detection_mapper', anonymous=True)
        
        # Parámetros
        self.detection_interval = rospy.get_param('~detection_interval', 2.0)  # Segundos entre detecciones
        self.data_path = os.path.expanduser(rospy.get_param('~data_path', '~/detected_objects'))
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_rgb_optical_frame')
        self.yolo_conf_threshold = rospy.get_param('~yolo_conf_threshold', 0.5)
        self.max_depth = rospy.get_param('~max_depth', 5.0)  # Profundidad máxima en metros
        
        # Crear directorios para guardar datos
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(os.path.join(self.data_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, 'detections'), exist_ok=True)
        
        # Cargar modelo YOLO (usando YOLOv7 con PyTorch)
        model_path = rospy.get_param('~yolo_model_path', 'yolov7s.pt')
        self.model = torch.hub.load('/yolov7', 'custom', '/runs/exp_custom17/weights/best.pt', force_reload=True)
        self.model.conf = self.yolo_conf_threshold
        
        # Inicializar el bridge para convertir imágenes
        self.bridge = CvBridge()
        
        # TF para transformaciones entre marcos de referencia
        self.tf_listener = tf.TransformListener()
        
        # Variables para almacenar datos
        self.last_detection_time = rospy.Time.now()
        self.current_map = None
        self.current_map_time = None
        self.camera_info = None
        self.depth_image = None
        
        # Publicadores
        self.detection_pub = rospy.Publisher('/object_detections', String, queue_size=10)
        self.markers_pub = rospy.Publisher('/detected_objects_markers', MarkerArray, queue_size=10)
        
        # Suscriptores
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        self.image_sub = rospy.Subscriber('/k4a/rgb/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/k4a/depth/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/k4a/rgb/camera_info', CameraInfo, self.camera_info_callback)
        
        rospy.loginfo("Object Detection Mapper initialized")
    
    def map_callback(self, msg):
        """Almacena el mapa actual y su timestamp"""
        self.current_map = msg
        self.current_map_time = msg.header.stamp
        rospy.loginfo("New map received at time: %f", self.current_map_time.to_sec())
    
    def camera_info_callback(self, msg):
        """Almacena la información de la cámara"""
        self.camera_info = msg
    
    def depth_callback(self, msg):
        """Almacena la imagen de profundidad"""
        self.depth_image = msg
    
    def image_callback(self, msg):
        """Procesa imágenes de la cámara y ejecuta detección si es el momento"""
        # Verificar si es hora de hacer una nueva detección
        now = rospy.Time.now()
        if (now - self.last_detection_time).to_sec() > self.detection_interval:
            self.last_detection_time = now
            self.process_image(msg)
    
    def process_image(self, msg):
        """Ejecuta YOLO en la imagen y guarda los resultados"""
        if self.current_map is None:
            rospy.logwarn("No map available for object detection")
            return
            
        try:
            # Convertir imagen ROS a formato OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Ejecutar detección YOLO
            results = self.model(cv_image)
            detections = results.pandas().xyxy[0]  # Obtener bounding boxes
            
            if len(detections) == 0:
                rospy.loginfo("No objects detected in this frame")
                return
                
            # Guardar imagen con timestamp
            img_time = msg.header.stamp.to_sec()
            img_filename = os.path.join(self.data_path, 'images', f"image_{img_time:.6f}.jpg")
            cv2.imwrite(img_filename, cv_image)
            
            # Preparar datos para guardar
            objects_data = []
            markers_array = MarkerArray()
            marker_id = 0
            
            # Para cada detección
            for _, detection in detections.iterrows():
                class_name = detection['name']
                confidence = detection['confidence']
                bbox = [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']]
                
                # Calcular centro del objeto en la imagen
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                # Obtener profundidad si está disponible
                object_distance = None
                if self.depth_image is not None:
                    try:
                        depth_cv = self.bridge.imgmsg_to_cv2(self.depth_image, "32FC1")
                        # Obtener un promedio de profundidad en el área del objeto
                        depth_roi = depth_cv[max(0, center_y-5):min(depth_cv.shape[0], center_y+5), 
                                            max(0, center_x-5):min(depth_cv.shape[1], center_x+5)]
                        # Filtrar valores nan e inf
                        valid_depths = depth_roi[np.isfinite(depth_roi)]
                        if len(valid_depths) > 0:
                            object_distance = np.median(valid_depths)
                            if object_distance > self.max_depth:
                                object_distance = None
                    except Exception as e:
                        rospy.logwarn(f"Error processing depth data: {e}")
                
                # Proyectar punto 2D a 3D si tenemos profundidad y camera_info
                map_position = None
                if object_distance is not None and self.camera_info is not None:
                    # Obtener parámetros de la cámara
                    fx = self.camera_info.K[0]
                    fy = self.camera_info.K[4]
                    cx = self.camera_info.K[2]
                    cy = self.camera_info.K[5]
                    
                    # Proyección de 2D a 3D (en el marco de la cámara)
                    x = (center_x - cx) * object_distance / fx
                    y = (center_y - cy) * object_distance / fy
                    z = object_distance
                    
                    # Crear punto en el marco de la cámara
                    camera_point = PoseStamped()
                    camera_point.header.frame_id = self.camera_frame
                    camera_point.header.stamp = msg.header.stamp
                    camera_point.pose.position = Point(z, -x, -y)  # Ajuste para el marco de coordenadas ROS
                    camera_point.pose.orientation = Quaternion(0, 0, 0, 1)
                    
                    # Transformar al marco del mapa
                    try:
                        self.tf_listener.waitForTransform(self.map_frame, self.camera_frame, 
                                                         msg.header.stamp, rospy.Duration(0.5))
                        map_point = self.tf_listener.transformPose(self.map_frame, camera_point)
                        map_position = [
                            map_point.pose.position.x,
                            map_point.pose.position.y,
                            map_point.pose.position.z
                        ]
                        
                        # Crear marcador para RViz
                        marker = Marker()
                        marker.header.frame_id = self.map_frame
                        marker.header.stamp = rospy.Time.now()
                        marker.ns = "detected_objects"
                        marker.id = marker_id
                        marker.type = Marker.CUBE
                        marker.action = Marker.ADD
                        marker.pose = map_point.pose
                        marker.scale.x = 0.2
                        marker.scale.y = 0.2
                        marker.scale.z = 0.2
                        marker.color.r = 1.0 if class_name == "person" else 0.2
                        marker.color.g = 0.2 if class_name == "person" else 1.0
                        marker.color.b = 0.2
                        marker.color.a = 0.8
                        marker.lifetime = rospy.Duration(30)  # Duración del marcador: 30 segundos
                        marker.text = f"{class_name}: {confidence:.2f}"
                        
                        markers_array.markers.append(marker)
                        marker_id += 1
                        
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                        rospy.logwarn(f"TF Error: {e}")
                
                # Guardar datos del objeto
                object_data = {
                    "class": class_name,
                    "confidence": float(confidence),
                    "bbox": bbox,
                    "image_position": [center_x, center_y],
                    "depth": float(object_distance) if object_distance is not None else None,
                    "map_position": map_position,
                    "image_timestamp": img_time,
                    "map_timestamp": self.current_map_time.to_sec(),
                    "time_diff": img_time - self.current_map_time.to_sec()
                }
                objects_data.append(object_data)
            
            # Guardar detecciones en un archivo JSON
            detection_filename = os.path.join(
                self.data_path, 'detections', f"objects_{img_time:.6f}.json"
            )
            with open(detection_filename, 'w') as f:
                json.dump({
                    "image_timestamp": img_time,
                    "map_timestamp": self.current_map_time.to_sec(),
                    "objects": objects_data
                }, f, indent=2)
            
            # Publicar marcadores para visualización
            if len(markers_array.markers) > 0:
                self.markers_pub.publish(markers_array)
            
            # Publicar resumen de detecciones
            summary = f"Detected {len(objects_data)} objects: " + \
                      ", ".join([f"{obj['class']} ({obj['confidence']:.2f})" for obj in objects_data])
            self.detection_pub.publish(String(summary))
            
            rospy.loginfo(f"Processed image with {len(objects_data)} objects detected")
            
        except Exception as e:
            rospy.logerr(f"Error in object detection: {e}")
    
    def run(self):
        """Función principal de ejecución"""
        rospy.loginfo("Object Detection Mapper running")
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = ObjectDetectionMapper()
        detector.run()
    except rospy.ROSInterruptException:
        pass