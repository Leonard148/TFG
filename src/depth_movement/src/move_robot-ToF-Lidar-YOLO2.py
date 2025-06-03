#!/usr/bin/env python3

import rospy
import numpy as np
import os
import signal
import subprocess
import sys
import time
import json
import psutil
from datetime import datetime  # Añadido para usar datetime.now()
from sensor_msgs.msg import LaserScan, Image, PointCloud2, CameraInfo
from rtabmap_msgs.msg import MapData
from nav_msgs.msg import OccupancyGrid
import tf
import random
import math
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.srv import GetMap
from std_msgs.msg import String, Bool
import cv2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import threading
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
import open3d as o3d
import ctypes

# Modificada la lógica de importación para manejar mejor las dependencias
# Esto evita instalar paquetes durante la ejecución, lo que podría fallar por permisos
pyransac3d_installed = False
try:
    import pyransac3d as pyrsc

    pyransac3d_installed = True
except ImportError:
    rospy.logwarn(
        "pyransac3d no encontrado. Por favor instale manualmente con: pip install pyransac3d"
    )
    # No intentamos instalar automáticamente, ya que podría fallar por permisos


class SafeAutonomousExplorer:
    def __init__(self):
        rospy.init_node("safe_autonomous_explorer", anonymous=True)

        # Inicializar el procesador de Azure Kinect
        self.kinect_device = (
            None  # Inicializar como None antes de intentar configurarlo
        )
        try:
            pykinect.initialize_libraries()
            self.kinect_device = pykinect.start_device()
            self.transformation = self.kinect_device.transformation
            rospy.loginfo("Azure Kinect inicializado correctamente")

        except Exception as e:
            rospy.logerr(f"Error al inicializar Azure Kinect: {e}")
            # Ya se inicializó como None, no es necesario establecerlo de nuevo

        # Define base_frame first (esto debe estar fuera del try/except)
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # Define tf_listener (también fuera del try/except)
        self.tf_listener = tf.TransformListener()

        # Inicializar parámetros de cámara
        self.fx = 504.7  # Valor predeterminado para Azure Kinect
        self.fy = 504.7  # Valor predeterminado para Azure Kinect
        self.cx = 320.0  # Valor predeterminado para Azure Kinect
        self.cy = 288.0  # Valor predeterminado para Azure Kinect

        # Intentar obtener parámetros de cámara desde ROS
        try:
            camera_info_topic = rospy.get_param(
                "~camera_info_topic", "/k4a/depth/camera_info"
            )
            camera_info = rospy.wait_for_message(
                camera_info_topic, CameraInfo, timeout=5.0
            )
            self.fx = camera_info.K[0]
            self.fy = camera_info.K[4]
            self.cx = camera_info.K[2]
            self.cy = camera_info.K[5]
            rospy.loginfo(
                f"Parámetros de cámara obtenidos: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
            )
        except Exception as e:
            rospy.logwarn(
                f"No se pudieron obtener parámetros de cámara, usando valores predeterminados: {e}"
            )

        self.initial_pose = None
        # Wait until tf is ready
        try:
            rospy.loginfo(
                "Esperando transformada inicial de 'map' a 'base_link'...")
            self.tf_listener.waitForTransform(
                "map", self.base_frame, rospy.Time(0), rospy.Duration(10.0)
            )
            trans, rot = self.tf_listener.lookupTransform(
                "map", self.base_frame, rospy.Time(0)
            )
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
        # 30 grados a cada lado para la zona frontal
        self.front_angle = math.radians(30)
        # 90 grados para las zonas laterales
        self.side_angle = math.radians(90)
        self.max_speed = rospy.get_param("~max_speed", 200)
        self.min_speed = rospy.get_param("~min_speed", 100)
        self.min_safe_distance = rospy.get_param("~min_safe_distance", 1.2)
        self.critical_distance = rospy.get_param("~critical_distance", 0.5)
        self.map_check_interval = rospy.get_param("~map_check_interval", 5)
        self.rotation_min_time = rospy.get_param("~rotation_min_time", 1.5)
        self.rotation_max_time = rospy.get_param("~rotation_max_time", 3.0)
        self.target_coverage = rospy.get_param("~target_coverage", 0.95)
        self.frontier_search_distance = rospy.get_param(
            "~frontier_search_distance", 3.0
        )
        self.lidar_safety_angle = math.radians(
            rospy.get_param("~lidar_safety_angle", 30)
        )
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
        self.ransac_threshold = rospy.get_param("~ransac_threshold", 0.03)
        self.floor_height_threshold = rospy.get_param(
            "~floor_height_threshold", 0.03)
        self.drop_threshold = rospy.get_param("~drop_threshold", 0.12)
        self.danger_distance = rospy.get_param("~danger_distance", 1.0)
        self.camera_frame = rospy.get_param("~camera_frame", "camera_link")
        self.avoid_stairs_enabled = rospy.get_param("~avoid_stairs", True)

        # YOLO parameters - pesos específicos para ejecución
        self.yolo_weights = [
            "/home/jetson/catkin_ws/src/yolov7/runs/new_custom4/weights/best.pt",
            "/home/jetson/catkin_ws/src/yolov7/runs/rampa4/weights/best.pt",
            "/home/jetson/catkin_ws/src/yolov7/runs/puerta4/weights/best.pt",
            "/home/jetson/catkin_ws/src/yolov7/runs/stairs3/weights/best.pt",
            "/home/jetson/catkin_ws/src/yolov7/runs/ascensor3/weights/best.pt",
        ]
        self.yolo_img_size = 640
        self.yolo_conf_thres = 0.25
        self.yolo_process = None
        self.yolo_script_path = self.find_yolo_script()

        # State variables
        self.running = True
        self.exploration_state = "INITIAL_SCAN"
        self.last_lidar_time = rospy.Time.now()
        self.laser_scan = None
        self.initial_scan_start = None
        self.map_data = None
        self.map_save_path = os.path.expanduser("~/saved_maps")
        self.last_map_check_time = rospy.Time.now()
        self.map_coverage = 0.0
        self.stair_detected = False
        self.floor_plane_eq = None
        self.danger_zone_detected = False
        self.emergency_stop_flag = False
        self.current_move_direction = "STOP"
        self.current_motor_speeds = [0, 0, 0, 0]
        self.movement_lock = False
        self.last_movement_command_time = rospy.Time.now()
        self.consecutive_obstacles = 0
        self.max_consecutive_obstacles = rospy.get_param(
            "~max_consecutive_obstacles", 5
        )
        self.recovery_rotation_factor = rospy.get_param(
            "~recovery_rotation_factor", 2.0
        )

        # Mutex locks
        self.scan_lock = threading.RLock()
        self.movement_command_lock = threading.RLock()

        # Configuración de directorios
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_map_directory = os.path.join(
            self.map_save_path, f"map_session_{timestamp}"
        )
        try:
            os.makedirs(self.current_map_directory, exist_ok=True)
        except Exception as e:
            rospy.logerr(f"Error creando directorio de mapas: {e}")
            self.current_map_directory = self.map_save_path

        # Configuración para captura de imágenes
        self.image_capture_path = os.path.join(
            self.current_map_directory, "captures")
        os.makedirs(self.image_capture_path, exist_ok=True)
        self.last_capture_time = rospy.Time.now()
        self.capture_interval = rospy.Duration(2.0)  # Captura cada 2 segundos
        self.captured_images = []

        # Utility instances
        self.bridge = CvBridge()
        # Inicializar ransac solo si pyransac3d está instalado
        if pyransac3d_installed:
            self.ransac = pyrsc.Plane()
        else:
            self.ransac = None
            rospy.logwarn(
                "pyransac3d no disponible - la detección de planos no funcionará"
            )

        # Publishers
        self.cmd_pub = rospy.Publisher(
            "/robot/move/raw", String, queue_size=20)
        self.move_cmd_pub = rospy.Publisher(
            "/robot/move/direction", String, queue_size=20
        )
        self.danger_pub = rospy.Publisher(
            "/stair_detector/danger", Bool, queue_size=5)
        self.stair_mask_pub = rospy.Publisher(
            "/stair_detector/stair_mask", Image, queue_size=5
        )
        self.status_pub = rospy.Publisher(
            "/explorer/status", String, queue_size=5)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.diagnostics_pub = rospy.Publisher(
            "/explorer/diagnostics", String, queue_size=5
        )

        # Subscribers
        self.laser_sub = rospy.Subscriber(
            "/scan", LaserScan, self.laser_callback, queue_size=1, buff_size=2**24
        )
        self.map_sub = rospy.Subscriber(
            "/rtabmap/mapData", MapData, self.map_callback, queue_size=1
        )
        self.grid_map_sub = rospy.Subscriber(
            "/map", OccupancyGrid, self.grid_map_callback, queue_size=1
        )
        self.depth_sub = rospy.Subscriber(
            "/k4a/depth/image_raw",
            Image,
            self.depth_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.rgb_sub = rospy.Subscriber(
            "/k4a/rgb/image_raw",
            Image,
            self.rgb_callback,
            queue_size=1,
            buff_size=2**24,
        )

        # Start exploration with safety delay
        rospy.Timer(
            rospy.Duration(1.0), lambda event: self.start_initial_scan(), oneshot=True
        )

        # Watchdog timer for system health monitoring
        self.watchdog_timer = rospy.Timer(
            rospy.Duration(5.0), self.watchdog_callback)

        # System diagnostic check method
    def system_diagnostic_check(self, event=None):
        """Perform diagnostic checks when issues are detected"""
        rospy.loginfo("Running system diagnostic checks...")

        try:
            # Check ROS node status
            nodes_cmd = ["rosnode", "list"]
            nodes_result = subprocess.run(
                nodes_cmd, capture_output=True, text=True)

            # Check topic publishing rates
            topic_hz_cmd = ["rostopic", "hz", "/scan", "--window=10"]
            topic_hz_process = subprocess.Popen(
                topic_hz_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Wait briefly for data
            rospy.sleep(3.0)
            topic_hz_process.terminate()

            # Add memory usage check (psutil ya está importado al inicio)
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            if memory_usage > 1000:  # More than 1GB
                rospy.logwarn(f"High memory usage: {memory_usage:.1f}MB")

            # Check time since last scan
            time_since_last_scan = (
                rospy.Time.now() - self.last_lidar_time
            ).to_sec()
            if time_since_last_scan > 5.0:
                rospy.logwarn(
                    f"No LiDAR data received in {time_since_last_scan:.1f} seconds"
                )

            # Publish diagnostic results
            diagnostic_result = (
                "Diagnostic check complete - attempting to resume operation"
            )
            self.diagnostics_pub.publish(diagnostic_result)

            # Reset emergency flag after diagnostic check
            self.emergency_stop_flag = False

        except Exception as e:
            rospy.logerr(f"Error in diagnostic check: {e}")

    def watchdog_callback(self, event):
        """Monitorea el estado del sistema y realiza acciones correctivas si es necesario"""
        try:
            # Verificar tiempo desde último dato de LiDAR
            time_since_last_scan = (
                rospy.Time.now() - self.last_lidar_time
            ).to_sec()
            if time_since_last_scan > self.scan_timeout.to_sec():
                rospy.logwarn(
                    f"¡Alerta! No se han recibido datos del LiDAR en {time_since_last_scan:.1f} segundos"
                )

                # Si no hay datos por mucho tiempo, realizar diagnóstico
                if time_since_last_scan > self.scan_timeout.to_sec() * 2:
                    rospy.logerr(
                        "Posible fallo de sensor LiDAR. Ejecutando diagnóstico..."
                    )
                    self.system_diagnostic_check()

            # Verificar si el robot está atascado (sin movimiento por mucho tiempo)
            if self.exploration_state == "EXPLORING":
                time_since_last_movement = (
                    rospy.Time.now() - self.last_movement_command_time
                ).to_sec()
                if time_since_last_movement > 30.0:  # 30 segundos sin movimiento
                    rospy.logwarn(
                        "Robot posiblemente atascado. Iniciando maniobra de recuperación..."
                    )
                    self.execute_recovery_maneuver()

            # Verificar uso de memoria y recursos
            self.check_system_resources()

        except Exception as e:
            rospy.logerr(f"Error en watchdog_callback: {e}")

    def check_system_resources(self):
        """Verifica recursos del sistema como memoria y CPU"""
        try:
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent(interval=0.1)

            if memory_usage > 1500:  # Más de 1.5GB
                rospy.logwarn(f"Uso de memoria crítico: {memory_usage:.1f}MB")

            if cpu_percent > 90:  # Más del 90% de CPU
                rospy.logwarn(f"Uso de CPU crítico: {cpu_percent:.1f}%")

        except Exception as e:
            rospy.logerr(f"Error al verificar recursos del sistema: {e}")

    def check_map_completion(self):
        """Verifica si el mapa está completo y ejecuta start_processing si es así"""
        if self.map_coverage >= self.target_coverage:
            rospy.loginfo(
                f"Mapa completado con cobertura {self.map_coverage:.2f}. Ejecutando start_processing independientemente del retorno a posición inicial."
            )
            self.start_processing()
            return True
        return False

    def grid_map_callback(self, msg):
        """Procesa el mapa de ocupación para calcular la cobertura"""
        try:
            # Calcular cobertura del mapa
            total_cells = len(msg.data)
            unknown_cells = msg.data.count(-1)
            known_cells = total_cells - unknown_cells

            if total_cells > 0:
                self.map_coverage = known_cells / total_cells

                # Verificar si el mapa está completo
                self.check_map_completion()

                # Publicar estado
                status_msg = {
                    "state": self.exploration_state,
                    "coverage": self.map_coverage,
                    "timestamp": rospy.Time.now().to_sec(),
                }
                self.status_pub.publish(json.dumps(status_msg))

                # Guardar mapa periódicamente
                current_time = rospy.Time.now()
                if (
                    current_time - self.last_map_check_time
                ).to_sec() > self.map_check_interval:
                    self.last_map_check_time = current_time
                    self.save_current_map()

        except Exception as e:
            rospy.logerr(f"Error en grid_map_callback: {e}")

    def execute_recovery_maneuver(self):
        """Ejecuta una maniobra de recuperación cuando el robot está atascado"""
        rospy.loginfo("Ejecutando maniobra de recuperación...")

        # Detener movimiento actual
        self.stop_robot()
        rospy.sleep(1.0)

        # Retroceder brevemente
        self.move("BACKWARD", duration=2.0)
        rospy.sleep(2.5)

        # Girar aleatoriamente
        direction = "ROTATE_LEFT" if random.random() > 0.5 else "ROTATE_RIGHT"
        rotation_time = random.uniform(2.0, 4.0)
        self.move(direction, duration=rotation_time)

        # Actualizar estado
        self.consecutive_obstacles = 0
        self.last_movement_command_time = rospy.Time.now()

    def rgb_callback(self, msg):
        """Capture images periodically during exploration"""
        current_time = rospy.Time.now()

        if (current_time - self.last_capture_time) >= self.capture_interval:
            try:
                # Convert ROS image to OpenCV
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                # Save image with timestamp for detec-mapa-YOLO2.py
                timestamp = current_time.to_sec()

                # Asegurar que el directorio de detecciones YOLO existe
                yolo_detections_path = os.path.expanduser(
                    "~/saved_maps/yolo_detections"
                )
                os.makedirs(yolo_detections_path, exist_ok=True)

                # Guardar la imagen en el directorio que usa detec-mapa2
                filename = os.path.join(
                    yolo_detections_path, f"capture_{int(timestamp)}.jpg"
                )
                cv2.imwrite(filename, cv_image)

                # También guardar en el directorio original si existe
                if hasattr(self, "image_capture_path"):
                    original_filename = os.path.join(
                        self.image_capture_path, f"capture_{int(timestamp)}.jpg"
                    )
                    cv2.imwrite(original_filename, cv_image)
                    self.captured_images.append(original_filename)

                self.last_capture_time = current_time

                # Crear archivo de metadatos básico para detec-mapa2
                metadata = {"timestamp": timestamp, "detections": []}
                metadata_filename = os.path.join(
                    yolo_detections_path, f"metadata_capture_{int(timestamp)}.json"
                )
                with open(metadata_filename, "w") as f:
                    # Eliminar importación redundante de json
                    json.dump(metadata, f, indent=2)

                # Cambiado a loginfo para mejor visibilidad
                rospy.loginfo(f"Imagen capturada para detec-mapa2: {filename}")
            except Exception as e:
                rospy.logerr(f"Error capturando imagen: {e}")

    def find_yolo_script(self):
        """Busca el script detec-mapa-YOLO2.py en ubicaciones comunes"""
        possible_locations = [
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)
                                ), "detec-mapa-YOLO2.py"
            ),
            "/home/jetson/catkin_ws/src/yolov7/detec-mapa-YOLO2.py",
            os.path.expanduser("~/yolov7/detec-mapa-YOLO2.py"),
            "./detec-mapa-YOLO2.py",
        ]

        for location in possible_locations:
            if os.path.exists(location):
                rospy.loginfo(f"Script YOLO encontrado en: {location}")
                return location

        rospy.logerr(
            "No se encontró el script detec-mapa-YOLO2.py en ninguna ubicación conocida"
        )
        return None

    def run_yolo_detection(self, execution_timestamp):
        """Ejecuta el script detec-mapa-YOLO2.py con los 4 pesos diferentes"""
        if self.yolo_process is not None and self.yolo_process.poll() is None:
            rospy.logwarn("YOLO ya está en ejecución")
            return

        if self.yolo_script_path is None:
            rospy.logerr("No se puede ejecutar YOLO - script no encontrado")
            return

        if not hasattr(self, 'yolo_weights') or not self.yolo_weights:
            rospy.logerr("No se han definido pesos para YOLO")
            return

        rospy.loginfo(
            f"Iniciando detección YOLO con 4 modelos diferentes - Ejecución: {execution_timestamp}"
        )

        # Ejecutar las 4 detecciones secuencialmente
        for i, weight in enumerate(self.yolo_weights):
            self.run_single_detection(weight, i + 1, execution_timestamp)

    def run_single_detection(self, weight_path, run_number, execution_timestamp):
        """Función para ejecutar una detección individual"""
        try:
            # Extraer solo el nombre del archivo de peso sin la ruta
            weight_name = (
                os.path.basename(os.path.dirname(os.path.dirname(weight_path)))
                + "_"
                + os.path.basename(weight_path).split(".")[0]
            )

            cmd = [
                "python3",
                self.yolo_script_path,
                "--weights",
                weight_path,
                "--img",
                str(self.yolo_img_size),
                "--conf",
                str(self.yolo_conf_thres),
                "--project",
                os.path.join(
                    self.current_map_directory, f"detections_weight_{run_number}"
                ),
                "--name",
                f"run_{int(time.time())}",
                "_yolo_weight_name:=" + weight_name,
                "_execution_timestamp:="
                + execution_timestamp,  # Pasar el timestamp global
            ]

            rospy.loginfo(
                f"Ejecutando YOLO con peso {run_number}: {weight_path}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            # Esperar a que termine este proceso antes de continuar
            process.wait()

            if process.returncode != 0:
                rospy.logwarn(
                    f"YOLO con peso {run_number} terminó con código {process.returncode}"
                )
            else:
                rospy.loginfo(
                    f"YOLO con peso {run_number} completado exitosamente")

        except Exception as e:
            rospy.logerr(f"Error al ejecutar YOLO con peso {run_number}: {e}")

    def find_mapa_simbolico_script(self):
        """Busca el script mapa_simbolico-YOLO2.py en ubicaciones comunes"""
        possible_locations = [
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)
                                ), "mapa_simbolico-YOLO2.py"
            ),
            "/home/jetson/catkin_ws/src/barrier_map/scripts/mapa_simbolico-YOLO2.py",
            os.path.expanduser("~/barrier_map/mapa_simbolico-YOLO2.py"),
            "./mapa_simbolico-YOLO2.py",
        ]

        for location in possible_locations:
            if os.path.exists(location):
                rospy.loginfo(
                    f"Script Mapa Simbolico encontrado en: {location}")
                return location

        rospy.logerr(
            "No se encontró el script mapa_simbolico-YOLO2.py en ninguna ubicación conocida"
        )
        return None

    def run_mapa_simbolico(self, execution_timestamp):
        """Ejecuta el script mapa_simbolico-YOLO2.py después de todas las detecciones YOLO"""
        try:
            mapa_simbolico_path = self.find_mapa_simbolico_script()

            if mapa_simbolico_path is None:
                rospy.logerr(
                    "No se encontró el script mapa_simbolico-YOLO2.py en ninguna ubicación conocida"
                )
                return

            rospy.loginfo(
                f"Ejecutando mapa_simbolico-YOLO2.py desde: {mapa_simbolico_path}"
            )

            # Ejecutar el script con los parámetros necesarios
            cmd = [
                "python3",
                mapa_simbolico_path,
                "--session_id",
                execution_timestamp,
                "--map_save_path",
                self.current_map_directory,
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            # Esperar a que termine este proceso
            process.wait()

            if process.returncode != 0:
                stderr = process.stderr.read()
                rospy.logwarn(
                    f"mapa_simbolico-YOLO2.py terminó con código {process.returncode}: {stderr}"
                )
            else:
                rospy.loginfo(
                    "mapa_simbolico-YOLO2.py completado exitosamente")

                # Ejecutar los scripts MQTT después de que el mapa simbólico ha terminado
                rospy.loginfo("Iniciando scripts MQTT secuencialmente...")

                # Ruta para las imágenes YOLO (la misma que se usa para el análisis de imágenes)
                yolo_images_folder = os.path.expanduser(
                    "~/saved_maps/yolo_detections"
                )

                # Ruta para guardar los mapas
                maps_save_folder = self.current_map_directory

                # Asegurar que la carpeta de imágenes existe
                os.makedirs(yolo_images_folder, exist_ok=True)

                # Señal para saber cuando el publisher ha terminado
                signal_file = os.path.join(
                    os.path.dirname(
                        yolo_images_folder), "publisher_complete.signal"
                )
                if os.path.exists(signal_file):
                    os.remove(signal_file)

                # Iniciar mqtt_publisher.py primero
                mqtt_publisher_cmd = [
                    "python",
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "mqtt_publisher.py",
                    ),
                    "--folder",
                    yolo_images_folder,
                ]

                rospy.loginfo(
                    f"Iniciando mqtt_publisher.py (folder={yolo_images_folder})..."
                )
                mqtt_publisher_process = subprocess.Popen(
                    mqtt_publisher_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )

                # Guardar referencia al proceso
                self.mqtt_publisher_process = mqtt_publisher_process

                # Iniciar mqtt_json_saver.py
                mqtt_saver_cmd = [
                    "python",
                    os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "mqtt_json_saver.py",
                    ),
                    "--folder",
                    maps_save_folder,
                ]

                rospy.loginfo(
                    f"Iniciando mqtt_json_saver.py (folder={maps_save_folder})..."
                )
                mqtt_saver_process = subprocess.Popen(
                    mqtt_saver_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )

                # Guardar referencia al proceso
                self.mqtt_saver_process = mqtt_saver_process

                # Iniciar un hilo para monitorear la finalización del publisher
                mqtt_monitor_thread = threading.Thread(
                    target=self.monitor_mqtt_processes,
                    args=(signal_file, mqtt_publisher_process,
                          mqtt_saver_process),
                )
                mqtt_monitor_thread.daemon = True
                mqtt_monitor_thread.start()

                # Guardar referencia al hilo de monitoreo
                self.mqtt_monitor_thread = mqtt_monitor_thread

        except Exception as e:
            rospy.logerr(
                f"Error al ejecutar mapa_simbolico-YOLO2.py o scripts MQTT: {e}"
            )

    def monitor_mqtt_processes(self, signal_file, publisher_process, saver_process):
        """Monitorea los procesos MQTT y termina el saver después de que el publisher termina
        y verifica que se hayan procesado todos los archivos antes de continuar"""
        try:
            # Esperar a que el publisher termine naturalmente (cuando se detecte el archivo de señal)
            while not os.path.exists(signal_file):
                # Si el proceso del publisher ya terminó por alguna razón, salir del bucle
                if publisher_process.poll() is not None:
                    rospy.loginfo(
                        "mqtt_publisher.py ha terminado inesperadamente.")
                    break
                time.sleep(1)

            # Leer la cantidad de archivos procesados del publisher
            publisher_file_count = 0
            if os.path.exists(signal_file):
                try:
                    with open(signal_file, "r") as f:
                        content = f.read()
                        if "Files Sent:" in content:
                            file_count_line = [
                                line
                                for line in content.split("\n")
                                if "Files Sent:" in line
                            ][0]
                            publisher_file_count = int(
                                file_count_line.split(":")[1].strip()
                            )
                            rospy.loginfo(
                                f"El publisher procesó {publisher_file_count} archivos."
                            )
                except Exception as e:
                    rospy.logwarn(
                        f"Error al leer el conteo de archivos del publisher: {e}"
                    )

            # Verificar si existe el archivo señal del saver
            saver_signal_file = os.path.join(
                os.path.dirname(signal_file), "saver_complete.signal"
            )

            # Esperar hasta que el saver termine o se confirme que procesó todos los archivos
            timeout_counter = 0
            saver_completed = False
            while (
                not saver_completed and timeout_counter < 60
            ):  # Máximo 60 segundos de espera
                if os.path.exists(saver_signal_file):
                    rospy.loginfo("El saver ha finalizado su procesamiento.")
                    saver_completed = True
                    break

                if saver_process.poll() is not None:
                    rospy.loginfo("mqtt_json_saver.py ha terminado.")
                    saver_completed = True
                    break

                # Si han pasado 30 segundos, verificar manualmente los archivos procesados
                if timeout_counter == 30:
                    rospy.loginfo(
                        "Verificando manualmente los archivos procesados..."
                    )

                    # Contar archivos en la carpeta del saver
                    try:
                        if hasattr(self, "current_map_directory"):
                            saver_files = [
                                f
                                for f in os.listdir(self.current_map_directory)
                                if f.endswith(".json")
                                and os.path.isfile(
                                    os.path.join(self.current_map_directory, f)
                                )
                            ]
                            saver_file_count = len(saver_files)

                            rospy.loginfo(
                                f"El saver ha procesado {saver_file_count} archivos de {publisher_file_count}."
                            )

                            # Si el saver ha procesado todos los archivos (o más), considerarlo completo
                            if (
                                publisher_file_count > 0
                                and saver_file_count >= publisher_file_count
                            ):
                                rospy.loginfo(
                                    "El saver parece haber procesado todos los archivos. Continuando..."
                                )
                                saver_completed = True
                                break
                    except Exception as e:
                        rospy.logwarn(
                            f"Error al verificar archivos procesados: {e}"
                        )

                time.sleep(1)
                timeout_counter += 1

            # Si no se completó después del tiempo de espera, continuar de todos modos
            if not saver_completed:
                rospy.logwarn(
                    "Tiempo de espera excedido para el saver. Continuando de todos modos..."
                )

            # Terminar el proceso del saver si aún está en ejecución
            if saver_process.poll() is None:  # Si aún está en ejecución
                rospy.loginfo("Terminando mqtt_json_saver.py...")
                saver_process.terminate()
                try:
                    saver_process.wait(timeout=5)
                except:
                    saver_process.kill()  # Si no termina con terminate, forzar con kill
                rospy.loginfo("mqtt_json_saver.py ha sido terminado.")

            # Eliminar los archivos de señal si existen
            for signal in [signal_file, saver_signal_file]:
                if os.path.exists(signal):
                    os.remove(signal)

            rospy.loginfo(
                "Todos los procesos MQTT han sido terminados correctamente."
            )
        except Exception as e:
            rospy.logerr(f"Error durante el monitoreo de procesos MQTT: {e}")
            # Intentar terminar los procesos en caso de error
            for process in [publisher_process, saver_process]:
                if process and process.poll() is None:
                    try:
                        process.terminate()
                        process.wait(timeout=2)
                    except:
                        try:
                            process.kill()
                        except:
                            pass

    def shutdown_hook(self):
        """Función que se ejecuta cuando el nodo ROS se cierra"""
        rospy.loginfo(
            "Cerrando el nodo. Terminando procesos en segundo plano...")

        # Terminar procesos MQTT si existen
        for attr_name in ["mqtt_publisher_process", "mqtt_saver_process"]:
            if hasattr(self, attr_name):
                process = getattr(self, attr_name)
                if process and process.poll() is None:
                    try:
                        process.terminate()
                        process.wait(timeout=2)
                        rospy.loginfo(
                            f"Proceso {attr_name} terminado correctamente."
                        )
                    except Exception as e:
                        rospy.logwarn(f"Error al terminar {attr_name}: {e}")
                        try:
                            process.kill()
                        except:
                            pass

        rospy.loginfo("Limpieza finalizada.")

    def run_all_processes(self):
        # Generar el timestamp aquí para que esté disponible para todas las llamadas
        execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rospy.loginfo(
            f"Iniciando procesamiento con timestamp global: {execution_timestamp}")

        # Ejecutar todas las detecciones YOLO con el timestamp generado
        # Pasar el timestamp a esta función
        self.run_yolo_detection(execution_timestamp)

        # Una vez completadas todas las detecciones YOLO, ejecutar mapa_simbolico-YOLO2.py
        rospy.loginfo(
            "Todas las detecciones YOLO completadas. Ejecutando mapa_simbolico-YOLO2.py..."
        )
        self.run_mapa_simbolico(execution_timestamp)  # Usar el mismo timestamp

    def start_processing(self):
        """Inicia el procesamiento YOLO y mapa_simbolico en un thread separado"""
        rospy.loginfo("Iniciando procesamiento en thread separado...")
        yolo_thread = threading.Thread(target=self.run_all_processes)
        yolo_thread.daemon = True
        yolo_thread.start()

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
                        f"LiDAR data quality low: only {range_percentage:.1f}% valid ranges"
                    )

    # Enhanced depth_to_pointcloud with SIMD optimizations
    def depth_to_pointcloud(self, depth_image):
        """Convertir imagen de profundidad a nube de puntos usando Azure Kinect"""
        try:
            if self.kinect_device is None:
                raise Exception("Dispositivo Azure Kinect no inicializado")

            # Obtener nube de puntos usando el SDK de Azure Kinect
            points_3d = self.transformation.depth_image_to_point_cloud(
                depth_image, _k4a.K4A_CALIBRATION_TYPE_DEPTH
            )

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

    def detect_stairs(self, depth_image, header):
        """Enhanced stair detection with better plane fitting and visualization"""
        try:
            # Check image quality
            if depth_image.mean() < 1e-6 or depth_image.std() < 1e-6:
                rospy.logwarn("Low quality depth image")
                return

            # Convert depth image to point cloud
            points_3d = self.depth_to_pointcloud(depth_image)[
                0
            ]  # Only use the first return value

            if len(points_3d) < 50:  # Increased minimum points for better reliability
                rospy.logwarn("Not enough valid points for stair detection")
                return

            # 1. Detect floor plane using RANSAC with optimized parameters
            best_eq, inliers = self.ransac.fit(
                points_3d, self.ransac_threshold, maxIteration=1000
            )

            # Calculate inlier percentage after RANSAC
            inlier_percentage = (len(inliers) / len(points_3d)) * 100

            # Additional plane detection validation
            if inlier_percentage < 30:  # Less than 30% of points are inliers
                rospy.logwarn("Unreliable plane detection")
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
            rospy.logdebug(
                f"Floor plane detection: {inlier_percentage:.1f}% points as inliers, "
                f"vertical alignment: {vertical_alignment:.3f}"
            )

            # More strict horizontal plane requirement (was 0.8)
            if vertical_alignment > 0.9:
                self.floor_plane_eq = best_eq

                # Create visualization image with more detailed information
                visualization = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
                visualization = cv2.normalize(
                    visualization, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )

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
                expected_z = (-a * x - b * y - d) / np.clip(
                    c, 1e-6, None
                )  # Avoid division by zero

                # Calculate depth differences
                depth_diff = z - expected_z
                stair_mask = (depth_diff > self.drop_threshold) & valid_depth

                # Process mask to remove noise with adaptive morphology
                stair_mask = stair_mask.astype(np.uint8) * 255
                # Adaptive kernel size based on image width
                kernel_size = max(3, int(width / 100))
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                stair_mask = cv2.morphologyEx(
                    stair_mask, cv2.MORPH_OPEN, kernel)
                stair_mask = cv2.morphologyEx(
                    stair_mask, cv2.MORPH_CLOSE, kernel)

                # Find contours with minimum area requirement
                contours, _ = cv2.findContours(
                    stair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                significant_contours = [
                    cnt for cnt in contours if cv2.contourArea(cnt) > 300
                ]  # Reduced from 500

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
                    min_distance = float("inf")
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
                        cv2.putText(
                            visualization,
                            "DANGER: STAIR DETECTED",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )
                    elif min_distance < self.danger_distance * 1.5:  # Warning zone
                        cv2.putText(
                            visualization,
                            "WARNING: STAIR NEARBY",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255),
                            2,
                        )

                # Informational overlay
                cv2.putText(
                    visualization,
                    f"Floor confidence: {inlier_percentage:.1f}%",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # Publish visualization with contour information
                cv2.drawContours(
                    visualization, significant_contours, -1, (0, 0, 255), 2
                )
                self.danger_pub.publish(Bool(self.danger_zone_detected))

                # Publish visualization image
                stair_mask_msg = self.bridge.cv2_to_imgmsg(
                    visualization, encoding="bgr8"
                )
                stair_mask_msg.header = header
                self.stair_mask_pub.publish(stair_mask_msg)

                # Log detection status
                if self.danger_zone_detected:
                    rospy.logwarn(
                        f"DANGER! Stair detected at {min_distance:.2f}m")
                elif self.stair_detected:
                    rospy.loginfo(
                        f"Stair detected at safe distance: {min_distance:.2f}m"
                    )

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
        valid_mask = np.logical_and(
            ranges > self.laser_scan.range_min, ranges < self.laser_scan.range_max
        )

        # More efficient outlier filtering using vectorized operations where possible
        window_size = 5
        # Create a view into the data with rolling windows
        # This implementation can be optimized further with specialized libraries like scipy
        filtered_valid_mask = valid_mask.copy()
        for i in range(len(ranges)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(ranges), i + window_size // 2 + 1)
            window = ranges[start_idx:end_idx]
            valid_window = np.logical_and(
                window > self.laser_scan.range_min, window < self.laser_scan.range_max
            )
            # Skip filtering if we have too many invalid readings in the window
            if np.sum(valid_window) < window_size // 2:
                continue

            valid_window_values = window[valid_window]
            if len(valid_window_values) > 0:
                median = np.median(valid_window_values)
                # If value is significantly different from median
                if abs(ranges[i] - median) > 0.5:
                    filtered_valid_mask[i] = False  # Mark as invalid

        # Angle array for easier zone calculation
        angles = np.arange(len(ranges)) * angle_increment + angle_min

        # Define zones - front, left, right with overlap for better awareness
        front_mask = np.abs(angles) <= self.front_angle
        left_mask = np.logical_and(
            angles > -self.side_angle, angles < -self.front_angle * 0.5
        )
        right_mask = np.logical_and(
            angles < self.side_angle, angles > self.front_angle * 0.5
        )

        # Rear zone for better situational awareness
        rear_mask = np.abs(angles) >= math.radians(150)

        # Combine masks with valid readings
        front_valid = np.logical_and(front_mask, filtered_valid_mask)
        left_valid = np.logical_and(left_mask, filtered_valid_mask)
        right_valid = np.logical_and(right_mask, filtered_valid_mask)
        rear_valid = np.logical_and(rear_mask, filtered_valid_mask)

        # Calculate zone statistics with better handling of empty zones
        front_distances = (
            ranges[front_valid] if np.any(
                front_valid) else np.array([float("inf")])
        )
        left_distances = (
            ranges[left_valid] if np.any(
                left_valid) else np.array([float("inf")])
        )
        right_distances = (
            ranges[right_valid] if np.any(
                right_valid) else np.array([float("inf")])
        )
        rear_distances = (
            ranges[rear_valid] if np.any(
                rear_valid) else np.array([float("inf")])
        )

        # Calculate more robust zone metrics including quantiles
        zones = {
            "front": {
                "min_distance": np.min(front_distances),
                "mean_distance": np.mean(front_distances),
                "valid_count": np.sum(front_valid),
                "total_count": np.sum(front_mask),
                "closest_percentile": (
                    np.percentile(front_distances, 10)
                    if len(front_distances) > 5
                    else np.min(front_distances)
                ),
            },
            "left": {
                "min_distance": np.min(left_distances),
                "mean_distance": np.mean(left_distances),
                "valid_count": np.sum(left_valid),
                "total_count": np.sum(left_mask),
                "closest_percentile": (
                    np.percentile(left_distances, 10)
                    if len(left_distances) > 5
                    else np.min(left_distances)
                ),
            },
            "right": {
                "min_distance": np.min(right_distances),
                "mean_distance": np.mean(right_distances),
                "valid_count": np.sum(right_valid),
                "total_count": np.sum(right_mask),
                "closest_percentile": (
                    np.percentile(right_distances, 10)
                    if len(right_distances) > 5
                    else np.min(right_distances)
                ),
            },
            "rear": {
                "min_distance": np.min(rear_distances),
                "mean_distance": np.mean(rear_distances),
                "valid_count": np.sum(rear_valid),
                "total_count": np.sum(rear_mask),
                "closest_percentile": (
                    np.percentile(rear_distances, 10)
                    if len(rear_distances) > 5
                    else np.min(rear_distances)
                ),
            },
        }

        # Log data quality metrics
        for zone_name, zone_data in zones.items():
            valid_ratio = zone_data["valid_count"] / \
                max(1, zone_data["total_count"])
            if valid_ratio < 0.5:  # Less than 50% valid readings
                rospy.logdebug(
                    f"Low LiDAR quality in {zone_name} zone: {valid_ratio:.2f}"
                )

        return zones

    # Improved process_lidar_for_navigation with more organized obstacle handling
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
            if self.current_move_direction in [
                "FORWARD",
                "FORWARD_LEFT",
                "FORWARD_RIGHT",
            ]:
                self.execute_evasion_maneuver()
            return

        # Handle robot state-specific behaviors
        if self._handle_robot_state():
            return

        # Analyze LiDAR data by zones (front, left, right, rear)
        zones_analysis = self.analyze_lidar_zones()

        if zones_analysis is None:
            rospy.logwarn("No valid LiDAR data for navigation")
            return

        # Log zone analysis for debugging
        self._log_zone_analysis(zones_analysis)

        # Enhanced obstacle classification using percentile values for more robustness
        obstacles = self._classify_obstacles(zones_analysis)

        # Check for critical obstacles
        if zones_analysis["front"]["min_distance"] < self.critical_distance:
            self._handle_critical_obstacle(
                zones_analysis["front"]["min_distance"])
            return

        # Track consecutive obstacle encounters for stuck detection
        if obstacles["front"]:
            self._track_obstacle_encounters()
            if self.consecutive_obstacles > self.max_consecutive_obstacles:
                rospy.logwarn(
                    "Potentially stuck - initiating recovery behavior")
                self.start_recovery_behavior()
                return
        else:
            # Reset counter when path is clear
            self.consecutive_obstacles = 0

        # Decide movement based on obstacle configuration
        self._decide_movement(obstacles, zones_analysis)

    # Helper methods to organize the navigation logic
    def _handle_robot_state(self):
        """Handle different robot states like initial scan, return to dock, etc."""
        # Handle initial scan state
        if self.exploration_state == "INITIAL_SCAN":
            if (rospy.Time.now() - self.initial_scan_start) > rospy.Duration(10):
                self.stop_robot()
                self.exploration_state = "SCANNING"
                rospy.loginfo("Initial scan complete, beginning exploration")
                self.status_pub.publish("INITIAL_SCAN_COMPLETE")
            return True

        # Handle return to dock state
        if self.exploration_state == "RETURN_TO_DOCK":
            # This would ideally use localization and path planning
            self.stop_robot()
            rospy.loginfo("Return to dock requested - stopping for now")
            self.status_pub.publish("STOPPED_FOR_DOCK")
            return True

        return False

    def _log_zone_analysis(self, zones_analysis):
        """Log LiDAR zone analysis data for debugging"""
        rospy.logdebug(
            f"LiDAR zone analysis - Front: {zones_analysis['front']['min_distance']:.2f}m"
        )
        rospy.logdebug(
            f"LiDAR zone analysis - Left: {zones_analysis['left']['min_distance']:.2f}m"
        )
        rospy.logdebug(
            f"LiDAR zone analysis - Right: {zones_analysis['right']['min_distance']:.2f}m"
        )
        rospy.logdebug(
            f"LiDAR zone analysis - Rear: {zones_analysis['rear']['min_distance']:.2f}m"
        )

    def _classify_obstacles(self, zones_analysis):
        """Classify obstacles in different zones"""
        return {
            "front": zones_analysis["front"]["closest_percentile"]
            < self.min_safe_distance,
            "left": zones_analysis["left"]["closest_percentile"]
            < self.min_safe_distance * 0.8,
            "right": zones_analysis["right"]["closest_percentile"]
            < self.min_safe_distance * 0.8,
            "rear": zones_analysis["rear"]["closest_percentile"]
            < self.min_safe_distance * 0.5,
        }

    def _handle_critical_obstacle(self, distance):
        """Handle critical obstacle encounter"""
        rospy.logwarn(f"Critical obstacle at {distance:.2f}m - EMERGENCY STOP")
        self.emergency_stop_flag = True
        self.stop_robot()
        self.start_rotation()
        self.consecutive_obstacles += 1

    def _track_obstacle_encounters(self):
        """Track consecutive obstacle encounters"""
        self.consecutive_obstacles += 1

    def _decide_movement(self, obstacles, zones_analysis):
        """Decide movement based on obstacle configuration"""
        if obstacles["front"]:
            if not obstacles["left"] and not obstacles["right"]:
                self._choose_rotation_direction(zones_analysis)
            elif obstacles["left"] and not obstacles["right"]:
                self._handle_right_clear_path(zones_analysis)
            elif obstacles["right"] and not obstacles["left"]:
                self._handle_left_clear_path(zones_analysis)
            else:
                # Completely blocked, rotate with adaptive strategy
                self.start_rotation()
        else:
            # No front obstacle, check if we should find frontier
            if self.should_find_frontier():
                self.find_exploration_direction(zones_analysis)
            else:
                # Forward path clear - check if we should go straight or adjust
                if obstacles["left"] and not obstacles["right"]:
                    # Obstacle on left, bias slightly right
                    self.move("FORWARD_RIGHT", speed_factor=0.8)
                elif obstacles["right"] and not obstacles["left"]:
                    # Obstacle on left, bias slightly right
                    self.move("FORWARD_LEFT", speed_factor=0.8)
                else:
                    # Clear path forward
                    self.move("FORWARD")

    def _choose_rotation_direction(self, zones_analysis):
        """Choose rotation direction based on which side has more space"""
        if (
            zones_analysis["left"]["mean_distance"]
            > zones_analysis["right"]["mean_distance"] * 1.2
        ):
            # Left is significantly clearer
            self.move("ROTATE_LEFT")
        elif (
            zones_analysis["right"]["mean_distance"]
            > zones_analysis["left"]["mean_distance"] * 1.2
        ):
            # Right is significantly clearer
            self.move("ROTATE_RIGHT")
        else:
            # Similar clearance, make a weighted random choice
            left_weight = zones_analysis["left"]["mean_distance"]
            right_weight = zones_analysis["right"]["mean_distance"]
            total_weight = left_weight + right_weight
            if random.random() < (left_weight / total_weight):
                self.move("ROTATE_LEFT")
            else:
                self.move("ROTATE_RIGHT")

    def _handle_right_clear_path(self, zones_analysis):
        """Handle case when right path is clear"""
        if zones_analysis["front"]["min_distance"] < self.min_safe_distance * 0.7:
            # Very close front obstacle - rotate instead of forward-right
            self.move("ROTATE_RIGHT")
        else:
            self.move("FORWARD_RIGHT")

    def _handle_left_clear_path(self, zones_analysis):
        """Handle case when left path is clear"""
        if zones_analysis["front"]["min_distance"] < self.min_safe_distance * 0.7:
            # Very close front obstacle - rotate instead of forward-left
            self.move("ROTATE_LEFT")
        else:
            self.move("FORWARD_LEFT")

    # Improved obstacle evasion method
    def execute_evasion_maneuver(self):
        """Execute a maneuver to evade an obstacle or hazard"""
        rospy.logwarn("Executing evasion maneuver")

        # First stop to ensure safety
        self.stop_robot()

        # Then back up slightly
        self.move("BACKWARD", duration=1.0)

        # After backing up, rotate to find clear path
        rospy.Timer(
            rospy.Duration(1.2), lambda event: self.start_rotation(), oneshot=True
        )

        # Reset emergency flag after evasion
        rospy.Timer(
            rospy.Duration(5.0),
            lambda event: setattr(self, "emergency_stop_flag", False),
            oneshot=True,
        )

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
        rospy.Timer(
            rospy.Duration(0.5),
            lambda event: self.move("BACKWARD", duration=2.0),
            oneshot=True,
        )

        # 3. Rotate longer than normal
        rospy.Timer(
            rospy.Duration(2.7),
            lambda event: self.start_rotation(
                min_time=self.rotation_min_time * self.recovery_rotation_factor
            ),
            oneshot=True,
        )

        # 4. Resume normal operation
        rotation_time = self.rotation_min_time * self.recovery_rotation_factor + 0.5
        rospy.Timer(
            rospy.Duration(2.7 + rotation_time),
            lambda event: self.reset_after_recovery(),
            oneshot=True,
        )

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
            # Check if stairs are detected
            if self.stair_detected or self.danger_zone_detected:
                rospy.logwarn(
                    "Movement canceled! Stair or danger zone detected")
                self.stop_robot()
                return False

            if self.movement_lock and direction != "STOP":
                return False

            # Don't allow movement if emergency stop is active
            if self.emergency_stop_flag and direction != "STOP":
                rospy.logwarn("Movement blocked by emergency stop flag")
                return False

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
                return False

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
                rospy.Timer(
                    rospy.Duration(duration),
                    lambda event: self.release_movement_lock(),
                    oneshot=True,
                )

            rospy.logdebug(f"Moving {direction} with speeds {speeds}")
            return True

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
                if zones["left"]["mean_distance"] > zones["right"]["mean_distance"]:
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
        rospy.Timer(
            rospy.Duration(rotation_time),
            lambda event: self.stop_after_rotation(),
            oneshot=True,
        )

    # Method to stop after rotation
    def stop_after_rotation(self):
        """Stop robot after rotation and reset flags"""
        try:
            self.stop_robot()
            self.movement_lock = False

            # Reset emergency flag if it was set
            if self.emergency_stop_flag:
                rospy.Timer(
                    rospy.Duration(0.5),
                    lambda event: setattr(self, "emergency_stop_flag", False),
                    oneshot=True,
                )
        except Exception as e:
            rospy.logerr(f"Error in stop_after_rotation: {e}")
            # Ensure we reset the movement lock even if there's an error
            self.movement_lock = False

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

        # Calculate exploration efficiency metrics - fixed potential division by zero
        if occupied_cells > 0:
            free_to_occupied_ratio = free_cells / occupied_cells
        else:
            # Infinite ratio if no occupied cells
            free_to_occupied_ratio = float("inf")

        rospy.loginfo(
            f"Map coverage: {coverage:.2f}, Free/Occupied ratio: {free_to_occupied_ratio:.2f}"
        )
        rospy.loginfo(
            f"Map cells - Free: {free_cells}, Occupied: {occupied_cells}, Unknown: {unknown_cells}"
        )

        return coverage

    # Enhanced frontier detection
    def should_find_frontier(self):
        """Determine if robot should seek frontier or finish mapping"""
        if self.exploration_state == "RETURN_TO_DOCK":
            return False

        if (
            rospy.Time.now() - self.last_map_check_time
        ).to_sec() < self.map_check_interval:
            return False

        self.last_map_check_time = rospy.Time.now()

        # Update map coverage - unificado para usar map_data consistentemente
        current_map = self.map_data  # Usando map_data consistentemente
        if current_map is not None:
            self.map_coverage = self.estimate_map_coverage(current_map)

        if (
            self.exploration_state != "RETURN_TO_DOCK"
            and self.map_coverage > self.target_coverage
        ):
            rospy.loginfo(
                f"Map coverage ({self.map_coverage:.2f}) exceeds target ({self.target_coverage:.2f})"
            )
            self.exploration_state = "RETURN_TO_DOCK"
            self.return_to_initial_position()
            return False

        chance = 0.5 * (1.0 - self.map_coverage / self.target_coverage)
        return random.random() < chance

    def check_map_coverage(self):
        """Check if the map has reached target coverage"""
        if self.map_data is None:
            return False

        # Calculate coverage
        total_cells = len(self.map_data.data)
        known_cells = sum(1 for cell in self.map_data.data if cell != -1)
        self.map_coverage = known_cells / total_cells if total_cells > 0 else 0

        rospy.loginfo(f"Map coverage: {self.map_coverage:.2%}")

        if self.map_coverage >= self.target_coverage:
            rospy.loginfo("Target map coverage reached!")
            self.save_map()
            return True

        return False

    def return_to_initial_position(self):
        """Return to initial position and process images when complete"""
        try:
            if self.initial_pose is None:
                rospy.logerr("No initial position recorded")
                return False

            # Get current position
            try:
                current_trans, current_rot = self.tf_listener.lookupTransform(
                    "map", self.base_frame, rospy.Time(0)
                )
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                rospy.logerr(f"Error getting current position: {e}")
                return False

            # Calculate distance to initial position
            distance = math.sqrt(
                (current_trans[0] - self.initial_pose.pose.position.x) ** 2
                + (current_trans[1] - self.initial_pose.pose.position.y) ** 2
            )

            if distance < 0.1:  # If close to initial position
                rospy.loginfo("Returned to initial position")
                return True

            # Move toward initial position
            cmd_vel = Twist()
            speed_factor = min(1.0, distance / 2.0)
            cmd_vel.linear.x = 0.5 * speed_factor
            self.cmd_vel_pub.publish(cmd_vel)

            rospy.loginfo(
                f"Returning to initial position. Distance: {distance:.2f}m")

            # Para evitar recursión infinita, utilizamos un temporizador en lugar de llamadas recursivas
            if not hasattr(self, "_return_timer") or self._return_timer is None:
                self._return_timer = rospy.Timer(
                    rospy.Duration(1.0),
                    lambda event: self._check_return_progress(),
                    oneshot=True,
                )

            return False

        except Exception as e:
            rospy.logerr(f"Error in return_to_initial_position: {e}")
            return False

    def _check_return_progress(self):
        """Check progress of return to initial position"""
        try:
            # Limpiar el temporizador
            self._return_timer = None

            # Comprobar si seguimos necesitando volver
            if (
                self.exploration_state == "RETURN_TO_DOCK"
                and not self.is_at_initial_position()
            ):
                # En lugar de llamar recursivamente, volvemos a intentar la función principal
                self.return_to_initial_position()
        except Exception as e:
            rospy.logerr(f"Error in _check_return_progress: {e}")

    def is_at_initial_position(self):
        """Verifica si el robot está en la posición inicial"""
        try:
            if self.initial_pose is None:
                return False

            current_trans, _ = self.tf_listener.lookupTransform(
                "map", self.base_frame, rospy.Time(0)
            )

            distance = math.sqrt(
                (current_trans[0] - self.initial_pose.pose.position.x) ** 2
                + (current_trans[1] - self.initial_pose.pose.position.y) ** 2
            )
            return (
                distance < 0.1
            )  # Considerar que está en posición inicial si está a menos de 10cm

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
        front_score = zones_analysis["front"]["mean_distance"] * 1.5
        left_score = zones_analysis["left"]["mean_distance"]
        right_score = zones_analysis["right"]["mean_distance"]

        # Penalize directions with obstacles
        if zones_analysis["front"]["min_distance"] < self.min_safe_distance:
            front_score = 0
        if zones_analysis["left"]["min_distance"] < self.min_safe_distance * 0.8:
            left_score = 0
        if zones_analysis["right"]["min_distance"] < self.min_safe_distance * 0.8:
            right_score = 0

        # Choose direction with highest score
        if front_score > left_score and front_score > right_score:
            self.move("FORWARD")
        elif left_score > right_score:
            self.move("ROTATE_LEFT", duration=1.0)
            # Schedule forward movement after rotation
            rospy.Timer(
                rospy.Duration(1.2), lambda event: self.move("FORWARD"), oneshot=True
            )
        else:
            self.move("ROTATE_RIGHT", duration=1.0)
            # Schedule forward movement after rotation
            rospy.Timer(
                rospy.Duration(1.2), lambda event: self.move("FORWARD"), oneshot=True
            )

        # Log decision
        rospy.loginfo(
            f"Frontier scores - Front: {front_score:.2f}, Left: {left_score:.2f}, Right: {right_score:.2f}"
        )

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
        if hasattr(self, "map_data") and self.map_data is not None:
            try:
                map_file = os.path.join(
                    self.map_save_path, f"final_map_{int(time.time())}.pgm"
                )
                rospy.loginfo(f"Saving final map to {map_file}")
                # This would call map_saver functionality
            except Exception as e:
                rospy.logerr(f"Error saving map: {e}")

        # Small delay to ensure messages are sent
        rospy.sleep(0.5)

        rospy.loginfo("Safe autonomous explorer shutdown complete")

    def __del__(self):
        """Limpieza de recursos al destruir la instancia"""
        try:
            if hasattr(self, "yolo_process") and self.yolo_process:
                self.stop_yolo_detector()
            if hasattr(self, "watchdog_timer"):
                self.watchdog_timer.shutdown()
            # Cerrar todos los subscribers
            if hasattr(self, "laser_sub"):
                self.laser_sub.unregister()
            if hasattr(self, "depth_sub"):
                self.depth_sub.unregister()
            # Cerrar el dispositivo Azure Kinect
            if hasattr(self, "kinect_device") and self.kinect_device is not None:
                self.kinect_device.stop()
        except Exception as e:
            rospy.logerr(f"Error durante la limpieza: {e}")

    # Method to start initial scan

    def start_initial_scan(self):
        """Inicia el escaneo inicial del entorno"""
        rospy.loginfo("Iniciando escaneo inicial del entorno...")
        self.exploration_state = "INITIAL_SCAN"
        self.initial_scan_start = rospy.Time.now()

        # Publicar estado
        self.status_pub.publish("INITIAL_SCAN")

        # Girar lentamente para escanear el entorno
        self.send_movement_command("ROTATE_LEFT", duration=10.0)

        # Programar transición a exploración después del escaneo inicial
        scan_duration = rospy.Duration(10.0)  # 10 segundos de escaneo inicial
        rospy.Timer(scan_duration,
                    lambda event: self.start_exploration(), oneshot=True)

    def send_movement_command(self, direction, duration=None):
        """Envía un comando de movimiento al robot con opción de duración"""
        with self.movement_command_lock:
            self.current_move_direction = direction
            self.move_cmd_pub.publish(direction)
            self.last_movement_command_time = rospy.Time.now()

            # Registrar comando
            rospy.logdebug(f"Enviando comando de movimiento: {direction}")

            # Si se especifica duración, programar detención
            if duration is not None:
                rospy.Timer(
                    rospy.Duration(duration),
                    lambda event: self.stop_after_duration(),
                    oneshot=True,
                )

    def stop_after_duration(self):
        """Detiene el robot después de una duración programada"""
        with self.movement_command_lock:
            if self.current_move_direction != "STOP":
                self.move_cmd_pub.publish("STOP")
                self.current_move_direction = "STOP"
                rospy.logdebug(
                    "Deteniendo robot después de duración programada")

    def start_exploration(self):
        """Inicia la exploración después del escaneo inicial"""
        if self.exploration_state == "INITIAL_SCAN":
            rospy.loginfo(
                "Escaneo inicial completado. Iniciando exploración...")
            self.exploration_state = "EXPLORING"
            self.status_pub.publish("EXPLORING")

            # Guardar mapa inicial
            self.save_current_map("initial_map")

    # Method to save the latest map
    def map_callback(self, msg):
        """Callback mejorado para el procesamiento de datos del mapa"""
        try:
            self.map_data = msg

            # Verificar si es tiempo de guardar el mapa
            current_time = rospy.Time.now()
            if (current_time - self.last_map_save_time) > self.map_save_interval:
                if self.save_current_map():
                    self.last_map_save_time = current_time

        except Exception as e:
            rospy.logerr(f"Error en map_callback: {e}")

    def save_current_map(self, map_name="map"):
        """Guarda el mapa actual con el nombre especificado"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            map_filename = f"{map_name}_{timestamp}"
            map_path = os.path.join(self.current_map_directory, map_filename)

            # Crear comando para guardar mapa
            save_cmd = ["rosrun", "map_server", "map_saver", "-f", map_path]

            # Ejecutar comando
            rospy.loginfo(f"Guardando mapa en: {map_path}")
            process = subprocess.Popen(
                save_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Esperar a que termine
            process.wait(timeout=30)

            if process.returncode == 0:
                rospy.loginfo(f"Mapa guardado exitosamente en: {map_path}")
                return True
            else:
                stderr = process.stderr.read().decode("utf-8")
                rospy.logerr(f"Error al guardar mapa: {stderr}")
                return False

        except Exception as e:
            rospy.logerr(f"Error al guardar mapa: {e}")
            return False

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def depth_callback(self, msg):
        """Procesa la imagen de profundidad para detectar escaleras y obstáculos"""
        try:
            # Convertir mensaje ROS a imagen OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough")

            # Llamar al método de detección de escaleras
            if self.avoid_stairs_enabled:
                self.detect_stairs(depth_image)

            # Publicar información de diagnóstico
            if self.stair_detected or self.danger_zone_detected:
                self.status_pub.publish(
                    "¡PRECAUCIÓN! Escalera o zona de peligro detectada"
                )

        except Exception as e:
            rospy.logerr(f"Error en depth_callback: {e}")

    # Run method to start exploration

    def run(self):
        """Main run loop for the explorer"""
        rospy.loginfo("Starting exploration run")
        rospy.on_shutdown(self.shutdown)

        rate = rospy.Rate(10)  # 10 Hz control loop

        while not rospy.is_shutdown() and self.running:
            # Process LiDAR for obstacle avoidance
            self.process_lidar_for_navigation()

            # Verificar si hay escaleras detectadas
            if self.stair_detected or self.danger_zone_detected:
                rospy.logwarn(
                    "¡Escalera o zona de peligro detectada! Deteniendo movimiento"
                )
                self.stop_robot()
                # Esperar un momento antes de intentar maniobrar
                rospy.sleep(1.0)
                # Ejecutar maniobra de evasión
                self.execute_recovery_maneuver()

            # Check if we should return to initial position
            if self.exploration_state == "RETURN_TO_DOCK":
                if self.return_to_initial_position():
                    self.exploration_state = "IDLE"

            # Check for map coverage periodically
            if (
                rospy.Time.now() - self.last_map_check_time
            ).to_sec() >= self.map_check_interval:
                if self.check_map_coverage():
                    self.exploration_state = "RETURN_TO_DOCK"

            rate.sleep()


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
