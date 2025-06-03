#!/usr/bin/env python3

import yaml
from sklearn.cluster import DBSCAN
import math
import sys
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib
import tf
from nav_msgs.msg import OccupancyGrid
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import os
import rospy
import json
import cv2
cv2.setUseOpenVX(False)  # Deshabilitar OpenVX para evitar interfaces gráficas
os.environ["OPENCV_OPENCL_RUNTIME"] = ""  # Deshabilitar OpenCL
# Configurar backend no interactivo antes de importar pyplot
matplotlib.use('Agg')
# Configurar Qt para modo sin pantalla
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["DISPLAY"] = ""  # Evitar uso de display X11

# Constantes globales
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_ESTIMATED_DEPTH = 2.0
FOV_H = math.radians(60)
FOV_V = math.radians(45)


class ObjectMapGenerator:
    def __init__(self, map_file_path=None):
        """
        Inicializa el generador de mapas de objetos.

        Args:
            map_file_path: Ruta al archivo PGM del mapa. Si es None, se debe llamar a load_map_from_file manualmente.
        """
        # Configuración básica de parámetros
        self.output_path = os.path.expanduser(
            rospy.get_param("~output_path", "~/annotated_maps"))
        self.object_marker_size = rospy.get_param(
            "~object_marker_size", 0.5)  # in meters
        self.cluster_distance = rospy.get_param(
            "~cluster_distance", 0.5)  # for DBSCAN
        self.min_detections = rospy.get_param(
            "~min_detections", 1)  # min detections per object
        self.min_confidence = rospy.get_param("~min_confidence", 0.4)

        # Colores para clases de objetos
        self.class_colors = {
            'stairs': (255, 0, 0),
            'ramp': (0, 0, 255),
            'door': (0, 255, 0),
            'lifter': (255, 165, 0),
            'person': (255, 0, 255)
        }

        self.class_mpl_colors = {
            'stairs': 'red',
            'ramp': 'blue',
            'door': 'green',
            'lifter': 'orange',
            'person': 'magenta'
        }

        # Crear directorio de salida con timestamp único
        timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(
            self.output_path, f"map_{timestamp_dir}")
        os.makedirs(self.output_path, exist_ok=True)

        # TF listener para transformaciones de coordenadas
        self.tf_listener = tf.TransformListener()

        # Inicializar variables del mapa
        self.map_data = None
        self.map_info = None

        # Si se proporciona un archivo de mapa, cargarlo directamente
        if map_file_path:
            rospy.loginfo(f"Loading map from provided path: {map_file_path}")
            success = self.load_map_from_file(map_file_path)
            if success:
                rospy.loginfo("Map loaded successfully, generating object map")
                self.generate_object_map()
            else:
                rospy.logerr(f"Failed to load map from {map_file_path}")
                # No salimos, permitimos que el script de llamada maneje el error

    @staticmethod
    def find_latest_map(base_dir):
        """
        Encuentra el mapa más reciente en la estructura de directorios.

        Args:
            base_dir: Directorio base donde buscar los mapas

        Returns:
            Ruta al archivo PGM del mapa más reciente o None si no se encuentra
        """
        try:
            base_path = Path(os.path.expanduser(base_dir))
            if not base_path.exists():
                rospy.logwarn(f"El directorio base {base_dir} no existe")
                return None

            # Buscar directorios de sesión de mapa recientes
            session_dirs = sorted(base_path.glob(
                "map_session_*"), reverse=True)

            for session_dir in session_dirs:
                if not session_dir.is_dir():
                    continue

                # Primero buscar mapas finales
                final_maps_dir = session_dir / "final_maps"

                if final_maps_dir.exists():
                    final_maps = sorted(final_maps_dir.glob(
                        "final_map_*.pgm"), reverse=True)
                    if final_maps:
                        map_path = final_maps[0]
                        yaml_path = map_path.with_suffix('.yaml')

                        if yaml_path.exists():
                            rospy.loginfo(f"Found saved map: {map_path}")
                            return str(map_path)

            # Si no hay mapas finales, buscar mapas periódicos
            for session_dir in session_dirs:
                periodic_maps_dir = session_dir / "periodic_maps"

                if periodic_maps_dir.exists():
                    periodic_maps = sorted(
                        periodic_maps_dir.glob("map_*.pgm"), reverse=True)
                    if periodic_maps:
                        map_path = periodic_maps[0]
                        yaml_path = map_path.with_suffix('.yaml')

                        if yaml_path.exists():
                            rospy.loginfo(
                                f"Found saved periodic map: {map_path}")
                            return str(map_path)

            rospy.logwarn("No se encontraron mapas guardados")
            return None

        except Exception as e:
            rospy.logerr(f"Error buscando mapas guardados: {e}")
            return None

    def load_map_from_file(self, pgm_path):
        """Carga el mapa desde archivos PGM y YAML"""
        try:
            # Cargar el archivo YAML con los metadatos del mapa
            yaml_path = pgm_path.replace('.pgm', '.yaml')
            with open(yaml_path, 'r') as yaml_file:
                map_metadata = yaml.safe_load(yaml_file)

            # Cargar el archivo PGM con la imagen del mapa
            map_img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)

            if map_img is None:
                rospy.logerr(f"Failed to load map image from {pgm_path}")
                return False

            # Crear estructura de información del mapa
            self.map_info = {
                'resolution': map_metadata['resolution'],
                'origin': map_metadata['origin'],
                'width': map_img.shape[1],
                'height': map_img.shape[0]
            }

            # Convertir imagen a formato OccupancyGrid
            self.map_data = np.zeros(map_img.shape, dtype=np.int8)
            # Negro (0) en PGM es ocupado (100) en OccupancyGrid
            self.map_data[map_img == 0] = 100
            # Blanco (254) en PGM es libre (0) en OccupancyGrid
            self.map_data[map_img == 254] = 0
            # Gris (205) en PGM es desconocido (-1) en OccupancyGrid
            self.map_data[map_img == 205] = -1

            rospy.loginfo(f"Successfully loaded map from {pgm_path}")
            return True

        except Exception as e:
            rospy.logerr(f"Error loading map from file: {e}")
            return False

    def parse_timestamp(self, timestamp_str):
        """Parse timestamp from various possible formats"""
        timestamp_formats = [
            lambda ts: datetime.fromisoformat(ts) if 'T' in ts else None,
            lambda ts: datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S.%f"),
            lambda ts: datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S"),
            lambda ts: datetime.fromtimestamp(float(ts))
        ]

        for parser in timestamp_formats:
            try:
                result = parser(timestamp_str)
                if result:
                    return result
            except (ValueError, TypeError):
                continue

        rospy.logwarn(
            f"Failed to parse timestamp: {timestamp_str}, using current time")
        return datetime.now()

    def load_yolo_detections(self):
        """Carga todos los archivos de detección YOLO"""
        detections = []
        yolo_detections_base_path = os.path.expanduser(
            rospy.get_param("~yolo_detections_base_path", "~/detections"))

        try:
            # Buscar directorios con timestamp
            timestamp_dirs = list(Path(yolo_detections_base_path).glob("*"))

            for dir_path in timestamp_dirs:
                if not dir_path.is_dir():
                    continue

                # Buscar archivos JSON de metadatos
                json_dir = dir_path / "json"
                if not json_dir.exists():
                    continue

                metadata_files = list(json_dir.glob("metadata_*.json"))

                for file_path in metadata_files:
                    try:
                        with open(file_path, 'r') as f:
                            detections.append(json.load(f))
                    except Exception as e:
                        rospy.logerr(
                            f"Error loading detection file {file_path}: {e}")

            rospy.loginfo(f"Loaded {len(detections)} YOLO detection sets")
        except Exception as e:
            rospy.logerr(f"Error searching for detection files: {e}")

        return detections

    def normalize_detection(self, detection_set):
        """Normaliza los diferentes formatos de detección"""
        normalized_detections = []
        timestamp_str = detection_set.get('timestamp')

        if not timestamp_str:
            return []

        capture_time = rospy.Time.from_sec(
            self.parse_timestamp(timestamp_str).timestamp())

        # Formato 1: position_2d directo
        if 'position_2d' in detection_set and 'class' in detection_set:
            center = detection_set['position_2d']
            depth = detection_set.get('depth')
            class_name = detection_set['class']
            confidence = detection_set.get('confidence', 1.0)

            if depth and depth > 0 and confidence >= self.min_confidence:
                img_w, img_h = detection_set.get(
                    'image_size', [DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT])
                x_norm = (center[0] - img_w / 2) / (img_w / 2)
                angle_h = x_norm * FOV_H / 2

                normalized_detections.append({
                    'class': class_name,
                    'angle_h': angle_h,
                    'depth': depth,
                    'capture_time': capture_time,
                    'confidence': confidence
                })

        # Formato 2: camera_parameters y detections
        elif 'camera_parameters' in detection_set and 'detections' in detection_set:
            camera_params = detection_set['camera_parameters']
            image_size = camera_params.get(
                'resolution', [DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT])

            for detection in detection_set['detections']:
                class_name = detection['class']
                confidence = detection.get('confidence', 0)
                center_px = detection['center']
                depth = detection.get('depth')

                if confidence < self.min_confidence:
                    continue

                if not depth or depth <= 0:
                    depth = DEFAULT_ESTIMATED_DEPTH

                x_norm = (center_px[0] - image_size[0] /
                          2) / (image_size[0] / 2)
                y_norm = (image_size[1] / 2 - center_px[1]
                          ) / (image_size[1] / 2)
                angle_h = x_norm * FOV_H / 2
                angle_v = y_norm * FOV_V / 2

                normalized_detections.append({
                    'class': class_name,
                    'image_point': center_px,
                    'image_size': image_size,
                    'depth': depth,
                    'angle_h': angle_h,
                    'angle_v': angle_v,
                    'capture_time': capture_time,
                    'confidence': confidence
                })

        # Formato 3: center, class y depth
        elif 'class' in detection_set and 'depth' in detection_set and 'center' in detection_set:
            class_name = detection_set['class']
            depth = detection_set['depth']
            center_px = detection_set['center']
            confidence = detection_set.get('confidence', 1.0)

            if depth > 0 and confidence >= self.min_confidence:
                image_size = [DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT]
                x_norm = (center_px[0] - image_size[0] /
                          2) / (image_size[0] / 2)
                y_norm = (image_size[1] / 2 - center_px[1]
                          ) / (image_size[1] / 2)
                angle_h = x_norm * FOV_H / 2
                angle_v = y_norm * FOV_V / 2

                normalized_detections.append({
                    'class': class_name,
                    'image_point': center_px,
                    'image_size': image_size,
                    'depth': depth,
                    'angle_h': angle_h,
                    'angle_v': angle_v,
                    'capture_time': capture_time,
                    'confidence': confidence
                })

        return normalized_detections

    def transform_point_to_map(self, detection):
        """Transforma un punto a coordenadas del mapa"""
        try:
            capture_time = detection['capture_time']
            self.tf_listener.waitForTransform(
                "/map", "/base_link", capture_time, rospy.Duration(4.0))
            trans, rot = self.tf_listener.lookupTransform(
                "/map", "/base_link", capture_time)
            yaw = tf.transformations.euler_from_quaternion(rot)[2]

            # Calcular componentes forward y side
            angle_h = detection['angle_h']
            depth = detection['depth']

            if 'angle_v' in detection:
                angle_v = detection['angle_v']
                forward = depth * math.cos(angle_h) * math.cos(angle_v)
            else:
                forward = depth * math.cos(angle_h)

            side = depth * math.sin(angle_h)

            # Calcular posición en el mapa
            x_map = trans[0] + forward * math.cos(yaw) - side * math.sin(yaw)
            y_map = trans[1] + forward * math.sin(yaw) + side * math.cos(yaw)

            return (x_map, y_map)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error: {e}")
            return None

    def generate_object_map(self):
        """Función principal para generar el mapa de objetos"""
        try:
            map_img = self.process_map_image()
            detection_sets = self.load_yolo_detections()
            object_positions = {}

            if not detection_sets:
                rospy.logwarn(
                    "No detections found, generating empty object map")

            # Procesar todas las detecciones
            for detection_set in detection_sets:
                try:
                    normalized_detections = self.normalize_detection(
                        detection_set)

                    for detection in normalized_detections:
                        class_name = detection['class']
                        map_point = self.transform_point_to_map(detection)

                        if map_point is not None:
                            object_positions.setdefault(
                                class_name, []).append(map_point)

                except Exception as e:
                    rospy.logwarn(f"Error processing detection: {e}")
                    continue

            # Agrupar detecciones similares
            clustered_objects = self.cluster_object_detections(
                object_positions)

            # Generar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.create_annotated_map(map_img, clustered_objects, timestamp)
            self.save_object_data(clustered_objects, timestamp)
            self.generate_visualization(clustered_objects, timestamp)

        except Exception as e:
            rospy.logerr(f"Error in generate_object_map: {e}")

    def process_map_image(self):
        """Convierte los datos del mapa cargado a imagen"""
        if self.map_data is None or self.map_info is None:
            rospy.logerr("No map data available for processing")
            return None

        height, width = self.map_data.shape
        map_img = np.zeros((height, width, 3), dtype=np.uint8)

        map_img[self.map_data == -1] = [128, 128, 128]  # Gris para desconocido
        map_img[self.map_data == 0] = [255, 255, 255]   # Blanco para libre
        map_img[self.map_data == 100] = [0, 0, 0]       # Negro para ocupado

        return map_img

    def cluster_object_detections(self, positions_by_class):
        """Agrupa detecciones similares con DBSCAN"""
        clustered_positions = {}

        for class_name, positions in positions_by_class.items():
            if len(positions) < 2:
                clustered_positions[class_name] = positions
                continue

            positions_array = np.array(positions)
            clustering = DBSCAN(eps=self.cluster_distance,
                                min_samples=1).fit(positions_array)
            labels = clustering.labels_

            clusters = {}
            for i, label in enumerate(labels):
                clusters.setdefault(label, []).append(positions[i])

            clustered_positions[class_name] = []
            for cluster_positions in clusters.values():
                if len(cluster_positions) >= self.min_detections:
                    avg_x = sum(p[0] for p in cluster_positions) / \
                        len(cluster_positions)
                    avg_y = sum(p[1] for p in cluster_positions) / \
                        len(cluster_positions)
                    clustered_positions[class_name].append((avg_x, avg_y))

        return clustered_positions

    def create_annotated_map(self, map_img, object_positions, timestamp):
        """Crea el mapa con objetos anotados usando OpenCV (sin mostrar en pantalla)"""
        if map_img is None:
            rospy.logerr("No map image available for annotation")
            return

        annotated_map = map_img.copy()

        for class_name, positions in object_positions.items():
            color = self.class_colors.get(
                class_name.lower(), self.class_colors['default'])

            for (x, y) in positions:
                try:
                    # Convertir coordenadas del mundo a píxeles
                    origin_x = self.map_info['origin'][0]
                    origin_y = self.map_info['origin'][1]
                    resolution = self.map_info['resolution']

                    px = int((x - origin_x) / resolution)
                    py = int((y - origin_y) / resolution)

                    if 0 <= px < map_img.shape[1] and 0 <= py < map_img.shape[0]:
                        radius = int(self.object_marker_size / resolution)
                        cv2.circle(annotated_map, (px, py), radius, color, -1)

                        font_scale = 0.5
                        thickness = 1
                        text_size = cv2.getTextSize(
                            class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                        text_x = px - text_size[0] // 2
                        text_y = py - radius - 5

                        if 0 <= text_x < map_img.shape[1] and 0 <= text_y < map_img.shape[0]:
                            cv2.putText(annotated_map, class_name, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
                            cv2.putText(annotated_map, class_name, (text_x, text_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                except Exception as e:
                    rospy.logerr(f"Error drawing object on map: {e}")

        # Guardar resultado sin mostrar
        output_file = os.path.join(
            self.output_path, f"object_map_{timestamp}.png")
        cv2.imwrite(output_file, annotated_map)
        rospy.loginfo(f"Saved annotated object map to {output_file}")

    def save_object_data(self, object_positions, timestamp):
        """Guarda datos de posición de objetos en formato JSON"""
        if self.map_info is None:
            rospy.logerr("No map info available")
            return

        object_data = {
            'map_resolution': self.map_info['resolution'],
            'map_origin': {
                'x': self.map_info['origin'][0],
                'y': self.map_info['origin'][1]
            },
            'objects': {class_name: [list(pos) for pos in positions]
                        for class_name, positions in object_positions.items()}
        }

        json_file = os.path.join(
            self.output_path, f"object_positions_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(object_data, f, indent=2)

        rospy.loginfo(f"Saved object position data to {json_file}")

    def generate_visualization(self, object_positions, timestamp):
        """Genera visualización de alta calidad con Matplotlib (sin mostrar en pantalla)"""
        try:
            if self.map_info is None:
                rospy.logerr("No map info available for visualization")
                return

            # Usar el backend Agg que no requiere interfaz gráfica
            plt.switch_backend('Agg')

            fig, ax = plt.subplots(figsize=(12, 12))
            map_width = self.map_info['width'] * self.map_info['resolution']
            map_height = self.map_info['height'] * self.map_info['resolution']
            origin_x = self.map_info['origin'][0]
            origin_y = self.map_info['origin'][1]

            ax.set_xlim(origin_x, origin_x + map_width)
            ax.set_ylim(origin_y, origin_y + map_height)

            for class_name, positions in object_positions.items():
                color = self.class_mpl_colors.get(
                    class_name.lower(), self.class_mpl_colors['default'])

                for x, y in positions:
                    circle = Circle(
                        (x, y), self.object_marker_size / 2, color=color, alpha=0.7)
                    ax.add_patch(circle)
                    ax.text(x, y + self.object_marker_size / 2 + 0.1, class_name,
                            ha='center', va='bottom', fontsize=9)

            ax.set_title('Object Map Visualization')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.grid(True)

            output_file = os.path.join(
                self.output_path, f"object_map_viz_{timestamp}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            rospy.loginfo(f"Saved matplotlib visualization to {output_file}")
        except Exception as e:
            rospy.logerr(f"Error generating matplotlib visualization: {e}")


def main():
    try:
        # Inicializar el nodo ROS
        rospy.init_node("object_map_generator", anonymous=True)

        # Verificar si se proporcionó una ruta de mapa como argumento
        map_path = None
        if len(sys.argv) > 1:
            map_path = sys.argv[1]
            if not os.path.exists(map_path):
                rospy.logerr(
                    f"El archivo de mapa especificado no existe: {map_path}")
                sys.exit(1)
        else:
            # Buscar el mapa más reciente
            maps_base_dir = os.path.expanduser(
                rospy.get_param("~maps_base_dir", "~/saved_maps"))
            map_path = ObjectMapGenerator.find_latest_map(maps_base_dir)

            if map_path is None:
                rospy.logerr(
                    "No se encontró ningún mapa. Especifique la ruta del mapa como argumento.")
                sys.exit(1)

        # Inicializar el generador con la ruta del mapa
        generator = ObjectMapGenerator(map_path)
        rospy.loginfo("Object map generation complete.")

    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception in object map generator.")
    except Exception as e:
        rospy.logerr(f"Error in object map generator: {e}")


if __name__ == '__main__':
    main()
