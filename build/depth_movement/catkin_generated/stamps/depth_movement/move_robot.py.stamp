#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rtabmap_msgs.msg import MapData
from cv_bridge import CvBridge

class AutoExplorer:
    def __init__(self):
        rospy.init_node("auto_explorer", anonymous=True)
       
        # Suscripción a la imagen de profundidad
        self.depth_sub = rospy.Subscriber("/k4a/depth/image_resized", Image, self.depth_callback)

        # Suscripción a los datos del mapa
        self.map_sub = rospy.Subscriber("/rtabmap/mapData", MapData, self.map_callback)

        # Publicador de comandos de movimiento
        self.cmd_pub = rospy.Publisher("/robot/move", String, queue_size=10)

        self.bridge = CvBridge()
        self.min_safe_distance = 2.0  # Mínima distancia segura (metros)
        self.last_map_size = 0
        self.stop_count = 0  # Contador para verificar si el mapa está completo
        self.max_stop_checks = 10  # Número de verificaciones antes de detenerse
        self.running = True

    def depth_callback(self, msg):
        """ Procesa la imagen de profundidad y decide el movimiento """
        try:
            # Convertir imagen ROS a OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")

            # Tomar la distancia central
            height, width = depth_image.shape
            center_x, center_y = width // 2, height // 2
            distance = depth_image[center_y, center_x] / 1000.0  # Convertir a metros

            rospy.loginfo(f"Distancia frontal: {distance:.2f} m")

            if self.running:
                if distance > self.min_safe_distance:
                    self.cmd_pub.publish("M:200:200:200:200")  # Avanzar
                else:
                    self.cmd_pub.publish("M:0:0:0:0")  # Detenerse
                    rospy.sleep(1)
                    self.cmd_pub.publish("M:-200:-200:200:200")  # Girar
                    rospy.sleep(1)
                    self.cmd_pub.publish("M:0:0:0:0")  # Parar
        except Exception as e:
            rospy.logerr(f"Error procesando imagen de profundidad: {e}")

    def map_callback(self, msg):
        """ Verifica el progreso del mapeo y detiene el robot si el mapa está completo """
        current_map_size = len(msg.data)

        # Verificar si el mapa dejó de crecer
        if current_map_size == self.last_map_size:
            self.stop_count += 1
            rospy.loginfo(f"El mapa no ha cambiado ({self.stop_count}/{self.max_stop_checks})")
        else:
            self.stop_count = 0  # Reiniciar contador si el mapa sigue creciendo

        self.last_map_size = current_map_size

        # Si no hay cambios después de varias verificaciones, detener el robot
        if self.stop_count >= self.max_stop_checks:
            self.running = False
            self.cmd_pub.publish("M:0:0:0:0")
            rospy.loginfo("Exploración completada. Mapa terminado.")

    def shutdown(self):
        """ Detener el robot al cerrar el nodo """
        self.running = False
        self.cmd_pub.publish("M:0:0:0:0")
        rospy.loginfo("Exploración automática detenida.")

if __name__ == "__main__":
    try:
        explorer = AutoExplorer()
        rospy.on_shutdown(explorer.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


