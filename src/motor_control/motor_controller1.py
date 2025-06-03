#!/usr/bin/env python3
"""
Motor Controller Node para ROS
Este nodo se encarga de controlar los motores del robot a través de un Arduino
utilizando comunicación serie con protocolo HDLC.
"""
import rospy
import serial
import time
import struct
from std_msgs.msg import String

# Constantes
HDLC_FLAG = b'\x7E'  # Delimitador HDLC
CRC_POLY = 0x1021    # Polinomio CRC-16 (CCITT-FALSE)
CRC_INIT = 0xFFFF    # Valor inicial para CRC-16
MOTOR_CMD_PREFIX = "M"  # Prefijo para comandos de motor

def crc16_ccitt(data: bytes, poly=CRC_POLY, init_val=CRC_INIT):
    """
    Calcula CRC-16 (CCITT-FALSE) para los datos proporcionados.
    
    Args:
        data: Bytes para calcular el CRC
        poly: Polinomio para CRC (por defecto 0x1021)
        init_val: Valor inicial (por defecto 0xFFFF)
        
    Returns:
        bytes: CRC calculado como bytes en orden big-endian
    """
    crc = init_val
    for byte in data:
        crc ^= (byte << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) & 0xFFFF) ^ poly
            else:
                crc = (crc << 1) & 0xFFFF
    return crc.to_bytes(2, 'big')

def build_hdlc_frame(payload: str) -> bytes:
    """
    Construye una trama HDLC a partir de un payload.
    
    Args:
        payload: Cadena de texto a encapsular en trama HDLC
        
    Returns:
        bytes: Trama HDLC completa con delimitadores y CRC
    """
    data = payload.encode('utf-8')
    crc = crc16_ccitt(data)
    return HDLC_FLAG + data + crc + HDLC_FLAG

class MotorController:
    """
    Controlador de motores para robot mediante comunicación serial.
    Suscribe a tópicos ROS y envía comandos al Arduino.
    """
    
    def __init__(self):
        """Inicializa el nodo ROS y configura la comunicación serial."""
        rospy.init_node('motor_controller', anonymous=True)

        # Cargar parámetros
        self._load_parameters()
        
        # Inicializar variables de estado
        self.last_command = "M:0:0:0:0"
        self.arduino = None
        
        # Conectar al puerto serie
        self._connect_serial()

        # Suscribirse a los tópicos de comandos
        self.raw_cmd_sub = rospy.Subscriber("/robot/move/raw", String, self.raw_command_callback)
        self.direction_cmd_sub = rospy.Subscriber("/robot/move/direction", String, self.direction_callback)
       
        rospy.loginfo("📡 Controlador de motores listo")

    def _load_parameters(self):
        """Carga parámetros desde el servidor de parámetros ROS."""
        # Parámetros de puerto serie
        self.serial_port = rospy.get_param("~serial_port", "/dev/ttyUSB0")
        self.baud_rate = rospy.get_param("~baud_rate", 9600)
        self.timeout = rospy.get_param("~serial_timeout", 1.0)
       
        # Parámetros de seguridad de los motores
        self.max_speed = rospy.get_param("~max_speed", 200)
        self.min_speed = rospy.get_param("~min_speed", 100)

    def _connect_serial(self):
        """Establece conexión con el puerto serie."""
        try:
            self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=self.timeout)
            time.sleep(2)  # Esperar inicialización del puerto serie
            rospy.loginfo(f"✅ Conectado a {self.serial_port} a {self.baud_rate} baudios")
        except serial.SerialException as e:
            rospy.logerr(f"⚠️ Error al abrir el puerto serie: {e}")
            rospy.signal_shutdown(f"Error de conexión serial: {e}")
            exit(1)

    def raw_command_callback(self, msg):
        """
        Procesa comandos de motor en crudo.
        
        Args:
            msg: Mensaje ROS con el comando para los motores
        """
        command_str = msg.data.strip()
       
        # Evitar enviar comandos repetidos para reducir tráfico
        if command_str == self.last_command:
            return
           
        # Actualizar el último comando
        self.last_command = command_str

        # Validar el formato del comando
        if not self.validate_command(command_str):
            rospy.logwarn(f"Comando inválido: {command_str}")
            return
           
        try:
            hdlc_command = build_hdlc_frame(command_str)
            
            rospy.loginfo(f"➡️ Enviando comando: {command_str}")
            self.arduino.write(hdlc_command)
            self.arduino.flush()
           
            # Leer respuesta del Arduino
            self.read_arduino_response()
        except Exception as e:
            rospy.logerr(f"Error al enviar comando: {e}")

    def direction_callback(self, msg):
        """
        Registra la dirección del movimiento.
        
        Args:
            msg: Mensaje ROS con la dirección de movimiento
        """
        rospy.loginfo(f"Dirección de movimiento: {msg.data}")

    def validate_command(self, command):
        """
        Valida que el comando tenga el formato correcto.
        
        Args:
            command: Comando a validar (formato "M:v1:v2:v3:v4")
            
        Returns:
            bool: True si el comando es válido, False en caso contrario
        """
        try:
            parts = command.split(':')
            
            # Verificar estructura básica
            if len(parts) != 5:
                rospy.logwarn(f"Comando con formato incorrecto: {command}. Se esperan 5 partes.")
                return False
                
            # Verificar prefijo
            if parts[0] != MOTOR_CMD_PREFIX:
                rospy.logwarn(f"Prefijo incorrecto: {parts[0]}. Se esperaba '{MOTOR_CMD_PREFIX}'")
                return False
                
            # Verificar valores de velocidad
            for i, speed in enumerate(parts[1:], 1):
                speed_int = int(speed)
                if abs(speed_int) > self.max_speed:
                    rospy.logwarn(f"Velocidad {i} fuera de rango: {speed_int}. Máximo permitido: {self.max_speed}")
                    return False
                    
            return True
        except ValueError as e:
            rospy.logwarn(f"Error de formato en comando: {e}")
            return False
        except Exception as e:
            rospy.logerr(f"Error inesperado al validar comando: {e}")
            return False

    def read_arduino_response(self):
        """Lee y procesa la respuesta del Arduino."""
        try:
            if self.arduino and self.arduino.is_open:
                response = self.arduino.readline().decode('utf-8').strip()
                if response:
                    rospy.loginfo(f"⬅️ Respuesta Arduino: {response}")
        except UnicodeDecodeError:
            rospy.logwarn("Respuesta del Arduino no válida (no UTF-8)")
        except serial.SerialException as e:
            rospy.logerr(f"Error de comunicación serial al leer respuesta: {e}")
        except Exception as e:
            rospy.logerr(f"Error inesperado al leer respuesta: {e}")

    def send_stop_command(self):
        """Envía comando de parada de emergencia."""
        try:
            stop_cmd = "M:0:0:0:0"
            hdlc_stop = build_hdlc_frame(stop_cmd)
            
            if self.arduino and self.arduino.is_open:
                self.arduino.write(hdlc_stop)
                self.arduino.flush()
                rospy.loginfo("Comando de parada enviado")
            else:
                rospy.logwarn("No se pudo enviar comando de parada: puerto serial cerrado")
        except Exception as e:
            rospy.logerr(f"Error al enviar comando de parada: {e}")

    def shutdown(self):
        """Apagado limpio del controlador."""
        rospy.loginfo("Apagando controlador de motores...")
        self.send_stop_command()
        
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            rospy.loginfo("Puerto serial cerrado correctamente")

    def run(self):
        """Bucle principal del nodo."""
        rospy.on_shutdown(self.shutdown)
        rospy.spin()

if __name__ == "__main__":
    try:
        controller = MotorController()
        controller.run()
    except rospy.ROSInterruptException:
        if hasattr(controller, 'shutdown'):
            controller.shutdown()
    except Exception as e:
        rospy.logerr(f"Error inesperado: {e}")