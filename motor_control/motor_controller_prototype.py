#!/usr/bin/env python3
import rospy
import serial
import time
import struct
from std_msgs.msg import String

# Función para calcular CRC-16 (CCITT-FALSE)
def crc16_ccitt(data: bytes, poly=0x1021, init_val=0xFFFF):
    crc = init_val
    for byte in data:
        for bit in range(8):
            xor_flag = (crc & 0x8000) != 0
            crc = (crc << 1) & 0xFFFF
            if byte & (1 << (7 - bit)):
                crc |= 1
            if xor_flag:
                crc ^= poly
    return crc.to_bytes(2, 'big')  # Devuelve CRC como bytes

# Función para construir una trama HDLC
def build_hdlc_frame(payload: str) -> bytes:
    frame_start = b'\x7E'  # Delimitador de inicio de HDLC
    data = payload.encode('utf-8')  # Convertir string a bytes
    crc = crc16_ccitt(data)  # Calcular CRC
    frame_end = b'\x7E'  # Delimitador de fin de HDLC
    return frame_start + data + crc + frame_end

class MotorController:
    def __init__(self):
        rospy.init_node('motor_controller', anonymous=True)

        # Configurar el puerto serie
        self.serial_port = rospy.get_param("~serial_port", "/dev/ttyUSB0")
        self.baud_rate = rospy.get_param("~baud_rate", 9600)
        self.timeout = rospy.get_param("~serial_timeout", 1.0)
       
        # Motor safety parameters - Use global parameter namespace for sharing
        self.max_speed = rospy.get_param("/robot/motor/max_speed", 200)
        self.min_speed = rospy.get_param("/robot/motor/min_speed", 100)
       
        # Restore command timeout safety feature
        self.command_timeout = rospy.get_param("~command_timeout", 2.0)
        self.last_command_time = rospy.Time.now()
       
        self.last_command = "M:0:0:0:0"  # Almacenar el último comando enviado
        self.connection_retries = 3  # Número de intentos para la conexión serial

        # Intentar establecer la conexión serial con reintentos
        self.connect_with_retry()

        # Suscribirse a los tópicos de comandos
        self.raw_cmd_sub = rospy.Subscriber("/robot/move/raw", String, self.raw_command_callback)
        self.direction_cmd_sub = rospy.Subscriber("/robot/move/direction", String, self.direction_callback)
       
        # Restaurar el timer de seguridad
        self.safety_timer = rospy.Timer(rospy.Duration(0.5), self.check_safety)
        
        # Publicador de estado para retroalimentación
        self.status_pub = rospy.Publisher("/robot/motor/status", String, queue_size=10)
       
        rospy.loginfo("📡 Controlador de motores listo")

    def connect_with_retry(self):
        """Establece la conexión serial con reintentos"""
        attempts = 0
        connected = False
        
        while attempts < self.connection_retries and not connected and not rospy.is_shutdown():
            try:
                self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=self.timeout)
                time.sleep(2)  # Esperar inicialización del puerto serie
                rospy.loginfo(f"✅ Conectado a {self.serial_port} a {self.baud_rate} baudios")
                connected = True
            except serial.SerialException as e:
                attempts += 1
                rospy.logwarn(f"⚠️ Intento {attempts}/{self.connection_retries} fallido: {e}")
                if attempts < self.connection_retries:
                    rospy.sleep(1)  # Esperar antes de reintentar
        
        if not connected:
            rospy.logerr("❌ No se pudo establecer la conexión serial después de varios intentos")
            rospy.signal_shutdown("Fallo en la conexión serial")
            exit(1)

    def raw_command_callback(self, msg):
        """Procesa comandos de motor en crudo"""
        command_str = msg.data.strip()
       
        # Actualizar el tiempo del último comando
        self.last_command_time = rospy.Time.now()
       
        # Evitar enviar comandos repetidos para reducir tráfico
        if command_str == self.last_command:
            return
           
        # Actualizar el último comando
        self.last_command = command_str

        # Validar el formato del comando
        if not self.validate_command(command_str):
            rospy.logwarn(f"Comando inválido: {command_str}")
            self.status_pub.publish(String("COMMAND_INVALID"))
            return
           
        # Enviar el comando con reintentos
        if self.send_command_with_retry(command_str):
            self.status_pub.publish(String("COMMAND_SENT"))
        else:
            self.status_pub.publish(String("COMMAND_FAILED"))

    def send_command_with_retry(self, command_str, max_retries=3):
        """Envía comando con reintento en caso de fallo"""
        retries = 0
        success = False
        
        while not success and retries < max_retries and not rospy.is_shutdown():
            try:
                hdlc_command = build_hdlc_frame(command_str)
                rospy.loginfo(f"➡️ Enviando comando: {command_str}")
                self.arduino.write(hdlc_command)
                self.arduino.flush()
                success = True
                
                # Leer respuesta del Arduino
                response = self.read_arduino_response()
                if response:
                    success = "OK" in response
                
            except serial.SerialException as e:
                retries += 1
                rospy.logwarn(f"⚠️ Error serial, reintento {retries}/{max_retries}: {e}")
                time.sleep(0.1)
                
                # Intentar reconectar si es necesario
                if retries == max_retries - 1:
                    try:
                        self.arduino.close()
                        time.sleep(1)
                        self.connect_with_retry()
                    except Exception as e:
                        rospy.logerr(f"❌ Error al reconectar: {e}")
        
        if not success:
            rospy.logerr(f"❌ Falló el envío del comando después de {max_retries} reintentos")
            return False
        return True

    def direction_callback(self, msg):
        """Registra la dirección del movimiento"""
        rospy.loginfo(f"Dirección de movimiento: {msg.data}")

    def validate_command(self, command):
        """Valida que el comando tenga el formato correcto"""
        try:
            parts = command.split(':')
            if len(parts) != 5:
                return False
            if parts[0] != "M":
                return False
            for speed in parts[1:]:
                speed_int = int(speed)
                if abs(speed_int) > self.max_speed:
                    rospy.logwarn(f"⚠️ Velocidad {speed_int} excede el máximo {self.max_speed}")
                    return False
            return True
        except ValueError:
            return False

    def read_arduino_response(self):
        """Lee y procesa la respuesta del Arduino"""
        try:
            response = self.arduino.readline().decode('utf-8').strip()
            if response:
                rospy.loginfo(f"⬅️ Respuesta Arduino: {response}")
                return response
        except UnicodeDecodeError:
            rospy.logwarn("Respuesta del Arduino no válida (no UTF-8)")
        except serial.SerialException as e:
            rospy.logwarn(f"Error al leer respuesta: {e}")
        return None

    def check_safety(self, event=None):
        """Verifica timeout de comandos y detiene motores si es necesario"""
        if (rospy.Time.now() - self.last_command_time).to_sec() > self.command_timeout:
            rospy.logwarn("⚠️ Timeout de comando - deteniendo motores")
            self.send_stop_command()

    def send_stop_command(self):
        """Envía comando de parada de emergencia"""
        stop_cmd = "M:0:0:0:0"
        self.send_command_with_retry(stop_cmd)
        rospy.loginfo("Comando de parada enviado")

    def shutdown(self):
        """Apagado limpio"""
        rospy.loginfo("Apagando controlador de motores...")
        self.send_stop_command()
        try:
            self.arduino.close()
            rospy.loginfo("Puerto serial cerrado correctamente")
        except Exception as e:
            rospy.logwarn(f"Error al cerrar puerto serial: {e}")

    def run(self):
        """Bucle principal"""
        rospy.on_shutdown(self.shutdown)
        rospy.spin()

if __name__ == "__main__":
    try:
        controller = MotorController()
        controller.run()
    except rospy.ROSInterruptException:
        controller.shutdown()