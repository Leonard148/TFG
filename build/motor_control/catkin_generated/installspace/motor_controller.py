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
       
        # Motor safety parameters
        self.max_speed = rospy.get_param("~max_speed", 200)
        self.min_speed = rospy.get_param("~min_speed", 100)
       
        # Eliminar el uso del command_timeout para evitar cortes de comunicación
        # self.command_timeout = rospy.get_param("~command_timeout", 2.0)  
       
        self.last_command = "M:0:0:0:0"  # Almacenar el último comando enviado

        try:
            self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=self.timeout)
            time.sleep(2)  # Esperar inicialización del puerto serie
            rospy.loginfo(f"✅ Conectado a {self.serial_port} a {self.baud_rate} baudios")
        except serial.SerialException as e:
            rospy.logerr(f"⚠️ Error al abrir el puerto serie: {e}")
            exit(1)

        # Suscribirse a los tópicos de comandos
        self.raw_cmd_sub = rospy.Subscriber("/robot/move/raw", String, self.raw_command_callback)
        self.direction_cmd_sub = rospy.Subscriber("/robot/move/direction", String, self.direction_callback)
       
        # Eliminar el timer de seguridad que verifica timeouts
        # self.safety_timer = rospy.Timer(rospy.Duration(0.5), self.check_safety)
       
        rospy.loginfo("📡 Controlador de motores listo")

    def raw_command_callback(self, msg):
        """Procesa comandos de motor en crudo"""
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
           
        hdlc_command = build_hdlc_frame(command_str)
       
        rospy.loginfo(f"➡️ Enviando comando: {command_str}")
        self.arduino.write(hdlc_command)
        self.arduino.flush()
       
        # Leer respuesta del Arduino
        self.read_arduino_response()

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
        except UnicodeDecodeError:
            rospy.logwarn("Respuesta del Arduino no válida (no UTF-8)")

    # Se elimina el método check_safety que detenía los motores por timeout


    def send_stop_command(self):
        """Envía comando de parada de emergencia"""
        stop_cmd = "M:0:0:0:0"
        hdlc_stop = build_hdlc_frame(stop_cmd)
        self.arduino.write(hdlc_stop)
        self.arduino.flush()
        rospy.loginfo("Comando de parada enviado")

    def shutdown(self):
        """Apagado limpio"""
        rospy.loginfo("Apagando controlador de motores...")
        self.send_stop_command()
        self.arduino.close()

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
