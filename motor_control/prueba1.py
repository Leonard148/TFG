#!/usr/bin/env python3
import serial
import time
import struct

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

try:
    arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    time.sleep(2)  # Esperar a que el puerto se inicialice

    # Construir y enviar comando HDLC para avanzar
    command = build_hdlc_frame("M:200:200:200:200")
    print(f"Enviando HDLC: {command}")
    arduino.write(command)
    arduino.flush()

    # Leer respuesta del Arduino
    response = arduino.readline()
    print(f"Respuesta: {response}")

    # Esperar 2 segundos antes de detener motores
    time.sleep(2)

    # Construir y enviar comando HDLC para detener motores
    stop_command = build_hdlc_frame("M:0:0:0:0")
    print(f"Enviando HDLC: {stop_command}")
    arduino.write(stop_command)
    arduino.flush()

    # Cerrar conexión
    arduino.close()

except serial.SerialException as e:
    print(f"⚠️ Error al abrir el puerto serie: {e}")
