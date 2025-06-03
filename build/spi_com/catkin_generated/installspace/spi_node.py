#!/usr/bin/env python3
import rospy
import spidev
from std_msgs.msg import Int32MultiArray  # Mensaje para enviar datos como lista

class SPIComNode:
    def __init__(self):
        # Iniciar nodo ROS
        rospy.init_node('spi_com_node', anonymous=True)
        self.spi_pub = rospy.Publisher('spi_data', Int32MultiArray, queue_size=10)
       
        # Configurar SPI
        self.spi = spidev.SpiDev()
        self.spi.open(1, 0)  # (bus, device)
        self.spi.max_speed_hz = 500000  # Velocidad del bus
       
        rospy.loginfo("SPI Node Initialized")

    def read_spi(self):
        while not rospy.is_shutdown():
            try:
                # Enviar y recibir datos (depende de tu placa controladora)
                response = self.spi.xfer2([0x00])  # Enviar un byte vacío y recibir
                msg = Int32MultiArray(data=response)
                self.spi_pub.publish(msg)
                rospy.loginfo(f"SPI Data: {response}")
            except Exception as e:
                rospy.logerr(f"SPI Error: {e}")
           
            rospy.sleep(0.1)  # Leer cada 100 ms

    def close(self):
        self.spi.close()

if __name__ == '__main__':
    try:
        node = SPIComNode()
        node.read_spi()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.close()


