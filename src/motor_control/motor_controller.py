#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist

def cmd_vel_callback(msg):
    rospy.loginfo(f"Velocidad recibida - Lineal: {msg.linear.x}, Angular: {msg.angular.z}")
    # Aquí debes agregar el código para enviar señales al controlador

def motor_controller():
    rospy.init_node('motor_controller', anonymous=True)
    rospy.Subscriber('/cmd_vel', Twist, cmd_vel_callback)
    rospy.spin()

if __name__ == '__main__':
    try:
        motor_controller()
    except rospy.ROSInterruptException:
         pass

