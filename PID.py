# Coded von Enzo Gomes Frese
# November Frühling 
# Dofbot + OpenCV

#!/usr/bin/env python3
#coding=utf-8

# Lib
import time
import cv2
import numpy as np
import time
from Arm_Lib import Arm_Device

# Aufwecken des Roboters
Arm = Arm_Device()
Arm.Arm_serial_servo_write(1, 90, 200)
time.sleep(0.1)
Arm.Arm_serial_servo_write(2, 90, 500)
time.sleep(0.1)
Arm.Arm_serial_servo_write(3, 90, 500)
time.sleep(0.1)
Arm.Arm_serial_servo_write(4,  0, 200)
time.sleep(0.1)
Arm.Arm_serial_servo_write(5, 90, 200)
time.sleep(0.1)

#HSV Range for RED 
lowerbound=np.array([170,70,50])
upperbound=np.array([180,255,255])

#Webcam
cam= cv2.VideoCapture(0)

#PID initialisation 
desired_posn = 200 #Center of the video
kp=0.4
ki=0
kd=0.1
previous_error=0
timenow=0
pid_i=0 #Constant initial value

while True:
 # Getting image from video
 ret, img=cam.read()
 # Resizing the image
 img=cv2.resize(img,(400,300))
 # Smoothning image using GaussianBlur
 imgblurred=cv2.GaussianBlur(img,(11,11),0)
 # Converting image to HSV format
 imgHSV=cv2.cvtColor(imgblurred,cv2.COLOR_BGR2HSV) #source:https://thecodacus.com/opencv-object-tracking-colour-detection-python/#.Wz9tQN6Wl_k
 # Masking red color
 mask=cv2.inRange(imgHSV,lowerbound,upperbound) #source:https://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
 # Removing Noise from the mask
 mask = cv2.erode(mask, None, iterations=2)
 mask = cv2.dilate(mask, None, iterations=2)
 # Extracting contour
 cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
 # Drawing Contour		
 cv2.drawContours(img,cnts,-1,(255,0,0),3)
 #Processing each contour
 for c in cnts:  #source: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
  # compute the center of the  maximum area contour
  m=max(cnts,key=cv2.contourArea) #finding the contour with maximum area
  M = cv2.moments(m)
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])
  # Drawing the max area contour and center of the shape on the image
  cv2.drawContours(img, [m], -1, (0, 255, 0), 2)
  cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
  cv2.putText(img, "center", (cX - 20, cY - 20),
   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  #Drawing a vertical central line with RED color(BGR)
  cv2.line(img,(200,0),(200,300),(0,0,255),2)
  #Drawing a vertical line at the centre with Blue color
  cv2.line(img,(cX,0),(cX,300),(255,0,0),2)
  #Displaying mask
 cv2.imshow("mask",mask)
  #Displaying image
 cv2.imshow("cam",img)
  
#PID calcuation
 error=(cX-desired_posn)/1.5
#Proportional
 pid_p=kp*error
#Integral
 if -30<error<30:
  pid_i=pid_i+(ki*error)
#Derivative
 time_previous=timenow
 timenow=time.time()
 elapsedTime=timenow-time_previous
 pid_d=kd*((error-previous_error)/elapsedTime)
 previous_error=error
 PID=pid_p+pid_i+pid_d
 servo_signal=90-PID
  
 if servo_signal<=5:
  servo_signal=5 
 if servo_signal>=170:
  servo_signal=170 
  
 print(servo_signal)# Debug

 #Press "q" to end the loop
 if cv2.waitKey(1) & 0xFF == ord('q'):
       break

#Sending signal to the servo
 Arm.Arm_serial_servo_write(5, servo_signal,400)

