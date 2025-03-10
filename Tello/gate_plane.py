from djitellopy import Tello
import cv2
import numpy
from video_to_gate_centre import gate_position_cf

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
parameters
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
mat = numpy.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
dist = numpy.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

tello = Tello()

tello.connect()

tello.streamon()
frame_read = tello.get_frame_read()
#tello.takeoff()

while True:
    img = frame_read.frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    cv2.imshow('Detected Markers', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if len(corners) < 2:
        continue
    translate = gate_position_cf(corners)
    print(translate)


tello.land()

    