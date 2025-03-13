from djitellopy import Tello
import cv2
import numpy as np
from video_to_gate_centre import gate_position_cf
import time

# Set this to true if testing on webcam
WEBCAM = False
# Set this to True to enable takeoff and gate traversal
FLIGHT = True

# Seconds for test flight before auto-land
TIMEOUT_STOP = 20

# Ensure linear velocity is below this for safety
V_PERCENT_MAX = 20

# State machine states
S0_IDLE = 0
S1_NO_GATE = 1
S2_CENTRE_GATE = 2
gate_centred = False
S3_TRAVERSE = 3

state = S0_IDLE

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
mat = np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]])
dist = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000])

if WEBCAM:
    cam = cv2.VideoCapture(2)
else:
    tello = Tello()
    tello.connect()
    tello.streamon()
    frame_read = tello.get_frame_read()

if FLIGHT:
    tello.takeoff()
    tello.send_rc_control( int(0), int(0), int(0), int(0))
    tello.move_up(50)
    t_takeoff = time.time()

height = 0

while True:

    # Read camera and process frame as required
    if WEBCAM:
        ret, img = cam.read()
    else:
        img = frame_read.frame
        height = tello.get_height()
    if img is None:
        raise ValueError("frame not loaded.")
    if not WEBCAM:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Look for aruco markers
    corners, ids, rejected = detector.detectMarkers(gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # State transitions
    if state == S0_IDLE:
        state = S1_NO_GATE

    elif state == S1_NO_GATE:
        if len(corners) >= 2:
            state = S2_CENTRE_GATE
    
    elif state == S2_CENTRE_GATE:
        if gate_centred:
            gate_centred = False
            state = S3_TRAVERSE

    elif state == S3_TRAVERSE:
        pass
    
    translate = np.array([0,0])
    if len(corners) >= 2:
        gate_centre, pt1, pt2 = gate_position_cf(corners)
        # Draw line to gate centre
        video_centre = np.array([gray.shape[1],gray.shape[0]])//2
        img = cv2.line(img, video_centre, gate_centre, color=(0,0,255), thickness=5)
        img = cv2.circle(img, gate_centre,10, color=(0,255,0),thickness = -1)

        img = cv2.circle(img, pt1, 10, color=(255,0,0),thickness = -1)
        img = cv2.circle(img, pt2, 10, color=(0,0,255),thickness = -1)

        translate = gate_centre - video_centre
        print(f'Translate: {translate}')

    cv2.imshow('Detected Markers', img)
    

    if FLIGHT:
        print(f'state: {state} t: {time.time()-t_takeoff:.2f}s')
        if time.time() - t_takeoff >= TIMEOUT_STOP:
            print('Flight time over - landing...')
            break

    if state == S1_NO_GATE:
        # Spin until gate found

        v_lr = 0 # left_right_velocity
        v_fb = 0 # forward_backward_velocity
        v_ud = 0 # up_down_velocity
        w_yaw = 30 # yaw_velocity
        print(state)
        tello.send_rc_control( int(v_lr), int(v_fb), int(v_ud), int(w_yaw))
    
    # P-control navigation to centre the gate
    Kp = 0.08 # Proportional gain

    if state == S2_CENTRE_GATE:
        if len(corners) >= 2:
            # Update velocity RC channels -100 to 100
            v_lr = np.sign(translate[0]) * min(Kp*abs(translate[0]),V_PERCENT_MAX) # left_right_velocity
            v_fb = 15 # forward_backward_velocity
            v_ud = -np.sign(translate[1]) * min(Kp*abs(translate[1]),V_PERCENT_MAX) # up_down_velocity
            w_yaw = 0 # yaw_velocity
            print(state)
            print(f'v_lr = {v_lr}')
            tello.send_rc_control( int(v_lr), int(v_fb), int(v_ud), int(w_yaw))



# Exit gracefully
if WEBCAM:
    pass
else:
    tello.streamoff()

if FLIGHT:
    tello.land()

    