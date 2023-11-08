import cv2
import numpy as np

# Load the default dictionary and create detector parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters_create()

# Initialize the video capture (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Define your camera intrinsic parameters
# DIM = (2592, 1944)
# K = np.array([[1257.9510336183691, 0.0, 1350.0415514897147],
#              [0.0, 1252.4955237784175, 923.7076920711385],
#              [0.0, 0.0, 1.0]])
# D = np.array([0.0007889742841409066, -0.4117067541364045, 2.032728410696636, -3.3188985820363777])

camera_matrix = np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]], dtype=np.float32)  # Adjust these values based on your camera
dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # May need to adjust if distortion is significant

while True:
    ret, frame = cap.read()

    # Detect markers in the frame
    corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

    if ids is not None:
        # Draw detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.1, camera_matrix, dist_coeffs)

            # Calculate distance to the marker
            distance = np.linalg.norm(tvec)

            # Calculate direction vector (assuming z-axis as forward)
            direction = tvec / distance

            # Convert the position to integers for drawing
            org = (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10)

            # Print the distance and direction
            x = float(direction[0][0][0])
            y = float(direction[0][0][1])
            z = float(direction[0][0][2])
            text = f"ID: {ids[i]}, Distance: {distance:.2f}, Direction: {x:.2f} {y:.2f}" # {z:.2f}
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            


    cv2.imshow("ArUco Marker Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
