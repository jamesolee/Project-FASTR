import picamera
import picamera.array
import cv2

# Initialize the camera
with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)  # Set the camera resolution as needed
    camera.framerate = 30  # Set the frame rate as needed

    with picamera.array.PiRGBArray(camera) as stream:
        # Allow some time for the camera to warm up
        camera.start_preview()

        # Main loop to capture and display frames
        for frame in camera.capture_continuous(stream, format="bgr", use_video_port=True):
            image = frame.array  # Convert the frame to a NumPy array
            cv2.imshow("Camera Feed", image)  # Display the frame

            # Exit the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Clean up
cv2.destroyAllWindows()
