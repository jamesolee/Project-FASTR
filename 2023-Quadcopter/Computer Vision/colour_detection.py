import cv2
import numpy as np

def colour_detect(img):
    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)

    # create mask for yellow color in hsv
    lower = (20,100,100)
    upper = (60,255,255)
    mask = cv2.inRange(hsv, lower, upper)

    # filter out background noises
    kernel = np.ones((5, 5), np.uint8) 
    img_erosion = cv2.erode(mask, kernel, iterations=10) 
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=10) 

    # count non-zero pixels
    height, width = img_dilation.shape
    l_count=np.count_nonzero(img_dilation[:, :round(width/2)])
    print('left count:', l_count)
    r_count=np.count_nonzero(img_dilation[:, round(width/2):])
    print('right count:', r_count)
    if (l_count-r_count) >= round(height*width/50): 
        # print("Go left, vector = -1")
        vector = -1
    elif (r_count-l_count) >= round(height*width/50): 
        # print("Go right, vector = 1")
        vector = 1
    else: 
        # print("Stay in centre, vector = 0")
        vector = 0
    
    return vector

def laptop_video_test():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv2.imshow('frame',frame)

        print("vector: " + colour_detect(frame))

        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    laptop_video_test()