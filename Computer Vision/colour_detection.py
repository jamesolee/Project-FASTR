# from picamera.array import PiRGBArray
# from picamera import PiCamera
# import time
import cv2
import numpy as np

# load image
img = cv2.imread('Computer Vision/test-images/middle.png')

# convert to hsv
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

# create mask for blue color in hsv
# blue is 120 in range 0 to 360, so for opencv it would be 60
lower = (20,100,100)
upper = (60,255,255)
mask = cv2.inRange(hsv, lower, upper)

kernel = np.ones((5, 5), np.uint8) 
img_erosion = cv2.erode(mask, kernel, iterations=10) 
img_dilation = cv2.dilate(img_erosion, kernel, iterations=10) 


# count non-zero pixels in mask
height, width = img_dilation.shape
l_count=np.count_nonzero(img_dilation[:, :round(width/2)])
print('left count:', l_count)
r_count=np.count_nonzero(img_dilation[:, round(width/2):])
print('right count:', r_count)
if (l_count-r_count) >= 50000: print("Go left, vector = -1")
elif (r_count-l_count) >= 50000: print("Go right, vector = 1")
else: print("Stay in centre, vector = 0")

# save output
cv2.imwrite('Computer Vision/test-images/left-filtered.png', img_dilation)

# display various images to see the steps
cv2.imshow('dilated',img_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()