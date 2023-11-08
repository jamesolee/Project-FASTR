# You should replace these 3 lines with the output in calibration step
import numpy as np
import cv2
import sys

DIM=(2592, 1944)
K=np.array([[1257.9510336183691, 0.0, 1350.0415514897147], [0.0, 1252.4955237784175, 923.7076920711385], [0.0, 0.0, 1.0]])
D=np.array([[0.0007889742841409066], [-0.4117067541364045], [2.032728410696636], [-3.3188985820363777]])
def undistort(img_path):
    img = cv2.imread('cal2.jpg')
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)
