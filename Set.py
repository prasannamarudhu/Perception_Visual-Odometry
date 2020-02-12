import cv2
import glob
from ReadCameraModel import *
from UndistortImage import *

i = 0
for img in glob.glob("stereo/centre/*.png"):
    image = cv2.imread(img,0)
    color_image = cv2.cvtColor(image, cv2.COLOR_BayerGR2BGR)
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('.\model')
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    undistorted_image = UndistortImage(color_image, LUT)
    filtered_image = cv2.GaussianBlur(undistorted_image, (5, 5), 0)
    #cv2.imwrite('Data/Final_Image{}.png'.format(i),filtered_image)
    #i = i+1

print(K)







