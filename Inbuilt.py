
import numpy as np
import cv2
from matplotlib import pyplot as plt


K = np.array([[964.828979  , 0.  ,     643.788025], [  0. ,     964.828979 ,484.40799 ], [  0.  ,       0.  ,       1.      ]])
R_t = np.identity(3)
T_t = np.array([[0],[0],[0]])
pos = np.array([[0],[0]])

for i in range(1,3872):

    img1 = cv2.imread('Data\Final_Image{}.png'.format(i))
    img2 = cv2.imread('Data\Final_Image{}.png'.format(i+1))
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)
    # print(matches)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    good = []

    for match in matches:
        good.append(match)

    pts1 = np.float32([kp1[match.queryIdx].pt for match in good])
    pts2 = np.float32([kp2[match.trainIdx].pt for match in good])

    E , mask = cv2.findEssentialMat(pts1,pts2,K,method= cv2.FM_RANSAC)

    points, R, t, mask = cv2.recoverPose(E, pts1, pts2,K)

    T_t = T_t + np.matmul(R_t,t)
    R_t = np.matmul(R_t , R)

    plt.scatter(-T_t[0],T_t[2])

    plt.pause(.0001)




