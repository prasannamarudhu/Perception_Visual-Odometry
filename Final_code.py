import numpy as np
import cv2
from matplotlib import pyplot as plt
from random import sample


def check(R, C):
	if np.linalg.det(R) < 0:
		return -R, -C

	else:
		return R, C


def RANSAC_fundamental(X1, X2):

	# converting the 2D img points to 3D
	X1_3d = np.hstack((X1, np.ones((X1.shape[0], 1))))
	X2_3d = np.hstack((X2, np.ones((X2.shape[0], 1))))

	# defining M iterations
	M = 600
	l = X1.shape[0]

	indices = [i for i in range(l)]
	num_in = -1
	in_indices = list()

	for i in range(M):

		x1_rand = np.empty((0, 2))
		x2_rand = np.empty((0, 2))

		# computing random indices
		rand_indices = sample(indices, 8)

		for r in rand_indices:
			x1_rand = np.append(x1_rand, [X1[r, :]], axis=0)
			x2_rand = np.append(x2_rand, [X2[r, :]], axis=0)

		F_i = get_fundamental_matrix(x1_rand, x2_rand)

		err = np.sum(np.matmul(X2_3d, F_i)*X1_3d, axis=1)
		condition = np.abs(err) < 0.006

		# storing the iterations when the condition is true
		index = np.where(condition)

		n = (index[0]).shape[0]
		if num_in < n:
			in_indices = index[0]
			num_in = n

	x1_in = np.empty((0, 2))
	x2_in = np.empty((0, 2))
	for i in in_indices:
		x1_in = np.append(x1_in, [[X1[i, 0], X1[i, 1]]], axis=0)
		x2_in = np.append(x2_in, [[X2[i, 0], X2[i, 1]]], axis=0)

	return x1_in, x2_in


def compute_P_from_essential(E):

	# make sure E is rank 2
	U, S, V = np.linalg.svd(E)
	E = np.dot(U, np.dot(np.diag([1, 1, 0]), V))

	U, S, V = np.linalg.svd(E)

	W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

	C1 = U[:, 2]
	C2 = -U[:, 2]
	C3 = U[:, 2]
	C4 = -U[:, 2]
	R1 = (np.dot(U, np.dot(W, V)))
	R2 = (np.dot(U, np.dot(W, V)))
	R3 = (np.dot(U, np.dot(W.T, V)))
	R4 = (np.dot(U, np.dot(W.T, V)))

	R1, C1 = check(R1, C1)
	R2, C2 = check(R2, C2)
	R3, C3 = check(R3, C3)
	R4, C4 = check(R4, C4)

	C = np.array([[C1], [C2], [C3], [C4]])
	R = np.array([[R1], [R2], [R3], [R4]])

	return C, R


def get_fundamental_matrix(x1, x2):

	P1, H1 = normalise(x1)
	P2, H2 = normalise(x2)

	n = x1.shape[0]

	if x2.shape[0] != n:
		raise ValueError("Number of points don't match.")

	# build matrix for equations
	A = np.vstack(([P1[:, 0]*P2[:, 0]], [P1[:, 1]*P2[:, 0]], [P1[:, 2]*P2[:, 0]],
		 [P1[:, 0]*P2[:, 1]], [P1[:, 1]*P2[:, 1]], [P1[:, 2]*P2[:, 1]],
		 [P1[:, 0]*P2[:, 2]], [P1[:, 1]*P2[:, 2]], [P1[:, 2]*P2[:, 2]]))

	A = A.T
	# compute linear least square solution

	U, S, V = np.linalg.svd(A)
	F = (V.T[:, -1]).reshape((3, 3))

	# constrain F
	# make rank 2 by zeroing out last singular value
	U, S, V = np.linalg.svd(F)

	S[2] = 0.0
	F = U.dot(np.diag(S)).dot(V)
	F = H2.T.dot(F).dot(H1)

	return F


def normalise(points):
	# print(points.shape)
	points = np.hstack((points, np.ones((points.shape[0], 1))))
	mean = np.mean(points, axis=0)
	sd = np.std(points, axis=0)
	d = (sd[0]+sd[1])/points.shape[0]

	H = np.array([[1 / d, 0, -mean[0] / d], [0, 1 / d, -mean[1] / d], [0, 0, 1]], np.float32)

	points_norm = np.matmul(points, H.T)

	return points_norm, H


def LinearTriangle(K, C, R, x1, x2):
	x1 = np.hstack((x1, np.ones((x1.shape[0], 1))))
	x2 = np.hstack((x2, np.ones((x2.shape[0], 1))))
	n = x1.shape[0]
	P2 = np.zeros((3, 4))
	H = np.zeros((4, 4))
	Xset = np.zeros((4, 4, n))
	XsetNew = np.zeros((4, 4, n))

	x = x1.T
	y = x2.T

	x1 = np.linalg.inv(K) @ x
	x2 = np.linalg.inv(K) @ y

	for j in range(4):

		rot = R[j]
		t = C[j]
		P1 = np.identity(3)
		P1 = np.insert(P1, 3, 0, axis=1)
		P2[0, :3] = rot[0, 0]
		P2[1, :3] = rot[0, 1]
		P2[2, :3] = rot[0, 2]
		P2[:, 3] = -t
		H[0:3, 0:4] = P2
		H[3, 3] = 1

		for i in range(n):
			A = np.array(
				[x1[0, i]*P1[2][:] - P1[0][:], x1[1, i]*P1[2][:] - P1[1][:], x2[0, i]*P2[2][:] - P2[0][:],
				 x2[1, i]*P2[2][:] - P2[1][:]])

			U, S, V = np.linalg.svd(A)
			X = V.T[:, -1]

			Xset[j, :, i] = np.true_divide(X, X[3])
			XsetNew[j, :, i] = np.dot(H, Xset[j, :, i])
	return Xset, XsetNew


def cheirality(Cset, Rset, Xset, XsetNew):
	L = [0, 0, 0, 0]

	for i in range(4):
		L[i] = np.sum(XsetNew[i, 2, :] > 0) + np.sum(Xset[i, 2, :] > 0)

	if L[0] == 0 and L[1] == 0 and L[2] == 0 and L[3] == 0:
		R = np.identity(3)
		t = np.zeros((3, 1))

	else:
		index = L.index(max(L))

		R = Rset[index]
		R = R.reshape(3, 3)

		t = Cset[index]
		t = t.T

		if t[2] < 0:
			# t[2] = -t[2]
			t = -t

	# NoiseReduction
	if np.absolute(R[0][2]) < 0.001:
		R[0][2] = 0

	if np.absolute(R[2][0]) < 0.001:
		R[2][0] = 0

	if np.absolute(t[0]) < 0.01 or (R[0][0]) > 0.99:
		t = [[0], [0], t[2]]

	return R, t


def trajectory_drift(location,location_cv,total):
	x_drift = np.absolute(location[0] ** 2 - location_cv[0] ** 2)
	y_drift = np.absolute(location[1] ** 2 - location_cv[1] ** 2)

	drift = np.sum((x_drift + y_drift) ** 0.5)
	total = total + drift
	return drift, total


def main():
	R_t = np.identity(3)
	T_t = np.zeros((3, 1))

	# declaring the variables for in-built function calc
	R_t_cv = np.identity(3)
	T_t_cv = np.array([[0], [0], [0]])
	total = 0

	# defining the camera calibration matrix
	K = np.array([[964.828979, 0., 643.788025], [0., 964.828979, 484.40799], [0., 0., 1.]])

	for i in range(19, 3872):

		print("image number --> ", i)
		# reading the consecutive img
		img1 = cv2.imread('Oxford_dataset\data\Final_Image{}.png'.format(i))  # queryImage
		img2 = cv2.imread('Oxford_dataset\data\Final_Image{}.png'.format(i + 1))  # trainImage

		# Initiate SIFT detector
		orb = cv2.ORB_create()

		# find the key-points and descriptors with SIFT
		kp1, des1 = orb.detectAndCompute(img1, None)
		kp2, des2 = orb.detectAndCompute(img2, None)

		# create BFMatcher object
		bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		# Match descriptors.
		matches = bf.match(des1, des2)


		# Sort them in the order of their distance.
		matches = sorted(matches, key=lambda x: x.distance)

		# Draw first 10 matches
		# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:, 100], None, flags=2)

		good = []
		for match in matches:
			good.append(match)

		# obtaining the correspondences between two images
		pts1 = np.float32([kp1[match.queryIdx].pt for match in good])
		pts2 = np.float32([kp2[match.trainIdx].pt for match in good])

		# obtaining the fundamental matrix

		x1, x2 = RANSAC_fundamental(pts1, pts2)

		# our F, E
		F = get_fundamental_matrix(x1, x2)
		E = np.dot(K.T, np.dot(F, K))

		# E from opencv
		E_opencv, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.FM_RANSAC)
		points, R_cv, t_cv, mask = cv2.recoverPose(E_opencv, pts1, pts2, K)

		# getting essential matrix
		# getting the 4 R and C pairs
		C, R = compute_P_from_essential(E)

		# linear triangulation
		Xset, XsetNew = LinearTriangle(K, C, R, x1,x2)
		R,t = cheirality(C,R,Xset,XsetNew)

		# computing translation and rotation
		T_t = T_t + np.dot(R_t, t)
		R_t = np.dot(R_t, R)

		# computing translation and rotation obtained from opencv
		T_t_cv = T_t_cv + np.matmul(R_t_cv, t_cv)
		R_t_cv = np.matmul(R_t_cv, R_cv)

		# drift calculation
		# uncomment to see drift values
		# drift, total = trajectory_drift([T_t[0], T_t[2]], [T_t_cv[0], T_t_cv[2]], total)

		# plotting the center of the camera

		# plotting our values
		plt.plot(-T_t[0], T_t[2], 'ro')

		# plotting values obtained from cv2
		# uncomment to see
		# plt.plot(-T_t_cv[0], -T_t_cv[2], 'bo')

		plt.xlim(-1100, 1100)
		plt.pause(0.000001)

	plt.show()


main()