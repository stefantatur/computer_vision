import cv2
import numpy as np
import os

folder_path = '/dataset1/image_l'
frames = sorted(os.listdir(folder_path))

camera_positions = []

def _load_calib(filepath):
    """
    Loads the calibration of the camera
    Parameters
    ----------
    filepath (str): The file path to the camera file

    Returns
    -------
    K (ndarray): Intrinsic parameters
    P (ndarray): Projection matrix
    """
    with open(filepath, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
    return K, P

K, P = _load_calib('/dataset1/calib.txt')
sift = cv2.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

img1 = cv2.imread('C:/Users/steph/PycharmProjects/SIFT_trajectory/img1.jpg')
img2 = cv2.imread('C:/Users/steph/PycharmProjects/SIFT_trajectory/img2.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

def find_camera_pose(img1,img2):
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
    good_matches = [m for m,n in matches if m.distance < 0.75 * n.distance]

    # Выделение координат ключевых точек
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    E, _ = cv2.findEssentialMat(src_pts, dst_pts, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, cameraMatrix=K)

    # Положение камеры в 3D-пространстве
    if len(camera_positions) == 0:
        world_R = np.eye(3)
        world_t = np.zeros((3, 1))
    else:
        world_R = camera_positions[-1][0]
        world_t = camera_positions[-1][1]

    # Обновляем мировые координаты камеры
    new_t = world_t + world_R @ t
    new_R = world_R @ R

    camera_positions.append((new_R, new_t))

    #camera_positions.append(camera_pos)


for i in range(len(frames)-1):
    file_path1 = os.path.join(folder_path, frames[i])
    file_path2 = os.path.join(folder_path, frames[i+1])

    img1 = cv2.imread(file_path1)
    img2 = cv2.imread(file_path2)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    find_camera_pose(img1, img2)

import pickle

with open('camera_positions.pkl', 'wb') as f:
    pickle.dump(camera_positions, f)