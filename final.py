import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from calibration import _load_calib

camera_positions = []

sift = cv2.SIFT_create()

bf = cv2.BFMatcher(cv2.NORM_L2)

# Матрица
K = _load_calib('C:/Users/steph/PycharmProjects/SIFT_trajectory/dataset2/calib.txt')


def find_camera_pose(img1, img2):
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    if descriptors_1 is None or descriptors_2 is None:
        return

    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 10:
        return

    '''  Это набор точек на первом изображении, которые были 
    сопоставлены с точками на втором изображении.'''

    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    '''Это набор точек на втором изображении (называемом целью), 
    которые были сопоставлены с точками на первом изображении.'''

    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    ''' Используется для восстановления внешних параметров 
    камеры (вращение R и смещение t) между двумя изображениями.'''

    E, _ = cv2.findEssentialMat(src_pts, dst_pts, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, cameraMatrix=K)

    # Обновляем мировые координаты камеры
    if len(camera_positions) == 0:
        world_R = np.eye(3)
        world_t = np.zeros((3, 1))
    else:
        world_R = camera_positions[-1][0]
        world_t = camera_positions[-1][1]

    new_t = world_t + world_R @ t
    new_R = world_R @ R

    camera_positions.append((new_R, new_t))

# Загружаем все кадры
folder_path = 'C:/Users/steph/PycharmProjects/SIFT_trajectory/dataset2/image_l'
frames = sorted(os.listdir(folder_path))

for i in range(len(frames)-1):
    file_path1 = os.path.join(folder_path, frames[i])
    file_path2 = os.path.join(folder_path, frames[i+1])

    img1 = cv2.imread(file_path1)
    img2 = cv2.imread(file_path2)

    if img1 is None or img2 is None:
        continue

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    find_camera_pose(img1, img2)

import pickle

# Сохраняем массив camera_positions в файл
with open('camera_positions.pkl', 'wb') as f:
    pickle.dump(camera_positions, f)

# Визуализация траектории
trajectory = np.array([pos[1].flatten() for pos in camera_positions])

plt.plot(trajectory[:, 1], trajectory[:, 2], 'bo-')
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Камера траектория")
plt.grid()
plt.show()