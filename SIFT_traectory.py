import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

class VisualOdometrySIFT():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir, "poses.txt"))
        self.images = self._load_images(os.path.join(data_dir, "image_l"))
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))  # Make it a 4x4 matrix
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):

        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    def get_matches(self, i):

        kp1, des1 = self.sift.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.sift.detectAndCompute(self.images[i], None)

        if des1 is None or des2 is None:
            raise ValueError(f"Frame {i}: Не удалось вычислить дескрипторы!")

        # Используем FLANN вместо BFMatcher
        matches = self.bf.knnMatch(des1, des2, k=2)

        # Применяем тест Lowe's Ratio
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        return q1, q2, kp1, kp2  # Возвращаем ключевые точки тоже!qq

    def get_pose(self, q1, q2):

        h,w = self.images[0].shape
        q1_norm = q1 / [w,h]
        q2_norm = q2 / [w, h]

        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=0.5)

        # Decompose the Essential matrix into rotation (R) and translation (t)
        retval, R, t, _ = cv2.recoverPose(E, q1, q2, self.K)
        t = t/np.linalg.norm(t)
        if not retval:
            raise ValueError("Pose recovery failed.")

        return R, t

    def compute_trajectory(self):

        trajectory = []
        gt_trajectory = []  # For storing ground truth x, z coordinates
        errors = []  # To store the errors between the ground truth and estimated trajectory

        # Initialize with the first pose
        cur_pose = np.eye(4)
        trajectory.append((cur_pose[0, 3], cur_pose[2, 3]))  # Add (x, y) position
        gt_trajectory.append((self.gt_poses[0][0, 3], self.gt_poses[0][2, 3]))  # True (x, z)

        # Iterate over the images
        for i in tqdm(range(1, len(self.images))):
            q1, q2, kp1, kp2 = self.get_matches(i)
            R, t = self.get_pose(q1, q2)

            # Update the pose (only consider 2D translation)
            cur_pose = np.dot(cur_pose, np.vstack((np.hstack((R, t)), [0, 0, 0, 1])))
            trajectory.append((cur_pose[0, 3], -cur_pose[2, 3]))  # Add (x, y) position

            # Append true x, z values
            gt_trajectory.append((self.gt_poses[i][0, 3], self.gt_poses[i][2, 3]))

            # Compute error (Euclidean distance between predicted and ground truth positions)
            estimated_position = np.array([cur_pose[0, 3], -cur_pose[2, 3]])
            gt_position = np.array([self.gt_poses[i][0, 3], self.gt_poses[i][2, 3]])
            error = np.linalg.norm(estimated_position - gt_position)  # Euclidean distance
            errors.append(error)

            # Debugging: Print errors for each frame
            print(f"Frame {i}: Error = {error:.2f} meters")

        return trajectory, gt_trajectory, errors, kp1, kp2

    def plot_trajectory(self, trajectory, gt_trajectory, errors):

        trajectory = np.array(trajectory)
        gt_trajectory = np.array(gt_trajectory)
        errors = np.array(errors)

        # Plotting the trajectories
        plt.figure(figsize=(12, 6))

        # First plot: Trajectory
        plt.subplot(1, 2, 1)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="Estimated Path")
        plt.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], label="Ground Truth", linestyle="--")
        plt.title("Camera Trajectory (2D)")
        plt.xlabel("X")
        plt.ylabel("Z")
        plt.legend()
        plt.grid(True)

        # Second plot: Error
        plt.subplot(1, 2, 2)
        plt.plot(errors, label="Error (Euclidean Distance)")
        plt.title("Error between Estimated and Ground Truth")
        plt.xlabel("Frame Index")
        plt.ylabel("Error (meters)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_keypoints(self, frame_idx, kp1, kp2):

        img = self.images[frame_idx]
        img_with_keypoints = cv2.drawKeypoints(img, kp2, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

        plt.figure(figsize=(10, 10))
        plt.imshow(img_with_keypoints, cmap='gray')
        plt.title(f"Keypoints in Frame {frame_idx}")
        plt.show()

def main():
    data_dir = "dataset2"  # Directory containing the dataset (images, calib.txt, poses.txt)
    vo = VisualOdometrySIFT(data_dir)

    # Compute trajectory using SIFT
    trajectory, gt_trajectory, errors, kp1, kp2 = vo.compute_trajectory()

    # Plot the trajectory and errors
    vo.plot_trajectory(trajectory, gt_trajectory, errors)

    # Find the first frame with error above threshold
    threshold = 1.0  # meters
    for i, error in enumerate(errors):
        if error > threshold:
            print(f"Error exceeds threshold at frame {i} (Error: {error:.2f} meters)")

            for i in range(1, min(5, len(trajectory))):
                print(f"Frame {i}: Estimated {trajectory[i]}, Ground Truth {gt_trajectory[i]}")

            vo.plot_keypoints(i, kp1, kp2)
            break

if __name__ == "__main__":
    main()