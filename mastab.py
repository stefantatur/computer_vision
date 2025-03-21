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
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    def get_matches(self, i):
        """
        This function detects and computes keypoints and descriptors from the i-1'th and i'th image using the class SIFT object
        """
        kp1, des1 = self.sift.detectAndCompute(self.images[i - 1], None)
        kp2, des2 = self.sift.detectAndCompute(self.images[i], None)

        # Match descriptors
        matches = self.bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # Get the matched points
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix between two images
        """
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into rotation (R) and translation (t)
        retval, R, t, mask = cv2.recoverPose(E, q1, q2, self.K)

        return R, t

    def compute_trajectory(self):
        """
        Computes the trajectory of the camera
        """
        trajectory = []

        # Initialize with the first pose
        cur_pose = np.eye(4)
        trajectory.append((cur_pose[0, 3], cur_pose[2, 3]))  # Add (x, y) position

        # Iterate over the images
        for i in tqdm(range(1, len(self.images))):
            q1, q2 = self.get_matches(i)
            R, t = self.get_pose(q1, q2)

            # Update the pose (only consider 2D translation)
            cur_pose = np.dot(cur_pose, np.vstack((np.hstack((R, t)), [0, 0, 0, 1])))
            trajectory.append((cur_pose[0, 3], cur_pose[2, 3]))  # Add (x, y) position

        return trajectory

    def plot_trajectory(self, trajectory):
        """
        Plots the 2D trajectory
        """
        trajectory = np.array(trajectory)

        plt.figure()
        plt.plot(trajectory[:, 0], trajectory[:, 1], label="Estimated Path")
        plt.title("Camera Trajectory (2D)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    data_dir = "dataset2"  # Directory containing the dataset (images, calib.txt, poses.txt)
    vo = VisualOdometrySIFT(data_dir)

    # Compute trajectory using SIFT
    trajectory = vo.compute_trajectory()

    # Plot the trajectory
    vo.plot_trajectory(trajectory)

if __name__ == "__main__":
    main()