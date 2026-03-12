import numpy as np


class Camera:
    def __init__(self, K, dist):
        self.K = np.array(K, dtype=np.float64)
        self.dist = np.array(dist, dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

    def ray_cast(self, uv):
        """
        >> takes a 2d point on frame and cast a ray from origin of the camera to that point.
        """
        u, v = float(uv[0]), float(uv[1])
        uv1 = np.array([u, v, 1.0], dtype=np.float64)
        ray = self.K_inv @ uv1
        ray = ray / np.linalg.norm(ray)
        return ray
