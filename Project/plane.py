import cv2
import numpy as np

from utils import ray_plane_intersection


class Plane:
    def __init__(self, name):
        self.name = name
        self.origin = None
        self.normal = None

    def fit_from_points(self, points_3d):
        """
        >>  fits an infinite plane to a set of 3D points using least-squares
            (SVD-based plane fitting)
            updates plane origin (centroid) and unit normal
        """
        points_3d = np.asarray(points_3d, dtype=np.float64)
        if len(points_3d) < 3:
            return False

        centroid = points_3d.mean(axis=0)
        X = points_3d - centroid
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        normal = vt[-1]
        nrm = np.linalg.norm(normal)
        if nrm < 1e-12:
            return False

        self.origin = centroid
        self.normal = normal / nrm
        return True


class MarkerPlane(Plane):
    def __init__(self, name, rectangle, width=23, height=13):
        super().__init__(name)
        self.rectangle = np.array(rectangle, dtype=np.float64)
        self.width = float(width)
        self.height = float(height)

        self.R = None
        self.T = None

    @property
    def poly2d_int(self):
        return self.rectangle.astype(np.int32).reshape((-1, 1, 2))

    def detect(self, K):
        """
        >>  estimates the pose of the planar marker using solvePnP
            sets plane origin, normal, rotation matrix, and translation vector
        """
        obj_pts = np.array(
            [
                [0, 0, 0],
                [self.width, 0, 0],
                [self.width, self.height, 0],
                [0, self.height, 0],
            ],
            dtype=np.float64,
        )

        img_pts = self.rectangle.reshape(-1, 1, 2).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise RuntimeError(f"solvePnP failed for {self.name}")

        self.R, _ = cv2.Rodrigues(rvec)
        self.T = tvec.reshape(3)

        # plane normal in camera coordinates
        n = self.R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # enforce consistent normal direction (toward camera)
        if n[2] > 0:
            n *= -1.0

        self.normal = n / np.linalg.norm(n)
        self.origin = self.T.copy()

    def draw(self, display, color=(0, 255, 0), opacity=0.3):
        """
        >>  draws plane
        """
        overlay = display.copy()
        cv2.fillPoly(overlay, [self.poly2d_int], color)
        cv2.putText(
            overlay,
            self.name,
            tuple(self.rectangle[0].astype(int) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            1,
        )
        cv2.addWeighted(overlay, opacity, display, 1 - opacity, 0, display)

    def distance(self):
        """
        >>  distance from camera origin
        """
        return abs(np.dot(self.origin, self.normal))


class LaserPlane(Plane):
    def __init__(self, hsv_min=(150, 20, 78), hsv_max=(200, 255, 255)):
        super().__init__("laser")
        self.hsv_min = hsv_min
        self.hsv_max = hsv_max

        # background subtractor for isolating laser pixels
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=16, detectShadows=False
        )

    def detect_laser_pixels(self, frame):
        """
        >>  takes a frame and detect laser points by removing the background,
            and looking for the points with color in a specific range of hsv
            color space
            returns a list of 2d laser points
        """
        fg_mask = self.backSub.apply(frame, learningRate=0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array((0, self.hsv_min[1], self.hsv_min[2]))
        upper_red_1 = np.array((10, self.hsv_max[1], self.hsv_max[2]))
        lower_red_2 = np.array((170, self.hsv_min[1], self.hsv_min[2]))
        upper_red_2 = np.array((179, self.hsv_max[1], self.hsv_max[2]))

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        color_mask = cv2.bitwise_or(mask1, mask2)

        combined = cv2.bitwise_and(color_mask, fg_mask)
        pts = cv2.findNonZero(combined)
        if pts is None:
            return None
        return pts.reshape(-1, 2)

    def points_inside_marker_planes(self, laser_pts_uv, marker_planes):
        """
        >>  groups detected laser pixels based on which marker plane polygon
            they fall inside in image space
            returns dict: marker_name -> list of (u, v)
        """
        result = {p.name: [] for p in marker_planes}
        if laser_pts_uv is None:
            return result

        for u, v in laser_pts_uv:
            for p in marker_planes:
                if cv2.pointPolygonTest(p.poly2d_int, (float(u), float(v)), False) >= 0:
                    result[p.name].append((float(u), float(v)))
                    break
        return result

    def estimate_from_marker_planes(self, camera, laser_pts_uv, marker_planes):
        """
        >>  estimates the laser plane by intersecting camera rays with
            reference marker planes and fitting a plane to the resulting 3D points
        """
        plane_pts_map = self.points_inside_marker_planes(laser_pts_uv, marker_planes)

        pts3d = []
        plane_by_name = {p.name: p for p in marker_planes}

        for name, uv_list in plane_pts_map.items():
            ref_plane = plane_by_name[name]
            for uv in uv_list:
                ray = camera.ray_cast(uv)
                p3 = ray_plane_intersection(ref_plane.origin, ref_plane.normal, ray)
                if p3 is not None:
                    pts3d.append(p3)

        return self.fit_from_points(pts3d)

    def draw(self, display, pts, color=(0, 0, 255), size=1):
        """
        >>  draws detected laser pixels on the image for visualization
        """
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.ndim == 2 and pts.shape[1] == 2:
            for u, v in pts:
                cv2.circle(display, (int(u), int(v)), size, color, -1)
