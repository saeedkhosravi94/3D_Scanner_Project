import math

import cv2
import numpy as np


def sort_corners(pts):
    """
    >>  uses atan2 to sort contour points by there angles
    """
    pts = pts.reshape(4, 2)
    center = pts.mean(axis=0)

    def angle(p):
        return math.atan2(p[1] - center[1], p[0] - center[0])

    pts = sorted(pts, key=angle)
    return np.array(pts, dtype=np.float32)


def find_markers(frame):
    """
    >>  uses contours and take ones with at least min_area area.
        returns 2 of the biggest rectangles
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    min_area = 1000
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rectangles.append(sort_corners(approx))

    rectangles.sort(key=lambda r: r[:, 1].mean())

    if len(rectangles) < 2:
        raise RuntimeError("Markers not found.")

    return rectangles[0], rectangles[1]


def ray_plane_intersection(plane_origin, plane_normal, ray_dir):
    """
    >>  returns the 3D intersection point in camera coordinates
        if not parallel to plane
    """
    denom = float(np.dot(ray_dir, plane_normal))
    if abs(denom) < 1e-12:
        return None
    d = float(np.dot(plane_origin, plane_normal)) / denom
    return ray_dir * d
