import cv2
import numpy as np
import open3d as o3d

from camera import Camera
from plane import LaserPlane, MarkerPlane
from utils import find_markers, ray_plane_intersection


def main():
    input_path = "./data/videos/cup1.mp4"
    output_path = "output.ply"

    K = np.loadtxt("./data/calibration/K.txt")
    dist = np.loadtxt("./data/calibration/dist.txt")

    camera = Camera(K, dist)
    cap = cv2.VideoCapture(input_path)

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Frame not found.")
    first = cv2.undistort(first, K, dist)

    laser_plane = LaserPlane()
    # take first frame as background
    laser_plane.backSub.apply(first)

    # find markers using contours and area using first frame
    wall_rect, table_rect = find_markers(first)

    wall_plane = MarkerPlane("wall", wall_rect)
    table_plane = MarkerPlane("table", table_rect)
    marker_planes = [wall_plane, table_plane]

    # detect origin and normal of marker plane objects
    wall_plane.detect(K)
    table_plane.detect(K)

    _3d_points = []
    _3d_colors = []

    while True:
        ok, frame = cap.read()
        if not ok or (cv2.waitKey(1) & 0xFF == ord("q")):
            break

        frame = cv2.undistort(frame, K, dist)
        laser_pts = laser_plane.detect_laser_pixels(frame)

        ok = laser_plane.estimate_from_marker_planes(camera, laser_pts, marker_planes)

        if ok and laser_pts is not None:
            for u, v in laser_pts:
                ray = camera.ray_cast((u, v))
                p3 = ray_plane_intersection(laser_plane.origin, laser_plane.normal, ray)

                # if the distance from camera is not normal don't add it.
                if p3 is None or np.linalg.norm(p3) > 70 or np.linalg.norm(p3) < 40:
                    continue

                _3d_points.append(p3)

                bgr = first[int(v), int(u)].astype(np.float64) / 255.0
                _3d_colors.append(bgr[::-1])

        display = frame.copy()
        wall_plane.draw(display, (150, 100, 100))
        table_plane.draw(display, (100, 150, 150))
        laser_plane.draw(display, laser_pts, color=(150, 100, 150), size=1)

        cv2.putText(
            display,
            "Press q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.imshow("3D scanner", display)

    cap.release()
    cv2.destroyAllWindows()

    if not _3d_points:
        raise RuntimeError("No 3D points generated.")

    # save and visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(_3d_points))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(_3d_colors))
    o3d.io.write_point_cloud(output_path, pcd)
    vis = o3d.visualization.Visualizer()
    vis = o3d.visualization.Visualizer()
    center = pcd.get_center()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_lookat(center.tolist())
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.8)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
