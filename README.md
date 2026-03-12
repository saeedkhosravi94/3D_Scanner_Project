# 3D Scanning via Laser Triangulation and Plane-Based Geometry

This repository features a 3D scanner implemented from scratch using a red line laser and a single calibrated camera. The project focuses on geometric computer vision, specifically plane-to-plane intersection and 3D point cloud reconstruction.

## Key Technical Features

* **Camera Modeling:** Implementation of ray-casting from 2D image coordinates to 3D space using the intrinsic matrix ($K$) and distortion coefficients.
* **Pose Estimation:** Utilizing `solvePnP` with reference markers (wall and table planes) to establish a world coordinate system.
* **Dynamic Laser Plane Fitting:** * Background subtraction (MOG2) and HSV filtering for robust laser line detection.
    * Estimating the 3D laser plane equation in real-time by intersecting camera rays with known reference planes.
    * Singular Value Decomposition (SVD) for optimal plane fitting from 3D points.
* **Point Cloud Generation:** Computing the intersection of camera rays with the estimated laser plane to recover 3D coordinates.
* **Visualization:** Integration with `Open3D` for real-time point cloud rendering and `.ply` export.

## How it Works

1.  **Calibration:** The camera is calibrated to obtain $K$ and distortion parameters.
2.  **Marker Detection:** Two rectangular markers (wall and table) are detected to define the scene's geometry.
3.  **Laser Tracking:** As the laser sweeps over an object, the system identifies laser pixels, projects them onto the marker planes, and fits a 3D "Laser Plane."
4.  **Reconstruction:** Every pixel on the laser line is back-projected to find its intersection with the laser plane, generating a 3D point.


## Technical Stack

* **Python 3.x**
* **OpenCV:** Image processing, PnP, and background subtraction.
* **NumPy:** Linear algebra operations and SVD-based plane fitting.
* **Open3D:** 3D data visualization and processing.

## Results
The project successfully reconstructs objects (like a cup) by moving a handheld laser. The final output is a dense point cloud preserved in `.ply` format.# 3D_Scanner_Project
