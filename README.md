# Simple pose estimator
A rather simple pose estimator by tracking the horizon using OpenCV. It may have applications in AR and robotics when model, camera intrinsics/extrinsics and odometry information are missing.

![Output from horizon_tracker.py.](result.gif)

A YouTube video from MotoGP rider Casey Stoner was used to demonstrate its performance. 

### Dependencies

[OpenCV](https://opencv.org): I briefly experimented with `solvePnP` and `solvePnPRansac`, i.e., to find the pose from 3D-2D point correspondences. However, it was not straightforward to identify good correspondences for all frames given the many unknowns.
