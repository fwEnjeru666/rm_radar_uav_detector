# rm_radar_lidar_detector Architecture

## Runtime Flow

1. `LidarDetector` receives `PointCloud2` messages and owns ROS/nodelet lifecycle.
2. `CloudProcessor` preprocesses point clouds: ROI crop, voxel downsample, plane removal, outlier removal, and local AABB accumulation.
3. `LidarDetector::clusterTimerCallback` builds candidate points in either dynamic or direct mode.
4. `DBSCAN` clusters candidate points.
5. `ClusterFilter` validates clusters, computes shape scores, and `CandidateSelector` chooses the best candidate.
6. `SingleTargetTracker` gates, smooths, predicts, and maintains the single target state.
7. `recognition.cpp` contains target front-view support algorithms, including `FrontViewProjector`, `OnlineTemplateRecognition`, and `TargetCloudExtractor`.
8. `Visualizer` creates clouds, markers, track markers, and TF output.

## Module Layout

- `lidar_detector.*`: ROS/nodelet integration layer. It owns lifecycle, parameter loading, callbacks, candidate generation, high-level orchestration, and ROS publish calls.
- `cloud_processor.*`: point-cloud preprocessing and local cloud accumulation.
- `dbscan.*`: reusable DBSCAN implementation over `ClusterPoint`.
- `cluster_filter.*`: cluster geometry validation, shape scoring, debug info, and candidate selection policy.
- `track.*`: single-target association, lock state, prediction, and smoothing.
- `recognition.*`: target-cloud extraction plus recognition-facing logic, including front-view projection, OpenCV front-view UI/click selection, patch matching, and online template recognition.
- `visualizer.*`: RViz-facing marker, cloud, trajectory, text, and TF visualization helpers.

## Dependency Direction

`LidarDetector` is the integration layer. Algorithm modules should not depend on ROS publishers/subscribers. UI/recognition logic that analyzes the target shape or front-view image belongs in `recognition.*`; RViz-only rendering belongs in `visualizer.*`. Prefer adding new detection logic to focused modules such as `ClusterFilter`, `SingleTargetTracker`, `CloudProcessor`, or `recognition.*` instead of extending the core orchestration in `lidar_detector.cpp`.
