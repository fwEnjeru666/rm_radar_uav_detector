#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "uav_detection_types.h"

namespace rm_radarplugin
{

class LidarCameraFusionDetector
{
public:
  // Projects lidar track geometry into the image and builds a bbox/ROI.
  UavDetectorOutput detect(const UavDetectorContext& context) const;

  // Finds the point cloud point whose projection is closest to the camera detection pixel.
  bool findNearestProjectedCloudPoint(const UavDetectorContext& context,
                                      geometry_msgs::Point& point,
                                      std::string& frame_id) const;

  // Stores the camera-selected point offset so it can be tracked when vision is missing.
  void updateMarkedKeypointOffset(const UavDetectorContext& context,
                                    const geometry_msgs::Point& matched_point,
                                    const std::string& frame_id);

  // Reprojects the stored keypoint offset using the current lidar track.
  UavDetectorOutput trackMarkedKeypoint(const UavDetectorContext& context) const;

  // Projects an arbitrary 3D point into the current image.
  bool projectPointToImage(const UavDetectorContext& context,
                           const geometry_msgs::Point& point,
                           const std::string& frame_id,
                           cv::Point2f& pixel) const;

private:
  struct MarkedKeypointOffsetState
  {
    // Offset is kept in the track frame and applied to future track centroids.
    bool valid{false};
    geometry_msgs::Point offset_from_centroid;
    std::string frame_id;
    ros::Time stamp;
  };

  bool getMarkedKeypointOffsetState(MarkedKeypointOffsetState& state) const;
  static geometry_msgs::Point markedKeypoint(const MarkedKeypointOffsetState& state,
                                             const rm_radar_msgs::DroneTrackData& track);
  static rm_radar_msgs::UavDetection makeTrackOutput(const rm_radar_msgs::DroneTrackData& track,
                                                     const ros::Time& stamp);

  static bool buildBbox(const std::vector<cv::Point2f>& projected_points,
                        const cv::Point2f& projected_target_point,
                        bool debug_mode,
                        bool need_roi,
                        float expand_ratio,
                        int image_width,
                        int image_height,
                        Bbox2D::Bbox& bbox);
  mutable std::mutex marked_keypoint_mutex_;
  MarkedKeypointOffsetState marked_keypoint_offset_state_;
};

}  // namespace rm_radarplugin
