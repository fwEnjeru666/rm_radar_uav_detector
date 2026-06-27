#pragma once

#include <mutex>

#include <rm_radar_msgs/DroneDetection.h>

#include "uav_detection_types.h"

namespace rm_radarplugin
{

class CameraDetector
{
public:
  // Updates the cached camera-only detection from image processor output.
  void updateDetection(const rm_radar_msgs::DroneDetection& msg);
  void clear();
  ImageDetection getLatestDetection() const;

  // Builds the vision part of UavDetection and optionally reprojects its 3D point.
  UavDetectorOutput detect(const UavDetectorContext& context) const;

private:
  static rm_radar_msgs::UavDetection makeMessage(const std_msgs::Header& header,
                                                 const ImageDetection& detection);

  UavDetectorOutput projectDetectionToImage(const UavDetectorContext& context) const;

  mutable std::mutex mutex_;
  ImageDetection latest_detection_;
};

}  // namespace rm_radarplugin
