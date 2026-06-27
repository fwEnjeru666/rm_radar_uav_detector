#pragma once

#include <array>
#include <memory>
#include <string>
#include <opencv2/core.hpp>

namespace rm_radarplugin
{

struct YoloDetectionOutput
{
  int class_id{-1};
  float confidence{0.0F};
  cv::Rect2f bbox{};
  std::array<cv::Point2f, 4> armor_points{};  // TL, TR, BR, BL
  cv::Point2f center{};
};

class YoloBackend
{
public:
  virtual ~YoloBackend() = default;

  virtual bool detect(const cv::Mat& image, int frame_count, YoloDetectionOutput& detection) = 0;
  virtual bool postprocess(
      double scale, cv::Mat& output, const cv::Mat& bgr_image, int frame_count, YoloDetectionOutput& detection) = 0;
};

class YoloCore
{
public:
  explicit YoloCore(const std::string& config_path, bool debug = false);

  bool detect(const cv::Mat& image, YoloDetectionOutput& detection, int frame_count = 0);
  bool postprocess(
      double scale, cv::Mat& output, const cv::Mat& bgr_image, YoloDetectionOutput& detection, int frame_count = 0);

private:
  std::unique_ptr<YoloBackend> yolo_;
};

class YOLO26;

}  // namespace rm_radarplugin
