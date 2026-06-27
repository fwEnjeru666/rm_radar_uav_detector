#include <yolo_detector/yolo.h>

#include <yolo_detector/models/yolo26.h>

namespace rm_radarplugin
{
YoloCore::YoloCore(const std::string& config_path, bool debug)
{
  yolo_ = std::make_unique<YOLO26>(config_path, debug);
}

bool YoloCore::detect(const cv::Mat& img, YoloDetectionOutput& detection, int frame_count)
{
  return yolo_->detect(img, frame_count, detection);
}

bool YoloCore::postprocess(
    double scale, cv::Mat& output, const cv::Mat& bgr_img, YoloDetectionOutput& detection, int frame_count)
{
  return yolo_->postprocess(scale, output, bgr_img, frame_count, detection);
}

}  // namespace rm_radarplugin
