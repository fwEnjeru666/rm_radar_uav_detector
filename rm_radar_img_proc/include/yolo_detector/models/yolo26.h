#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <opencv2/core.hpp>

#include <yolo_detector/yolo.h>

#if (defined(USE_OPENVINO) && USE_OPENVINO) || \
    (defined(RM_RADAR_IMG_PROC_USE_OPENVINO) && RM_RADAR_IMG_PROC_USE_OPENVINO)
#include <openvino/openvino.hpp>
#define HAS_OPENVINO 1
#else
#define HAS_OPENVINO 0
#endif

namespace rm_radarplugin
{

class YOLO26 final : public YoloBackend
{
public:
  explicit YOLO26(const std::string& config_path, bool debug = false);

  bool detect(const cv::Mat& image, int frame_count, YoloDetectionOutput& detection) override;
  bool postprocess(
      double scale, cv::Mat& output, const cv::Mat& bgr_image, int frame_count, YoloDetectionOutput& detection) override;

private:
  struct Candidate
  {
    int class_id{-1};
    float confidence{0.0F};
    cv::Rect2f bbox{};
    std::array<cv::Point2f, 4> points{};
  };

  static void sortArmorPoints(std::array<cv::Point2f, 4>& points);
  static cv::Rect2f clipBox(const cv::Rect2f& box, const cv::Size& image_size);

  bool parseYolo26Output(
      double scale, cv::Mat& output, const cv::Size& image_size, const cv::Point2f& offset,
      YoloDetectionOutput& detection) const;

  bool debug_{false};
  std::string model_path_{};
  std::string device_{"AUTO"};
  std::string cache_dir_{};
  int input_width_{640};
  int input_height_{640};
  float score_threshold_{0.4F};
  float nms_threshold_{0.45F};
  int class_num_{1};
  bool use_roi_{false};
  cv::Rect roi_{};
  bool model_loaded_{false};

#if HAS_OPENVINO
  struct InferenceSlot
  {
    ov::InferRequest request{};
    cv::Mat input_buffer{};
    cv::Size image_size{};
    cv::Point2f offset{};
    double scale{1.0};
    bool pending{false};
  };

  ov::Core core_{};
  ov::CompiledModel compiled_model_{};
  std::array<InferenceSlot, 2> infer_slots_{};
  std::size_t next_submit_slot_{0};
#endif
};

}  // namespace rm_radarplugin
