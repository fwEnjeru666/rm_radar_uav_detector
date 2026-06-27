#include <yolo_detector/models/yolo26.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <cerrno>
#include <sys/stat.h>

#include <opencv2/imgproc.hpp>
#include <ros/ros.h>

namespace rm_radarplugin
{
namespace
{
bool pathExists(const std::string& path)
{
  struct stat buffer;
  return !path.empty() && stat(path.c_str(), &buffer) == 0;
}

bool isDirectory(const std::string& path)
{
  struct stat buffer;
  return !path.empty() && stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode);
}

bool endsWith(const std::string& value, const std::string& suffix)
{
  return value.size() >= suffix.size() &&
         std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

std::string dirname(const std::string& path)
{
  const auto pos = path.find_last_of("/");
  if (pos == std::string::npos)
  {
    return ".";
  }
  if (pos == 0)
  {
    return "/";
  }
  return path.substr(0, pos);
}

bool ensureDirectory(const std::string& path)
{
  if (path.empty())
  {
    return false;
  }
  if (isDirectory(path))
  {
    return true;
  }

  const std::string parent = dirname(path);
  if (!parent.empty() && parent != path && !isDirectory(parent) && !ensureDirectory(parent))
  {
    return false;
  }

  return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

std::string trim(std::string value)
{
  const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
    return std::isspace(ch) || ch == '\'' || ch == '"';
  });
  const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
    return std::isspace(ch) || ch == '\'' || ch == '"';
  }).base();
  if (begin >= end)
  {
    return {};
  }
  return std::string(begin, end);
}

bool parseScalarLine(const std::string& line, std::string& key, std::string& value)
{
  const auto comment_pos = line.find('#');
  const std::string no_comment = line.substr(0, comment_pos);
  const auto colon_pos = no_comment.find(':');
  if (colon_pos == std::string::npos)
  {
    return false;
  }

  key = trim(no_comment.substr(0, colon_pos));
  value = trim(no_comment.substr(colon_pos + 1));
  return !key.empty() && !value.empty() && value.front() != '[';
}

int parsePositiveInt(const std::string& value, int fallback)
{
  try
  {
    const int parsed = std::stoi(value);
    return parsed > 0 ? parsed : fallback;
  }
  catch (...)
  {
    return fallback;
  }
}

float parseFloat(const std::string& value, float fallback)
{
  try
  {
    return std::stof(value);
  }
  catch (...)
  {
    return fallback;
  }
}

bool parseBool(const std::string& value, bool fallback)
{
  std::string lowered = value;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (lowered == "true" || lowered == "1" || lowered == "yes")
  {
    return true;
  }
  if (lowered == "false" || lowered == "0" || lowered == "no")
  {
    return false;
  }
  return fallback;
}

cv::Rect2f rectFromXyxy(float x1, float y1, float x2, float y2)
{
  const float left = std::min(x1, x2);
  const float top = std::min(y1, y2);
  const float right = std::max(x1, x2);
  const float bottom = std::max(y1, y2);
  return cv::Rect2f(left, top, right - left, bottom - top);
}
}  // namespace

YOLO26::YOLO26(const std::string& config_path, bool debug)
  : debug_(debug), model_path_(config_path)
{
  if (config_path.empty())
  {
    ROS_WARN("[YOLO26] Empty model/config path.");
    return;
  }

  std::string base_dir = dirname(config_path);
  if (isDirectory(config_path))
  {
    base_dir = config_path;
    const std::string yolo26n_xml = config_path + "/yolo26n.xml";
    const std::string model_xml = config_path + "/model.xml";
    if (pathExists(yolo26n_xml))
    {
      model_path_ = yolo26n_xml;
    }
    else if (pathExists(model_xml))
    {
      model_path_ = model_xml;
    }
  }
  else if (!endsWith(config_path, ".xml") && pathExists(config_path))
  {
    const auto resolve_model_path = [&](const std::string& path) {
      if (!path.empty() && path.front() != '/')
      {
        return base_dir + "/" + path;
      }
      return path;
    };

    const std::map<std::string, std::function<void(const std::string&)>> config_handlers = {
      {"model", [&](const std::string& value) { model_path_ = resolve_model_path(value); }},
      {"model_path", [&](const std::string& value) { model_path_ = resolve_model_path(value); }},
      {"yolo_model_path", [&](const std::string& value) { model_path_ = resolve_model_path(value); }},
      {"yolo26_model_path", [&](const std::string& value) { model_path_ = resolve_model_path(value); }},
      {"device", [&](const std::string& value) { device_ = value; }},
      {"cache_dir", [&](const std::string& value) { cache_dir_ = resolve_model_path(value); }},
      {"model_cache_dir", [&](const std::string& value) { cache_dir_ = resolve_model_path(value); }},
      {"openvino_cache_dir", [&](const std::string& value) { cache_dir_ = resolve_model_path(value); }},
      {"input_width", [&](const std::string& value) { input_width_ = parsePositiveInt(value, input_width_); }},
      {"input_height", [&](const std::string& value) { input_height_ = parsePositiveInt(value, input_height_); }},
      {"score_threshold", [&](const std::string& value) { score_threshold_ = parseFloat(value, score_threshold_); }},
      {"confidence_threshold", [&](const std::string& value) { score_threshold_ = parseFloat(value, score_threshold_); }},
      {"conf_threshold", [&](const std::string& value) { score_threshold_ = parseFloat(value, score_threshold_); }},
      {"nms_threshold", [&](const std::string& value) { nms_threshold_ = parseFloat(value, nms_threshold_); }},
      {"class_num", [&](const std::string& value) { class_num_ = parsePositiveInt(value, class_num_); }},
      {"num_classes", [&](const std::string& value) { class_num_ = parsePositiveInt(value, class_num_); }},
      {"use_roi", [&](const std::string& value) { use_roi_ = parseBool(value, use_roi_); }},
    };

    std::ifstream config(config_path);
    std::string line;
    while (std::getline(config, line))
    {
      std::string key;
      std::string value;
      if (!parseScalarLine(line, key, value))
      {
        continue;
      }

      const auto handler = config_handlers.find(key);
      if (handler != config_handlers.end())
      {
        handler->second(value);
      }
    }
  }

  if (isDirectory(model_path_))
  {
    const std::string yolo26n_xml = model_path_ + "/yolo26n.xml";
    const std::string model_xml = model_path_ + "/model.xml";
    if (pathExists(yolo26n_xml))
    {
      model_path_ = yolo26n_xml;
    }
    else if (pathExists(model_xml))
    {
      model_path_ = model_xml;
    }
  }

#if HAS_OPENVINO
  try
  {
    if (cache_dir_.empty())
    {
      cache_dir_ = dirname(model_path_) + "/.openvino_cache";
    }
    if (ensureDirectory(cache_dir_))
    {
      core_.set_property(ov::cache_dir(cache_dir_));
      ROS_INFO("[YOLO26] OpenVINO cache enabled: %s", cache_dir_.c_str());
    }
    else
    {
      ROS_WARN("[YOLO26] Failed to create OpenVINO cache dir: %s", cache_dir_.c_str());
      cache_dir_.clear();
    }

    auto model = core_.read_model(model_path_);
    ov::preprocess::PrePostProcessor ppp(model);
    auto& input = ppp.input();

    input.tensor()
      .set_element_type(ov::element::u8)
      .set_shape({1, static_cast<int64_t>(input_height_), static_cast<int64_t>(input_width_), 3})
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::BGR);

    input.model().set_layout("NCHW");
    input.preprocess()
      .convert_element_type(ov::element::f32)
      .convert_color(ov::preprocess::ColorFormat::RGB)
      .scale(255.0);

    model = ppp.build();
    compiled_model_ = core_.compile_model(
      model,
      device_,
      ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    for (auto& slot : infer_slots_)
    {
      slot.request = compiled_model_.create_infer_request();
      slot.input_buffer.create(input_height_, input_width_, CV_8UC3);
      slot.pending = false;
    }
    model_loaded_ = true;

    ROS_INFO("[YOLO26] Loaded OpenVINO model: %s on %s with THROUGHPUT hint",
             model_path_.c_str(), device_.c_str());
  }
  catch (const std::exception& e)
  {
    ROS_ERROR("[YOLO26] Failed to load OpenVINO model '%s': %s", model_path_.c_str(), e.what());
  }
#else
  ROS_WARN("[YOLO26] Built without OpenVINO support. Detection will return empty results.");
#endif
}

bool YOLO26::detect(const cv::Mat& image, int frame_count, YoloDetectionOutput& detection)
{
  (void)frame_count;
  if (image.empty())
  {
    return false;
  }

#if HAS_OPENVINO
  if (!model_loaded_)
  {
    ROS_WARN_THROTTLE(2, "[YOLO26] OpenVINO compiled model is not available.");
    return false;
  }

  cv::Point2f offset(0.0F, 0.0F);
  cv::Mat bgr_img;
  if (use_roi_)
  {
    roi_ &= cv::Rect(0, 0, image.cols, image.rows);
    if (roi_.width <= 1 || roi_.height <= 1)
    {
      ROS_WARN_THROTTLE(2, "[YOLO26] Invalid ROI, falling back to full image.");
      use_roi_ = false;
      bgr_img = image;
    }
    else
    {
      bgr_img = image(roi_);
      offset = cv::Point2f(static_cast<float>(roi_.x), static_cast<float>(roi_.y));
    }
  }
  else
  {
    bgr_img = image;
  }

  const double x_scale = static_cast<double>(input_width_) / bgr_img.cols;
  const double y_scale = static_cast<double>(input_height_) / bgr_img.rows;
  const double scale = std::min(x_scale, y_scale);
  const int resized_width = std::max(1, static_cast<int>(std::round(bgr_img.cols * scale)));
  const int resized_height = std::max(1, static_cast<int>(std::round(bgr_img.rows * scale)));

  const std::size_t submit_slot_index = next_submit_slot_;
  const std::size_t result_slot_index =
    (submit_slot_index + infer_slots_.size() - 1) % infer_slots_.size();
  auto& submit_slot = infer_slots_[submit_slot_index];

  submit_slot.input_buffer.setTo(cv::Scalar(0, 0, 0));
  cv::resize(bgr_img, submit_slot.input_buffer(cv::Rect(0, 0, resized_width, resized_height)),
             cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_NEAREST);
  submit_slot.scale = scale;
  submit_slot.image_size = image.size();
  submit_slot.offset = offset;

  ov::Tensor input_tensor(
    ov::element::u8,
    {1, static_cast<size_t>(input_height_), static_cast<size_t>(input_width_), 3},
    submit_slot.input_buffer.data);

  submit_slot.request.set_input_tensor(input_tensor);
  submit_slot.request.start_async();
  submit_slot.pending = true;
  next_submit_slot_ = (submit_slot_index + 1) % infer_slots_.size();

  auto& result_slot = infer_slots_[result_slot_index];
  if (!result_slot.pending)
  {
    return false;
  }

  result_slot.request.wait();
  auto output_tensor = result_slot.request.get_output_tensor();
  const auto output_shape = output_tensor.get_shape();
  result_slot.pending = false;
  if (output_shape.empty())
  {
    return false;
  }

  const int rows = output_shape.size() >= 2 ? static_cast<int>(output_shape[output_shape.size() - 2]) : 1;
  const int cols = static_cast<int>(output_shape.back());
  cv::Mat output(rows, cols, CV_32F, output_tensor.data());

  return parseYolo26Output(result_slot.scale, output, result_slot.image_size, result_slot.offset, detection);
#else
  (void)detection;
  return false;
#endif
}

bool YOLO26::postprocess(
    double scale, cv::Mat& output, const cv::Mat& bgr_image, int frame_count, YoloDetectionOutput& detection)
{
  (void)frame_count;
  return parseYolo26Output(scale, output, bgr_image.size(), cv::Point2f(0.0F, 0.0F), detection);
}

void YOLO26::sortArmorPoints(std::array<cv::Point2f, 4>& points)
{
  std::sort(points.begin(), points.end(), [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
    return lhs.y < rhs.y;
  });

  if (points[0].x > points[1].x)
  {
    std::swap(points[0], points[1]);
  }
  if (points[2].x < points[3].x)
  {
    std::swap(points[2], points[3]);
  }
}

cv::Rect2f YOLO26::clipBox(const cv::Rect2f& box, const cv::Size& image_size)
{
  const float x = std::max(0.0F, std::min(box.x, static_cast<float>(image_size.width)));
  const float y = std::max(0.0F, std::min(box.y, static_cast<float>(image_size.height)));
  const float right = std::max(x, std::min(box.x + box.width, static_cast<float>(image_size.width)));
  const float bottom = std::max(y, std::min(box.y + box.height, static_cast<float>(image_size.height)));
  return cv::Rect2f(x, y, right - x, bottom - y);
}

bool YOLO26::parseYolo26Output(
    double scale, cv::Mat& output, const cv::Size& image_size, const cv::Point2f& offset,
    YoloDetectionOutput& detection) const
{
  if (output.empty() || scale <= std::numeric_limits<double>::epsilon())
  {
    return false;
  }

  if (output.rows < output.cols && output.rows <= 128 && output.cols > 128)
  {
    cv::transpose(output, output);
  }

  for (int row = 0; row < output.rows; ++row)
  {
    const float* data = output.ptr<float>(row);
    const int cols = output.cols;
    if (cols != 6)
    {
      continue;
    }

    const cv::Rect2f box_xyxy = rectFromXyxy(data[0], data[1], data[2], data[3]);
    const float confidence = data[4];
    const int class_id = (class_num_ == 1) ? -1 : static_cast<int>(std::round(data[5]));

    if (confidence < score_threshold_)
    {
      continue;
    }

    cv::Rect2f box = box_xyxy;
    box.x = static_cast<float>(box.x / scale + offset.x);
    box.y = static_cast<float>(box.y / scale + offset.y);
    box.width = static_cast<float>(box.width / scale);
    box.height = static_cast<float>(box.height / scale);
    box = clipBox(box, image_size);
    if (box.width <= 1.0F || box.height <= 1.0F)
    {
      continue;
    }

    detection.class_id = class_id;
    detection.confidence = confidence;
    detection.bbox = box;
    detection.armor_points = {};
    detection.center = cv::Point2f(
      detection.bbox.x + detection.bbox.width * 0.5F,
      detection.bbox.y + detection.bbox.height * 0.5F);
    if (debug_)
    {
      ROS_INFO_THROTTLE(2, "[YOLO26] Parsed 1 detection.");
    }
    return true;
  }

  if (debug_)
  {
    ROS_INFO_THROTTLE(2, "[YOLO26] Parsed 0 detections.");
  }
  return false;
}

}  // namespace rm_radarplugin
