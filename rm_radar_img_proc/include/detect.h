#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <armor.h>
#include <object_options.h>
#include <tools.h>
#include <common.h>
#include <yolo_detector/yolo.h>

namespace rm_radarplugin
{
struct DetectOptions
{
    int detect_method{DetectMethod::TRADITIONAL};
    std::string model_config_path{};
    bool process_debug{false};
    bool enable_target_filter{true};
    bool save_yolo_samples{false};
    bool yolo_refine_enabled{false};
    bool target_is_red{false};
    int select_bar{};
    double refine_max_brightness{25.0};
    double refine_roi_scale{0.07};
    double refine_search_start{0.4};
    double refine_search_end{0.6};
    ObjectOptions object_options{};
};

class DetectCore
{
    struct ArmorCandidate
    {
        size_t top_idx;
        size_t bottom_idx;
        double score;
    };

public:
    explicit DetectCore(const DetectOptions& options);

    void setOptions(const DetectOptions& options);
    void reset();
    void resetBestArmor();

    void detect(const cv::Mat& raw_image);
    void detect(const cv::UMat& raw_image);
    void detect(const cv::UMat& raw_image, const cv::UMat* morphology_image);

    const std::vector<Bar>& getBars() const { return bars_; }
    const std::vector<Armor>& getDebugArmors() const { return debug_armors_; }
    const Armor* getBestArmor() const { return best_armor_.get(); }
    Armor* getBestArmor() { return best_armor_.get(); }

private:
    void updateGrayImage(const cv::Mat& image);
    void updateGrayImage(const cv::UMat& image);
    bool targetFilter(int class_id) const;
    bool targetFilter(const Armor& armor, const cv::Mat& bgr_img) const;
    bool shouldSaveYoloFailedSample() const;
    bool shouldSaveYoloUncertainSample(float confidence) const;
    void saveYoloSample(const cv::Mat& image, const std::string& reason, float confidence,
                        const cv::Rect* roi_box = nullptr) const;
    void findArmor(const cv::UMat& raw_image, const cv::UMat& morphology_image);
    std::unique_ptr<Armor> buildArmorFromYoloDetection(const YoloDetectionOutput& detection,
                                                       const cv::Mat& bgr_img);
    bool refineArmor(Armor& armor);
    void findBars(const cv::UMat& raw_image, const cv::UMat& morphology_image);
    std::vector<size_t> findBars(const cv::Rect& roi_box);
    bool isValidBar(const Bar& bar);
    bool calcArmorScore(const Bar& top_bar, const Bar& bottom_bar, double& score);
    bool getCorrectedBarEndpoints(const Bar* bar, const cv::Mat& gray_img,
                                  std::array<cv::Point2f, 2>& endpoints);

    DetectOptions detect_options_{};
    ObjectOptions object_options_{};
    tools::Debugger debugger_;
    std::vector<Bar> bars_{};
    std::unique_ptr<Armor> best_armor_{};
    std::vector<Armor> debug_armors_{};
    std::unique_ptr<YoloCore> yolo_core_{};
    cv::Mat gray_image_{};
};
}  // namespace rm_radarplugin
