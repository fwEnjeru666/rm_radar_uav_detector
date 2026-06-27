#pragma once

#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <common.h>

namespace rm_radarplugin
{
struct PreprocessOptions
{
    bool target_is_red{};
    int preprocess_method{};

    int red_h_min_low{};
    int red_h_max_low{};
    int red_h_min_high{};
    int red_h_max_high{};
    int red_s_min{};
    int red_s_max{};
    int red_v_min{};
    int red_v_max{};

    int blue_h_min{};
    int blue_h_max{};
    int blue_s_min{};
    int blue_s_max{};
    int blue_v_min{};
    int blue_v_max{};

    int binary_thresh{};
    int morph_type{};
    int binary_element{};
    int kernel_shape{};
    int kernel_w{};
    int kernel_h{};
    int kernel_angle_deg{};
    int morph_iterations{1};
};

class PreprocessCore
{
public:
    explicit PreprocessCore(const PreprocessOptions& options);

    void setOptions(const PreprocessOptions& options);
    const PreprocessOptions& options() const { return options_; }

    void process(const cv::UMat& image);

    const cv::UMat& getBinaryImage() const { return binary_image_; }
    const cv::UMat& getMorphologyImage() const { return morpro_image_; }

    static int sanitizeKernelSize(int value);

private:
    cv::Mat buildMorphKernel();
    void hsv2Binary();
    void bgr2Binary();

    PreprocessOptions options_{};
    cv::UMat source_image_{};
    cv::UMat binary_image_{};
    cv::UMat morpro_image_{};
    cv::UMat hsv_image_{};
    cv::UMat blue_channel_{};
    cv::UMat green_channel_{};
    cv::UMat red_channel_{};

    cv::Mat morph_kernel_cache_{};
    bool morph_kernel_dirty_{true};
    int last_kernel_shape_{-1};
    int last_kernel_w_{-1};
    int last_kernel_h_{-1};
    int last_kernel_angle_deg_{-1};
};
}  // namespace rm_radarplugin
