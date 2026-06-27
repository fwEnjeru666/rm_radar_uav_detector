#include <preprocess.h>

namespace rm_radarplugin
{
PreprocessCore::PreprocessCore(const PreprocessOptions& options)
{
    setOptions(options);
}

void PreprocessCore::setOptions(const PreprocessOptions& options)
{
    options_ = options;
    options_.binary_element = sanitizeKernelSize(options_.binary_element);
    if (options_.kernel_w > 0) options_.kernel_w = sanitizeKernelSize(options_.kernel_w);
    if (options_.kernel_h > 0) options_.kernel_h = sanitizeKernelSize(options_.kernel_h);
    options_.kernel_angle_deg = ((options_.kernel_angle_deg % 180) + 180) % 180;
    options_.morph_iterations = std::max(1, options_.morph_iterations);
    morph_kernel_dirty_ = true;
}

int PreprocessCore::sanitizeKernelSize(int value)
{
    const int clamped = std::max(1, value);
    return (clamped % 2 == 0) ? (clamped + 1) : clamped;
}

cv::Mat PreprocessCore::buildMorphKernel()
{
    int w = options_.kernel_w > 0 ? options_.kernel_w : options_.binary_element;
    int h = options_.kernel_h > 0 ? options_.kernel_h : options_.binary_element;
    w = sanitizeKernelSize(w);
    h = sanitizeKernelSize(h);

    const int shape = std::max(static_cast<int>(ELLIPSE), std::min(options_.kernel_shape, static_cast<int>(LINE)));
    const int angle = ((options_.kernel_angle_deg % 180) + 180) % 180;

    const bool need_rebuild = morph_kernel_dirty_ ||
                              morph_kernel_cache_.empty() ||
                              shape != last_kernel_shape_ ||
                              w != last_kernel_w_ ||
                              h != last_kernel_h_ ||
                              angle != last_kernel_angle_deg_;
    if (!need_rebuild)
    {
        return morph_kernel_cache_;
    }

    switch (shape)
    {
        case RECT:
            morph_kernel_cache_ = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(w, h), cv::Point(-1, -1));
            break;
        case LINE:
        {
            cv::Mat kernel = cv::Mat::zeros(h, w, CV_8UC1);
            const cv::Point center((w - 1) / 2, (h - 1) / 2);
            const double rad = angle * CV_PI / 180.0;
            const cv::Point2d dir(std::cos(rad), std::sin(rad));
            const double half_len = static_cast<double>(std::max(w, h));
            cv::Point p1(cvRound(center.x - dir.x * half_len), cvRound(center.y - dir.y * half_len));
            cv::Point p2(cvRound(center.x + dir.x * half_len), cvRound(center.y + dir.y * half_len));
            cv::clipLine(cv::Size(w, h), p1, p2);
            cv::line(kernel, p1, p2, cv::Scalar(255), 1, cv::LINE_8);
            morph_kernel_cache_ = kernel;
            break;
        }
        case ELLIPSE:
        default:
            morph_kernel_cache_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(w, h), cv::Point(-1, -1));
            break;
    }

    last_kernel_shape_ = shape;
    last_kernel_w_ = w;
    last_kernel_h_ = h;
    last_kernel_angle_deg_ = angle;
    morph_kernel_dirty_ = false;
    return morph_kernel_cache_;
}

void PreprocessCore::hsv2Binary()
{
    cv::cvtColor(source_image_, hsv_image_, cv::COLOR_BGR2HSV);
    if (options_.target_is_red == 1)
    {
        cv::UMat h_binary_low, h_binary_high;
        inRange(hsv_image_, cv::Scalar(options_.red_h_min_low, options_.red_s_min, options_.red_v_min),
                cv::Scalar(options_.red_h_max_low, options_.red_s_max, options_.red_v_max), h_binary_low);
        inRange(hsv_image_, cv::Scalar(options_.red_h_min_high, options_.red_s_min, options_.red_v_min),
                cv::Scalar(options_.red_h_max_high, options_.red_s_max, options_.red_v_max), h_binary_high);
        bitwise_or(h_binary_low, h_binary_high, binary_image_);
    }
    else
    {
        inRange(hsv_image_, cv::Scalar(options_.blue_h_min, options_.blue_s_min, options_.blue_v_min),
                cv::Scalar(options_.blue_h_max, options_.blue_s_max, options_.blue_v_max), binary_image_);
    }
}

void PreprocessCore::bgr2Binary()
{
    cv::extractChannel(source_image_, blue_channel_, 0);
    cv::extractChannel(source_image_, green_channel_, 1);
    cv::extractChannel(source_image_, red_channel_, 2);

    if (options_.target_is_red == 1)
    {
        cv::subtract(red_channel_, green_channel_, binary_image_);
    }
    else
    {
        cv::subtract(blue_channel_, green_channel_, binary_image_);
    }

    cv::threshold(binary_image_, binary_image_, options_.binary_thresh, 255, cv::THRESH_BINARY);
}

void PreprocessCore::process(const cv::UMat& image)
{
    image.copyTo(source_image_);
    const cv::Mat element = buildMorphKernel();

    switch (options_.preprocess_method)
    {
        case PreProcessMethod::HSV:
            hsv2Binary();
            break;
        case PreProcessMethod::SINGLE_CHANNEL:
            bgr2Binary();
            break;
        default:
            bgr2Binary();
            break;
    }

    switch (options_.morph_type)
    {
        case MorphType::ERODE:
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_ERODE, element, cv::Point(-1, -1), options_.morph_iterations);
            break;
        case MorphType::DILATE:
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_DILATE, element, cv::Point(-1, -1), options_.morph_iterations);
            break;
        case MorphType::OPEN:
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_OPEN, element, cv::Point(-1, -1), options_.morph_iterations);
            break;
        case MorphType::CLOSE:
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_CLOSE, element, cv::Point(-1, -1), options_.morph_iterations);
            break;
        case MorphType::GRADIENT:
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_GRADIENT, element, cv::Point(-1, -1), options_.morph_iterations);
            break;
        case MorphType::TOPHAT:
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_TOPHAT, element, cv::Point(-1, -1), options_.morph_iterations);
            break;
        case MorphType::BLACKHAT:
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_BLACKHAT, element, cv::Point(-1, -1), options_.morph_iterations);
            break;
        case MorphType::HITMISS:
            morphologyEx(binary_image_, morpro_image_, cv::MORPH_HITMISS, element, cv::Point(-1, -1), options_.morph_iterations);
            break;
        case MorphType::DISABLE:
            binary_image_.copyTo(morpro_image_);
            break;
        default:
            binary_image_.copyTo(morpro_image_);
            break;
    }
}
}  // namespace rm_radarplugin
