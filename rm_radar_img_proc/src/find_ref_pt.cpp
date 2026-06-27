#include <find_ref_pt.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace rm_radarplugin
{
namespace
{
cv::Point toImagePoint(const cv::Point2d& point)
{
    return cv::Point(static_cast<int>(std::round(point.x)), static_cast<int>(std::round(point.y)));
}
}  // namespace

FindRefPt::FindRefPt(ros::NodeHandle& nh) : nh_(nh), it_(nh_)
{
    nh_.param<std::string>("image_topic", image_topic_, "/hk_camera/image_raw");
    nh_.param<std::string>("output_path", output_path_, getDefaultOutputPath());
    nh_.param<std::string>("mode", mode_, "manual");
    nh_.param<std::string>("window_name", window_name_, "find_ref_pt");
    nh_.param("red_min_diff", red_min_diff_, 30);
    nh_.param("red_s_min", red_s_min_, 80);
    nh_.param("red_v_min", red_v_min_, 80);
    nh_.param("min_red_area", min_red_area_, 5);
    nh_.param("blur_size", blur_size_, 5);
    nh_.param("adjustment_mode", adjustment_mode_, false);
    nh_.param("center_cross_size", center_cross_size_, 240);
    nh_.param("fail_pixel_threshold", fail_pixel_threshold_, 20.0);

    std::transform(mode_.begin(), mode_.end(), mode_.begin(), [](unsigned char ch) { return std::tolower(ch); });
    if (mode_ != "manual" && mode_ != "auto")
    {
        ROS_WARN("Invalid mode '%s', fallback to manual", mode_.c_str());
        mode_ = "manual";
    }
    nh_.param("auto_save", auto_save_, mode_ == "auto");

    image_sub_ = it_.subscribe(image_topic_, 1, &FindRefPt::imageCallback, this);
    ROS_INFO("find_ref_pt subscribes image topic: %s", image_topic_.c_str());
    ROS_INFO("find_ref_pt output yaml: %s", output_path_.c_str());
}

void FindRefPt::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        auto image_msg = cv_bridge::toCvShare(msg, "bgr8");
        std::lock_guard<std::mutex> lock(image_mutex_);
        latest_image_msg_ = image_msg;
    }
    catch (const cv_bridge::Exception& err)
    {
        ROS_ERROR_THROTTLE(2.0, "cv_bridge failed: %s", err.what());
    }
}

cv::Point2d FindRefPt::findAutoPoint(const cv::Mat& image, bool& found) const
{
    found = false;
    if (image.empty() || image.channels() != 3)
    {
        return cv::Point2d();
    }

    cv::Mat source = image;
    cv::Mat blurred;
    if (blur_size_ > 1)
    {
        const int kernel_size = (blur_size_ % 2 == 1) ? blur_size_ : blur_size_ + 1;
        cv::GaussianBlur(source, blurred, cv::Size(kernel_size, kernel_size), 0.0);
        source = blurred;
    }

    std::vector<cv::Mat> bgr_channels;
    cv::split(source, bgr_channels);

    cv::Mat hsv_image;
    cv::cvtColor(source, hsv_image, cv::COLOR_BGR2HSV);

    cv::Mat red_low_mask;
    cv::Mat red_high_mask;
    cv::inRange(hsv_image, cv::Scalar(0, red_s_min_, red_v_min_), cv::Scalar(10, 255, 255), red_low_mask);
    cv::inRange(hsv_image, cv::Scalar(170, red_s_min_, red_v_min_), cv::Scalar(180, 255, 255), red_high_mask);
    cv::Mat red_mask = red_low_mask | red_high_mask;

    cv::Mat red_channel;
    cv::Mat green_channel;
    cv::Mat blue_channel;
    bgr_channels[2].convertTo(red_channel, CV_32F);
    bgr_channels[1].convertTo(green_channel, CV_32F);
    bgr_channels[0].convertTo(blue_channel, CV_32F);

    cv::Mat max_non_red;
    cv::max(green_channel, blue_channel, max_non_red);
    cv::Mat red_diff = red_channel - max_non_red;
    cv::Mat diff_mask;
    cv::threshold(red_diff, diff_mask, red_min_diff_, 255.0, cv::THRESH_BINARY);
    diff_mask.convertTo(diff_mask, CV_8U);

    cv::Mat candidate_mask = red_mask & diff_mask;
    if (cv::countNonZero(candidate_mask) == 0)
    {
        return cv::Point2d();
    }

    cv::Mat hsv_channels[3];
    cv::split(hsv_image, hsv_channels);
    cv::Mat value_channel;
    hsv_channels[2].convertTo(value_channel, CV_32F);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(candidate_mask, labels, stats, centroids, 8, CV_32S);
    if (component_count <= 1)
    {
        return cv::Point2d();
    }

    cv::Mat score = red_diff * 3.0 + value_channel;
    int best_label = -1;
    double best_component_score = -1.0;
    for (int label = 1; label < component_count; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < min_red_area_)
        {
            continue;
        }

        const cv::Mat component_mask = labels == label;
        const double component_score = cv::mean(score, component_mask)[0] * area;
        if (component_score > best_component_score)
        {
            best_component_score = component_score;
            best_label = label;
        }
    }

    if (best_label < 0)
    {
        return cv::Point2d();
    }

    found = true;
    return cv::Point2d(centroids.at<double>(best_label, 0), centroids.at<double>(best_label, 1));
}

cv::Mat FindRefPt::drawOverlay(const cv::Mat& image)
{
    cv::Mat draw_image = image.clone();
    auto_point_ = findAutoPoint(draw_image, has_auto_point_);
    updateSavedPointValidation();

    if (has_auto_point_)
    {
        const cv::Point auto_point = toImagePoint(auto_point_);
        cv::drawMarker(draw_image, auto_point, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 9, 2);
        cv::circle(draw_image, auto_point, 1, cv::Scalar(0, 0, 255), cv::FILLED);
    }

    if (!has_saved_point_ && has_manual_point_)
    {
        const cv::Point manual_point = toImagePoint(manual_point_);
        cv::drawMarker(draw_image, manual_point, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 9, 2);
        cv::circle(draw_image, manual_point, 1, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    if (has_saved_point_)
    {
        const cv::Point saved_point = toImagePoint(saved_point_);
        cv::drawMarker(draw_image, saved_point, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 9, 2);
        cv::circle(draw_image, saved_point, 1, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    if (adjustment_mode_)
    {
        const cv::Point center(draw_image.cols / 2, draw_image.rows / 2);
        const int cross_size = std::max(1, center_cross_size_);
        cv::line(draw_image, center + cv::Point(-cross_size, 0), center + cv::Point(cross_size, 0),
                 cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::line(draw_image, center + cv::Point(0, -cross_size), center + cv::Point(0, cross_size),
                 cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    std::ostringstream status;
    cv::Scalar status_color(255, 255, 255);
    if (failed_)
    {
        status_color = cv::Scalar(0, 0, 255);
        status << std::fixed << std::setprecision(3) << "FAILED! saved=(" << saved_point_.x << ", " << saved_point_.y << ")";
        if (has_auto_point_)
        {
            status << " auto=(" << auto_point_.x << ", " << auto_point_.y << ") diff=" << std::setprecision(1)
                   << saved_auto_error_px_ << "px > " << fail_pixel_threshold_ << "px | valid=false";
        }
        else
        {
            status << " auto=none | valid=false";
        }
    }
    else if (has_saved_point_)
    {
        status_color = cv::Scalar(0, 255, 0);
        status << std::fixed << std::setprecision(3) << "SAVED! (" << saved_point_.x << ", " << saved_point_.y << ")";
        if (has_auto_point_)
        {
            status << " auto=(" << auto_point_.x << ", " << auto_point_.y << ") diff=" << std::setprecision(1)
                   << saved_auto_error_px_ << "px";
        }
        else
        {
            status << " auto=none";
        }
    }
    else
    {
        status << "mode: " << mode_ << " | c: adjust cross " << (adjustment_mode_ ? "on" : "off")
               << " | click: manual save | s: save | r: reset | a: auto/manual | q/esc: quit";
    }

    cv::putText(draw_image, status.str(), cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX, 0.58, cv::Scalar(0, 0, 0), 4);
    cv::putText(draw_image, status.str(), cv::Point(10, 26), cv::FONT_HERSHEY_SIMPLEX, 0.58, status_color, 1);

    return draw_image;
}

void FindRefPt::updateSavedPointValidation()
{
    if (!has_saved_point_ || !has_auto_point_)
    {
        return;
    }

    const cv::Point2d delta = auto_point_ - saved_point_;
    saved_auto_error_px_ = std::sqrt(delta.x * delta.x + delta.y * delta.y);
    if (saved_auto_error_px_ <= fail_pixel_threshold_)
    {
        return;
    }

    if (failed_)
    {
        return;
    }

    failed_ = true;
    if (!yaml_invalidated_)
    {
        writeYaml(saved_point_, saved_mode_, false);
        yaml_invalidated_ = true;
    }
    ROS_ERROR("find_ref_pt failed: saved point (%.3f, %.3f), auto point (%.3f, %.3f), diff %.2f px > %.2f px",
              saved_point_.x, saved_point_.y, auto_point_.x, auto_point_.y, saved_auto_error_px_, fail_pixel_threshold_);
}

void FindRefPt::writeYaml(const cv::Point2d& point, const std::string& mode, bool valid)
{
    const auto slash_pos = output_path_.find_last_of('/');
    if (slash_pos != std::string::npos)
    {
        const std::string dir = output_path_.substr(0, slash_pos);
        mkdir(dir.c_str(), 0755);
    }

    const std::string tmp_path = output_path_ + ".tmp";
    std::ofstream output(tmp_path);
    if (!output.is_open())
    {
        ROS_ERROR("Failed to open ref point yaml: %s", tmp_path.c_str());
        return;
    }

    const std::time_t now = std::time(nullptr);
    std::tm local_time{};
    localtime_r(&now, &local_time);
    char time_buffer[32]{};
    std::strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", &local_time);

    output << std::fixed << std::setprecision(6);
    output << "valid: " << (valid ? "true" : "false") << "\n";
    output << "x: " << point.x << "\n";
    output << "y: " << point.y << "\n";
    output << "mode: " << mode << "\n";
    output << "image_topic: " << image_topic_ << "\n";
    output << "saved_at: '" << time_buffer << "'\n";
    output.close();

    if (std::rename(tmp_path.c_str(), output_path_.c_str()) != 0)
    {
        ROS_ERROR("Failed to rename %s to %s", tmp_path.c_str(), output_path_.c_str());
        return;
    }
}

void FindRefPt::savePoint(const cv::Point2d& point, const std::string& mode)
{
    writeYaml(point, mode, true);

    saved_once_ = true;
    saved_point_ = point;
    has_saved_point_ = true;
    saved_mode_ = mode;
    failed_ = false;
    yaml_invalidated_ = false;
    saved_auto_error_px_ = 0.0;
    ROS_INFO("Saved ref point (%.3f, %.3f) to %s", point.x, point.y, output_path_.c_str());
}

void FindRefPt::resetSelection()
{
    if (has_saved_point_)
    {
        writeYaml(saved_point_, saved_mode_, false);
        yaml_invalidated_ = true;
        ROS_WARN("Reset ref point selection, marked yaml invalid: %s", output_path_.c_str());
    }

    saved_once_ = false;
    has_manual_point_ = false;
    has_saved_point_ = false;
    failed_ = false;
    saved_mode_.clear();
    saved_auto_error_px_ = 0.0;
    manual_point_ = cv::Point2d();
    saved_point_ = cv::Point2d();
}

void FindRefPt::handleMouse(int event, int x, int y, int /*flags*/)
{
    if (event != cv::EVENT_LBUTTONDOWN)
    {
        return;
    }

    manual_point_ = cv::Point2d(static_cast<double>(x), static_cast<double>(y));
    has_manual_point_ = true;
    savePoint(manual_point_, "manual");
}

void FindRefPt::mouseCallback(int event, int x, int y, int flags, void* userdata)
{
    if (userdata == nullptr)
    {
        return;
    }
    static_cast<FindRefPt*>(userdata)->handleMouse(event, x, y, flags);
}

void FindRefPt::spin()
{
    cv::namedWindow(window_name_, cv::WINDOW_NORMAL);
    cv::setMouseCallback(window_name_, &FindRefPt::mouseCallback, this);

    ros::Rate rate(30.0);
    while (ros::ok())
    {
        cv_bridge::CvImageConstPtr image_msg;
        {
            std::lock_guard<std::mutex> lock(image_mutex_);
            image_msg = latest_image_msg_;
        }

        if (image_msg)
        {
            cv::Mat draw_image = drawOverlay(image_msg->image);
            cv::imshow(window_name_, draw_image);

            if (mode_ == "auto" && auto_save_ && !saved_once_ && has_auto_point_)
            {
                savePoint(auto_point_, "auto");
            }
        }

        const int key = cv::waitKey(1) & 0xff;
        if (key == 27 || key == 'q')
        {
            break;
        }
        if (key == 'a')
        {
            mode_ = (mode_ == "auto") ? "manual" : "auto";
            ROS_INFO("find_ref_pt mode switched to %s", mode_.c_str());
        }
        if (key == 'r')
        {
            resetSelection();
        }
        if (key == 'c')
        {
            adjustment_mode_ = !adjustment_mode_;
            ROS_INFO("find_ref_pt adjustment mode %s", adjustment_mode_ ? "enabled" : "disabled");
        }
        if (key == 's')
        {
            if (mode_ == "auto" && has_auto_point_)
            {
                savePoint(auto_point_, "auto");
            }
            else if (has_manual_point_)
            {
                savePoint(manual_point_, "manual");
            }
            else
            {
                ROS_WARN("No point available to save");
            }
        }

        ros::spinOnce();
        rate.sleep();
    }

    cv::destroyWindow(window_name_);
}
}  // namespace rm_radarplugin

int main(int argc, char** argv)
{
    ros::init(argc, argv, "find_ref_pt");
    ros::NodeHandle nh("~");
    rm_radarplugin::FindRefPt selector(nh);
    selector.spin();
    return 0;
}
