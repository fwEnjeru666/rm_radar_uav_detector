#pragma once

#include <mutex>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/core.hpp>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

namespace rm_radarplugin
{
class FindRefPt
{
public:
    explicit FindRefPt(ros::NodeHandle& nh);
    void spin();

private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void handleMouse(int event, int x, int y, int flags);
    static void mouseCallback(int event, int x, int y, int flags, void* userdata);

    cv::Point2d findAutoPoint(const cv::Mat& image, bool& found) const;
    cv::Mat drawOverlay(const cv::Mat& image);
    void savePoint(const cv::Point2d& point, const std::string& mode);
    void updateSavedPointValidation();
    void writeYaml(const cv::Point2d& point, const std::string& mode, bool valid);
    void resetSelection();
    std::string getDefaultOutputPath() const
    {
        const std::string package_path = ros::package::getPath("rm_radar_img_proc");
        if (package_path.empty())
        {
            return "ref_pt_pos.yaml";
        }
        return package_path + "/config/ref_pt_pos.yaml";
    }

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    cv_bridge::CvImageConstPtr latest_image_msg_;
    std::mutex image_mutex_;

    std::string image_topic_;
    std::string output_path_;
    std::string mode_;
    std::string window_name_;

    bool auto_save_{};
    bool saved_once_{};
    bool has_manual_point_{};
    bool has_auto_point_{};
    bool has_saved_point_{};
    bool failed_{};
    bool yaml_invalidated_{};
    bool adjustment_mode_{};
    std::string saved_mode_{};

    int red_min_diff_{};
    int red_s_min_{};
    int red_v_min_{};
    int min_red_area_{};
    int blur_size_{};
    int center_cross_size_{};
    double fail_pixel_threshold_{};
    double saved_auto_error_px_{};

    cv::Point2d manual_point_{};
    cv::Point2d auto_point_{};
    cv::Point2d saved_point_{};
};
}  // namespace rm_radarplugin
