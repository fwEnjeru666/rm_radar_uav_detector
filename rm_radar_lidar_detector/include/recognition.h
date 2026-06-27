#pragma once

#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <ros/time.h>

namespace rm_radar_lidar_detector
{



class TargetCloudExtractor
{
public:
    using PointT = pcl::PointXYZ;
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudPtr = PointCloud::Ptr;
    using PointCloudConstPtr = PointCloud::ConstPtr;

    struct Params
    {
        int min_support_points = 3;
        float height_window = 0.2f;
    };

    void setParams(const Params& params);
    const Params& getParams() const { return params_; }
    PointCloudPtr extract(const PointCloudConstPtr& detected_cluster) const;

private:
    Params params_;
};


class OnlineTemplateRecognition
{
public:
    using PointT = pcl::PointXYZ;
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudConstPtr = PointCloud::ConstPtr;

    struct Params
    {
        float support_radius = 0.25f;
        int radial_bins = 8;
        int min_support_points = 4;
        float max_descriptor_distance = 0.75f;
        float template_update_rate = 0.05f;
        float min_update_confidence = 0.65f;
        float previous_position_gate = 0.0f;
        float previous_position_weight = 0.0f;
    };

    struct Result
    {
        bool matched = false;
        PointT marked_point;
        PointT anchor_point;
        std::size_t anchor_index = 0;
        float confidence = 0.0f;
        float descriptor_distance = std::numeric_limits<float>::infinity();
        int support_points = 0;
    };

    OnlineTemplateRecognition() = default;
    explicit OnlineTemplateRecognition(const Params& params);

    void setParams(const Params& params);
    const Params& getParams() const;

    bool initialize(const PointCloudConstPtr& target_cloud, const PointT& marked_point);
    Result match(const PointCloudConstPtr& target_cloud) const;
    Result update(const PointCloudConstPtr& target_cloud);

    void reset();
    bool initialized() const;

    const PointT& lastMarkedPoint() const;
    const PointT& templateAnchorPoint() const;

private:
    struct Descriptor
    {
        bool valid = false;
        std::vector<float> radial_histogram;
        int support_points = 0;
        float mean_radius = 0.0f;
    };

    static bool isFinite(const PointT& point);
    static float squaredDistance(const PointT& first, const PointT& second);
    static PointT add(const PointT& point, const PointT& offset);
    static PointT subtract(const PointT& first, const PointT& second);

    bool sanitizeParams();
    int findNearestPointIndex(const PointCloudConstPtr& cloud, const PointT& point) const;
    Descriptor buildDescriptor(const PointCloudConstPtr& cloud, int anchor_index) const;
    float descriptorDistance(const Descriptor& candidate) const;
    float confidenceFromDistance(float descriptor_distance) const;
    void blendTemplateDescriptor(const Descriptor& candidate);

    Params params_;
    bool initialized_ = false;
    Descriptor template_descriptor_;
    PointT template_anchor_point_;
    PointT marked_offset_from_anchor_;
    PointT last_marked_point_;
};

class FrontViewProjector
{
public:
    using PointT = pcl::PointXYZ;
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudConstPtr = PointCloud::ConstPtr;

    struct Params
    {
        float grid_resolution = 0.001f;
        bool adaptive_resolution = false;
        float adaptive_resolution_scale = 1.5f;
        float max_grid_resolution = 0.01f;
        int max_image_size = 700;
        int min_points = 3;
    };

    struct Projection
    {
        bool valid = false;
        cv::Mat image;
        cv::Mat source_index_image;
        float resolution = 0.02f;
        float min_u = 0.0f;
        float max_u = 0.0f;
        float min_v = 0.0f;
        float max_v = 0.0f;
        int occupied_cells = 0;
        PointCloud::Ptr surface_cloud;
    };

    FrontViewProjector() = default;
    explicit FrontViewProjector(const Params& params);

    void setParams(const Params& params);
    const Params& getParams() const;

    bool initialize(const PointCloudConstPtr& target_cloud,
                    const Eigen::Vector3f& sensor_origin = Eigen::Vector3f::Zero());
    Projection project(const PointCloudConstPtr& target_cloud,
                       const Eigen::Vector3f& sensor_origin = Eigen::Vector3f::Zero()) const;

    void reset();
    bool initialized() const;

    const Eigen::Vector3f& origin() const;
    const Eigen::Vector3f& axisU() const;
    const Eigen::Vector3f& axisV() const;
    const Eigen::Vector3f& frontNormal() const;

private:
    struct ProjectedPoint
    {
        float u = 0.0f;
        float v = 0.0f;
        float range_sq = 0.0f;
        PointT point;
    };

    static bool isFinite(const PointT& point);
    static Eigen::Vector3f toEigen(const PointT& point);
    static bool computeCentroid(const PointCloudConstPtr& cloud, Eigen::Vector3f& centroid, int& finite_count);
    float estimateMedianNearestDistance(const PointCloudConstPtr& cloud) const;
    bool sanitizeParams();

    Params params_;
    bool initialized_ = false;
    Eigen::Vector3f origin_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f axis_u_ = Eigen::Vector3f::UnitX();
    Eigen::Vector3f axis_v_ = Eigen::Vector3f::UnitZ();
    Eigen::Vector3f front_normal_ = Eigen::Vector3f::UnitY();
};

class TargetFrontViewController
{
public:
    using PointT = pcl::PointXYZ;
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudPtr = PointCloud::Ptr;
    using PointCloudConstPtr = PointCloud::ConstPtr;
    using SelectedPointCallback = std::function<void(const PointT&, const std::string&, const ros::Time&)>;

    struct Params
    {
        bool enabled = false;
        std::string front_view_window = "target_front_view";
        std::string template_window = "target_template_view";
        int accumulation_frames = 1;
        int patch_radius = 20;
        double patch_min_score = 0.25;
        double patch_update_min_score = 0.75;
        double patch_learning_rate = 0.10;
        double patch_train_threshold = 0.45;
        int patch_train_min_points = 3;
        FrontViewProjector::Params projector;
    };

    struct Output
    {
        PointCloudPtr projected_surface_cloud;
    };

    TargetFrontViewController() = default;

    void setParams(const Params& params);
    const Params& getParams() const { return params_; }
    void setSelectedPointCallback(SelectedPointCallback callback);
    Output show(const PointCloudConstPtr& target_cloud, const std::string& frame_id, const ros::Time& stamp);
    void resetAccumulation();

private:
    static void onMouse(int event, int x, int y, int flags, void* userdata);
    void handleClick(int display_x, int display_y);
    bool findNearestProjectedPoint(const cv::Mat& source_index_image,
                                   int row,
                                   int col,
                                   int search_radius,
                                   int& point_index,
                                   int& nearest_row,
                                   int& nearest_col) const;
    cv::Mat buildProcessedTargetMask(const cv::Mat& source_index_image) const;
    cv::Mat buildRawOccupancyMask(const cv::Mat& source_index_image) const;
    cv::Mat buildSelectionIndexImage(const cv::Mat& source_index_image,
                                     const cv::Mat& processed_mask) const;
    cv::Mat extractPatchWithBorder(const cv::Mat& image, int center_row, int center_col, int radius) const;
    cv::Mat buildTemplateDisplayImage(const cv::Mat& patch) const;
    void initializeSelectedTemplate(const cv::Mat& patch, int patch_radius);
    bool trainSelectedTemplate(const cv::Mat& patch, int patch_radius);
    bool matchSelectedTargetPatch(const cv::Mat& raw_occupancy_mask,
                                  const cv::Mat& processed_mask,
                                  int& matched_row,
                                  int& matched_col,
                                  double& match_score) const;

    Params params_;
    FrontViewProjector projector_;
    SelectedPointCallback selected_point_callback_;

    FrontViewProjector::Projection latest_projection_;
    cv::Mat latest_selection_index_image_;
    cv::Mat latest_processed_mask_;
    cv::Mat latest_raw_occupancy_mask_;
    cv::Mat selected_patch_;
    cv::Mat selected_template_model_;
    std::vector<cv::Point> selected_template_points_;
    cv::Mat selected_template_display_image_;
    bool has_selected_template_display_ = false;
    std::string latest_projection_frame_id_;
    ros::Time latest_projection_stamp_;
    cv::Size latest_display_image_size_;
    double latest_display_scale_ = 1.0;
    double latest_display_offset_x_ = 0.0;
    double latest_display_offset_y_ = 0.0;
    int selected_projection_row_ = -1;
    int selected_projection_col_ = -1;
    int selected_click_display_x_ = -1;
    int selected_click_display_y_ = -1;
    bool has_selected_click_display_ = false;
    PointT selected_point_;
    bool has_selected_point_ = false;
    bool has_selected_patch_ = false;
    std::deque<PointCloudPtr> accumulation_queue_;
    std::mutex mutex_;
};

}  // namespace rm_radar_lidar_detector
