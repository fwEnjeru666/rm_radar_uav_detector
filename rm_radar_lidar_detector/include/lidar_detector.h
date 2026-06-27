#pragma once

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>
#include <rm_radar_msgs/DroneDetection.h>
#include <geometry_msgs/Point.h>

#include <dynamic_reconfigure/server.h>
#include <rm_radar_lidar_detector/FilterParamsConfig.h>
#include <ros/callback_queue.h>

#include <deque>
#include <mutex>
#include <memory>
#include <thread>
#include <vector>
#include <string>

#include "types.h"
#include "cloud_processor.h"
#include "dynamic_detector.h"
#include "cluster_filter.h"
#include "recognition.h"
#include "track.h"
#include "visualizer.h"

namespace rm_radar_lidar_detector
{

enum class DetectionMode {
    DYNAMIC = 0,  // Detect on dynamic points only 
    DIRECT = 1    // Detect on accumulated cloud 
};

class LidarDetector : public nodelet::Nodelet
{
public:
    LidarDetector() = default;
    virtual ~LidarDetector() override;

    virtual void onInit() override;

private:
    // 启动和参数
    void initialize(ros::NodeHandle& nh);
    void initParams();
    void initROS();
    void setupDynamicReconfigure();
    void dynamicReconfigureCallback(FilterParamsConfig& config, uint32_t level);

    using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
    using PointCloudPtr = PointCloud::Ptr;

    struct DetectionInput
    {
        std::deque<PointCloudPtr> cloud_queue;
        std::deque<PointCloudPtr> raw_cloud_queue;
        PointCloudPtr accumulated_cloud;
        PointCloudPtr raw_accumulated_cloud;
        PointCloudPtr raw_detection_source;
        std::string frame_id;
        ros::Time stamp;
    };

    struct DroneCandidate
    {
        bool evaluated = false;
        bool found = false;
        PointCloudPtr cluster;
        ClusterFilter::ClusterMap clusters;
        BBox3D bbox;
        Eigen::Vector4f centroid = Eigen::Vector4f::Zero();
        std::vector<ClusterDebugInfo> debug_infos;
        PointCloudPtr raw_source_cloud;
        PointCloudPtr raw_drone_cloud;
        PointCloudPtr debug_drone_cloud;
        PointCloudPtr target_source_drone_cloud;
    };

    // 1. 点订阅
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    void processInputCloud(const PointCloudPtr& cloud,
                           const std::string& frame_id,
                           const ros::Time& stamp);

    // 2. 处理
    void clusterTimerCallback(const ros::TimerEvent& event);
    DetectionInput processQueuedClouds();

    // 3. 找 drone
    DroneCandidate findDrone(const DetectionInput& input);
    DroneCandidate findDroneNearTrack(const DetectionInput& input);
    DroneCandidate findDynamicDrone(const DetectionInput& input);
    DroneCandidate findDirectDrone(const DetectionInput& input);
    DroneCandidate findBestClusterFromPoints(const std::vector<ClusterPoint>& points,
                                             const PointCloudPtr& raw_source_cloud);
    void handleDroneCandidate(DroneCandidate& candidate,
                              const std::string& frame_id,
                              const ros::Time& stamp);

    // 4. 找 target
    void findTarget(const PointCloudPtr& target_source_drone_cloud,
                    const std::string& frame_id,
                    const ros::Time& stamp);

    // 5. 发布
    void publishResults(const pcl::PointCloud<pcl::PointXYZ>::Ptr& detected_cluster,
                        const BBox3D& detected_bbox,
                        const Eigen::Vector4f& detected_centroid,
                        double confidence,
                        const std::string& frame_id,
                        const ros::Time& stamp,
                        const PointCloudPtr& debug_drone_cloud,
                        const PointCloudPtr& target_source_drone_cloud);
    void publishSelectedTargetPoint(const pcl::PointXYZ& point,
                                    const std::string& frame_id,
                                    const ros::Time& stamp);
    bool getLatestTargetPoint(const std::string& frame_id,
                              const ros::Time& stamp,
                              geometry_msgs::Point& target_point) const;
    void publishTrackState(const ros::Time& stamp,
                           const BBox3D* bbox_hint,
                           const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& track_cloud = nullptr);
    void publishClusterDebugMarkers(const std::vector<ClusterDebugInfo>& debug_infos,
                                    const std::string& frame_id,
                                    const ros::Time& stamp);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cropRawDroneCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_source_cloud,
        const BBox3D& detected_bbox);
    static double clamp01(double value)
    {
        return std::max(0.0, std::min(1.0, value));
    };
    
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    ros::Subscriber cloud_sub_;
    ros::Publisher filtered_cloud_pub_;
    ros::Publisher removed_plane_cloud_pub_;
    ros::Publisher accumulated_cloud_pub_;
    ros::Publisher drone_cloud_pub_;
    ros::Publisher target_cloud_pub_;
    ros::Publisher target_projected_surface_cloud_pub_;
    ros::Publisher drone_aabb_accumulated_cloud_pub_;
    ros::Publisher target_marker_pub_;
    ros::Publisher selected_target_point_pub_;
    ros::Publisher marker_pub_;
    ros::Publisher track_marker_pub_;
    ros::Publisher cluster_debug_marker_array_pub_;
    ros::Publisher detection_pub_;
    ros::Publisher track_pub_;
    
    ros::Timer cluster_timer_;
    tf::TransformBroadcaster tf_broadcaster_;
    
    boost::shared_ptr<dynamic_reconfigure::Server<FilterParamsConfig>> dr_server_;

    CloudProcessor cloud_processor_;
    DynamicDetector dynamic_detector_;
    ClusterFilter cluster_filter_;
    CandidateSelector candidate_selector_;
    TargetCloudExtractor target_cloud_extractor_;
    TargetFrontViewController target_front_view_controller_;
    SingleTargetTracker tracker_;
    Visualizer visualizer_;

    std::string cloud_topic_;
    std::string frame_id_;
    int cloud_queue_size_;
    size_t frame_gap_;
    float eps_;
    int minPts_;
    DetectionMode detection_mode_;
    bool verbose_log_;
    bool publish_cluster_debug_markers_;
    bool publish_plane_debug_clouds_;
    bool track_enable_;
    bool track_local_cluster_enable_;
    bool publish_track_text_;
    bool publish_track_target_marker_;
    bool publish_track_trajectory_;
    bool publish_raw_drone_cloud_;
    bool extract_target_from_input_cloud_;
    bool project_input_cloud_directly_;
    double selected_target_point_marker_scale_{0.01};
    double target_point_smoothing_alpha_{1.0};
    double target_point_max_jump_distance_{0.8};
    int accumulated_publish_divider_;
    int drone_aabb_accumulation_max_points_;
    double drone_aabb_accumulation_reset_distance_{0.25};
    uint64_t cluster_tick_count_;

    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_queue_;
    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> raw_cloud_queue_;
    std::vector<ClusterPoint> dynamic_points_buffer_;
    std::vector<ClusterPoint> airspace_points_buffer_;
    std::vector<int> nn_indices_buffer_;
    std::vector<float> nn_distances_buffer_;
    std::deque<geometry_msgs::Point> track_trajectory_points_;
    CloudProcessor::PlaneRemovalDebug plane_debug_buffer_;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cur_filtered_cloud_;
    ros::Time latest_cloud_stamp_;
    geometry_msgs::Point latest_target_point_;
    std::string latest_target_point_frame_id_;
    ros::Time latest_target_point_stamp_;
    bool has_latest_target_point_{false};
    
    std::mutex cloud_mutex_;
    mutable std::mutex target_point_mutex_;
    ros::CallbackQueue callback_queue_;
    std::thread worker_thread_;
    std::string track_frame_id_;
};

}  // namespace rm_radar_lidar_detector
