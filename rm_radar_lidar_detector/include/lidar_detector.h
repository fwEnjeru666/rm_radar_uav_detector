#pragma once

#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>
#include <rm_radar_msgs/DroneDetection.h>

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
#include "dbscan.h"
#include "cloud_processor.h"
#include "dynamic_detector.h"
#include "cluster_filter.h"
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
    void initialize(ros::NodeHandle& nh);
    void initParams();
    void initROS();
    void setupDynamicReconfigure();

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    void clusterTimerCallback(const ros::TimerEvent& event);
    void dynamicReconfigureCallback(FilterParamsConfig& config, uint32_t level);

    void publishResults(const pcl::PointCloud<pcl::PointXYZ>::Ptr& detected_cluster,
                        const BBox3D& detected_bbox,
                        const Eigen::Vector4f& detected_centroid,
                        const std::string& frame_id,
                        const ros::Time& stamp);
    void publishClusterDebugMarkers(const std::vector<ClusterDebugInfo>& debug_infos,
                                    const std::string& frame_id,
                                    const ros::Time& stamp);

    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    ros::Subscriber cloud_sub_;
    ros::Publisher filtered_cloud_pub_;
    ros::Publisher accumulated_cloud_pub_;
    ros::Publisher drone_cloud_pub_;
    ros::Publisher marker_pub_;
    ros::Publisher cluster_debug_marker_array_pub_;
    ros::Publisher detection_pub_;
    
    ros::Timer cluster_timer_;
    tf::TransformBroadcaster tf_broadcaster_;
    
    boost::shared_ptr<dynamic_reconfigure::Server<FilterParamsConfig>> dr_server_;

    CloudProcessor cloud_processor_;
    DynamicDetector dynamic_detector_;
    ClusterFilter cluster_filter_;
    Visualizer visualizer_;

    std::string cloud_topic_;
    std::string frame_id_;
    int cloud_queue_size_;
    size_t frame_gap_;
    float eps_;
    int minPts_;
    float expansion_m_;
    DetectionMode detection_mode_;
    bool verbose_log_;
    bool publish_cluster_debug_markers_;
    int accumulated_publish_divider_;
    uint64_t cluster_tick_count_;

    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_queue_;
    std::vector<ClusterPoint> dynamic_points_buffer_;
    std::vector<ClusterPoint> airspace_points_buffer_;
    std::vector<int> nn_indices_buffer_;
    std::vector<float> nn_distances_buffer_;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cur_filtered_cloud_;
    ros::Time latest_cloud_stamp_;
    
    std::mutex cloud_mutex_;
    ros::CallbackQueue callback_queue_;
    std::thread worker_thread_;
};

}  // namespace rm_radar_lidar_detector
