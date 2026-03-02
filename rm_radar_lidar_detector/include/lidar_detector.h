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

#include <deque>
#include <mutex>
#include <memory>

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
    virtual ~LidarDetector() = default;

    virtual void onInit() override;

private:
    void initialize(ros::NodeHandle& nh);
    void initParams();
    void initROS();
    void setupDynamicReconfigure();

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    void clusterTimerCallback(const ros::TimerEvent& event);
    void dynamicReconfigureCallback(FilterParamsConfig& config, uint32_t level);

    // Processing pipeline - two modes
    bool detectDynamic();   // Mode 0: Dynamic point detection
    bool detectDirect();    // Mode 1: Direct detection on accumulated cloud
    void updateTracking();
    void publishResults();

    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    ros::Subscriber cloud_sub_;
    ros::Publisher filtered_cloud_pub_;
    ros::Publisher accumulated_cloud_pub_;
    ros::Publisher drone_cloud_pub_;
    ros::Publisher marker_pub_;
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

    std::deque<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_queue_;
    std::vector<ClusterPoint> dynamic_points_;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cur_filtered_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr detected_cluster_;
    
    Eigen::Vector4f cur_centroid_;
    BBox3D cur_detected_bbox_;
    
    pcl::PointXYZ track_min_pt_;
    pcl::PointXYZ track_max_pt_;
    bool has_tracked_;
    bool first_cluster_;
    bool expand_lock_;
    
    std::mutex cloud_mutex_;
};

}  // namespace rm_radar_lidar_detector
