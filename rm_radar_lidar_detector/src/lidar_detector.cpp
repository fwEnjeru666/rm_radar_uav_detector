#include "lidar_detector.h"
#include <pluginlib/class_list_macros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rm_radar_msgs/DroneDetection.h>

PLUGINLIB_EXPORT_CLASS(rm_radar_lidar_detector::LidarDetector, nodelet::Nodelet)

namespace rm_radar_lidar_detector
{

    void LidarDetector::onInit()
    {
        NODELET_INFO("LidarDetector initializing...");
        
        ros::NodeHandle nh = getNodeHandle();
        ros::NodeHandle private_nh = getPrivateNodeHandle();
        
        initialize(private_nh);
        
        NODELET_INFO("LidarDetector initialized successfully");
    }

    void LidarDetector::initialize(ros::NodeHandle& nh)
    {
        nh_ = nh;
        private_nh_ = nh;
        
        initParams();
        initROS();
        setupDynamicReconfigure();
    }

    void LidarDetector::initParams()
    {
        // Topic and frame
        private_nh_.param<std::string>("cloud_topic", cloud_topic_, "/livox/lidar");
        private_nh_.param<std::string>("frame_id", frame_id_, "livox_frame");
        
        FilterParams filter_params;
        private_nh_.param<float>("pass_x_min", filter_params.pass_x_min, 0.0f);
        private_nh_.param<float>("pass_x_max", filter_params.pass_x_max, 35.0f);
        private_nh_.param<float>("pass_y_min", filter_params.pass_y_min, -10.0f);
        private_nh_.param<float>("pass_y_max", filter_params.pass_y_max, 8.0f);
        private_nh_.param<float>("pass_z_min", filter_params.pass_z_min, -50.0f);
        private_nh_.param<float>("pass_z_max", filter_params.pass_z_max, 7.0f);
        private_nh_.param<float>("voxel_leaf_size", filter_params.voxel_leaf, 0.1f);
        private_nh_.param<float>("radius_search", filter_params.radius_search, 0.5f);
        private_nh_.param<int>("min_neighbors", filter_params.min_neighbors, 5);
        private_nh_.param<float>("ransac_distance_threshold", filter_params.ransac_distance_threshold, 0.5f);
        cloud_processor_.setParams(filter_params);
        
        DynamicDetectorParams dynamic_params;
        private_nh_.param<float>("distance_threshold", dynamic_params.distance_threshold, 2.0f);
        private_nh_.param<float>("aircraft_min_z", dynamic_params.min_z, -3.0f);
        dynamic_detector_.setParams(dynamic_params);
        
        ClusterFilterParams cluster_params;
        private_nh_.param<double>("max_volume", cluster_params.max_volume, 10.0);
        private_nh_.param<double>("min_volume", cluster_params.min_volume, 0.1);
        private_nh_.param<double>("max_ratio", cluster_params.max_ratio, 10.0);
        private_nh_.param<double>("min_ratio", cluster_params.min_ratio, 0.1);
        float min_z;
        private_nh_.param<float>("aircraft_min_z", min_z, -3.0f);
        cluster_params.min_z = min_z;
        cluster_filter_.setParams(cluster_params);
        
        // Clustering parameters
        private_nh_.param<float>("eps", eps_, 0.5f);
        private_nh_.param<int>("minPts", minPts_, 5);
        private_nh_.param<float>("expansion_m", expansion_m_, 1.0f);
        
        int mode;
        private_nh_.param<int>("detection_mode", mode, 0);
        detection_mode_ = static_cast<DetectionMode>(mode);
        
        private_nh_.param<int>("cloud_queue_size", cloud_queue_size_, 10);
        int frame_gap_int;
        private_nh_.param<int>("frame_gap", frame_gap_int, 4);
        frame_gap_ = static_cast<size_t>(frame_gap_int);
        first_cluster_ = true;
        has_tracked_ = false;
        expand_lock_ = false;
        cur_filtered_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        accumulated_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        detected_cluster_.reset(new pcl::PointCloud<pcl::PointXYZ>);
        
        NODELET_INFO("Parameters initialized");
    }

    void LidarDetector::initROS()
    {
        // Subscribers
        cloud_sub_ = nh_.subscribe(cloud_topic_, 1, &LidarDetector::cloudCallback, this);
        
        // Publishers
        filtered_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
        accumulated_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("accumulated_cloud", 1);
        drone_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("drone_cloud", 1);
        marker_pub_ = nh_.advertise<visualization_msgs::Marker>("detection_marker", 10);
        detection_pub_ = nh_.advertise<rm_radar_msgs::DroneDetection>("lidar_detection", 1);  // 供 rm_track 融合使用
        
        // Timer for clustering (10 Hz)
        cluster_timer_ = nh_.createTimer(ros::Duration(0.1), &LidarDetector::clusterTimerCallback, this);
        
        // Initialize visualizer
        visualizer_.init(nh_);
        
        NODELET_INFO("ROS interfaces initialized");
    }

    void LidarDetector::setupDynamicReconfigure()
    {
        dr_server_.reset(new dynamic_reconfigure::Server<FilterParamsConfig>(nh_));
        dr_server_->setCallback(boost::bind(&LidarDetector::dynamicReconfigureCallback, this, _1, _2));
    }

    void LidarDetector::dynamicReconfigureCallback(FilterParamsConfig& config, uint32_t level)
    {
        // Update filter parameters
        FilterParams filter_params = cloud_processor_.getParams();
        filter_params.voxel_leaf = config.voxel_leaf_size;
        filter_params.pass_x_min = config.pass_x_min;
        filter_params.pass_x_max = config.pass_x_max;
        filter_params.pass_y_min = config.pass_y_min;
        filter_params.pass_y_max = config.pass_y_max;
        filter_params.pass_z_min = config.pass_z_min;
        filter_params.pass_z_max = config.pass_z_max;
        filter_params.radius_search = config.radius_search;
        filter_params.min_neighbors = config.min_neighbors;
        filter_params.ransac_distance_threshold = config.ransac_distance_threshold;
        cloud_processor_.setParams(filter_params);
        
        // Update dynamic detector parameters
        DynamicDetectorParams dynamic_params = dynamic_detector_.getParams();
        dynamic_params.distance_threshold = config.distance_threshold;
        dynamic_params.min_z = config.aircraft_min_z;
        dynamic_detector_.setParams(dynamic_params);
        
        // Update cluster filter parameters
        ClusterFilterParams cluster_params = cluster_filter_.getParams();
        cluster_params.max_volume = config.max_volume;
        cluster_params.min_volume = config.min_volume;
        cluster_params.max_ratio = config.max_ratio;
        cluster_params.min_ratio = config.min_ratio;
        cluster_params.min_z = config.aircraft_min_z;
        cluster_filter_.setParams(cluster_params);
        
        // Update clustering parameters
        eps_ = config.eps;
        minPts_ = config.minPts;
        expansion_m_ = config.expansion_m;
        
        // Update detection mode
        DetectionMode new_mode = static_cast<DetectionMode>(config.detection_mode);
        if (new_mode != detection_mode_) {
            detection_mode_ = new_mode;
            NODELET_INFO("Detection mode changed to: %s", 
                        detection_mode_ == DetectionMode::DYNAMIC ? "DYNAMIC" : "DIRECT");
        }
        
        NODELET_INFO("Dynamic reconfigure: voxel=%.3f, eps=%.3f, minPts=%d, mode=%s",
                    filter_params.voxel_leaf, eps_, minPts_,
                    detection_mode_ == DetectionMode::DYNAMIC ? "DYNAMIC" : "DIRECT");
    }

    void LidarDetector::cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*cloud_msg, *cloud);
        
        if (cloud->empty()) {
            NODELET_WARN_THROTTLE(5, "Received empty point cloud");
            return;
        }
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered = cloud_processor_.preprocess(cloud);
        
        if (filtered->empty()) {
            NODELET_WARN_THROTTLE(5, "Filtered cloud is empty");
            return;
        }
        
        {
            std::lock_guard<std::mutex> lock(cloud_mutex_);
            cloud_queue_.push_back(filtered);
            while (cloud_queue_.size() > static_cast<size_t>(cloud_queue_size_)) {
                cloud_queue_.pop_front();
            }
            cur_filtered_cloud_ = filtered;
            frame_id_ = cloud_msg->header.frame_id;
            accumulated_cloud_->clear();
            for (const auto& c : cloud_queue_) {
                *accumulated_cloud_ += *c;
            }
        }
        
        visualizer_.publishCloud(filtered, frame_id_, cloud_msg->header.stamp, filtered_cloud_pub_);
        visualizer_.publishCloud(accumulated_cloud_, frame_id_, cloud_msg->header.stamp, accumulated_cloud_pub_);
    }

    void LidarDetector::clusterTimerCallback(const ros::TimerEvent& event)
    {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        
        bool detected = false;
        
        if (detection_mode_ == DetectionMode::DYNAMIC) {
            detected = detectDynamic();
        } else {
            detected = detectDirect();
        }
        
        if (detected && detected_cluster_ && !detected_cluster_->empty()) {
            updateTracking();
            publishResults();
        }
    }

    bool LidarDetector::detectDynamic()
    {
        if (cloud_queue_.size() < frame_gap_ + 1) {
            NODELET_DEBUG_THROTTLE(1, "DYNAMIC: Waiting for more frames (%zu/%zu)", 
                                cloud_queue_.size(), frame_gap_ + 1);
            return false;
        }
        
        auto current_cloud = cloud_queue_.back();
        auto it = cloud_queue_.begin();
        std::advance(it, cloud_queue_.size() - frame_gap_ - 1);
        auto reference_cloud = *it;
        
        if (!current_cloud || current_cloud->empty() || 
            !reference_cloud || reference_cloud->empty()) {
            return false;
        }
        
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(reference_cloud);
        
        DynamicDetectorParams params = dynamic_detector_.getParams();
        std::vector<ClusterPoint> dynamic_points;
        
        for (const auto& pt : current_cloud->points) {
            if (pt.z < params.min_z) {
                continue;
            }
            
            std::vector<int> indices;
            std::vector<float> distances;
            
            if (kdtree.radiusSearch(pt, params.distance_threshold, indices, distances) == 0) {
                dynamic_points.emplace_back(pt.x, pt.y, pt.z);
            }
        }
        
        NODELET_INFO_THROTTLE(1, "DYNAMIC: current=%zu, ref=%zu, dynamic=%zu, threshold=%.2f", 
                            current_cloud->size(), reference_cloud->size(), 
                            dynamic_points.size(), params.distance_threshold);
        
        if (dynamic_points.size() < static_cast<size_t>(minPts_)) {
            return false;
        }
        
        DBSCAN dbscan(dynamic_points, eps_, minPts_);
        int num_clusters = dbscan.run();
        
        if (num_clusters == 0) {
            return false;
        }
        
        NODELET_INFO("DYNAMIC: Found %d clusters from %zu dynamic points", 
                    num_clusters, dynamic_points.size());
        detected_cluster_ = cluster_filter_.findBest(dbscan.getClusters(), 
                                                    cur_detected_bbox_, 
                                                    cur_centroid_);
        
        return (detected_cluster_ && !detected_cluster_->empty());
    }

    bool LidarDetector::detectDirect()
    {
        if (!accumulated_cloud_ || accumulated_cloud_->empty()) {
            return false;
        }
        
        DynamicDetectorParams params = dynamic_detector_.getParams();
        
        std::vector<ClusterPoint> airspace_points;
        airspace_points.reserve(accumulated_cloud_->size());
        
        for (const auto& pt : accumulated_cloud_->points) {
            if (pt.z >= params.min_z) {
                airspace_points.emplace_back(pt.x, pt.y, pt.z);
            }
        }
        
        NODELET_INFO_THROTTLE(1, "DIRECT: accumulated=%zu, airspace=%zu (min_z=%.1f)", 
                            accumulated_cloud_->size(), airspace_points.size(), params.min_z);
        
        if (airspace_points.size() < static_cast<size_t>(minPts_)) {
            return false;
        }
        
        DBSCAN dbscan(airspace_points, eps_, minPts_);
        int num_clusters = dbscan.run();
        
        if (num_clusters == 0) {
            return false;
        }
        
        NODELET_INFO("DIRECT: Found %d clusters from %zu airspace points", 
                    num_clusters, airspace_points.size());
        detected_cluster_ = cluster_filter_.findBest(dbscan.getClusters(), 
                                                    cur_detected_bbox_, 
                                                    cur_centroid_);
        
        return (detected_cluster_ && !detected_cluster_->empty());
    }

    void LidarDetector::updateTracking()
    {
        // Update tracking state based on current detection
        if (first_cluster_) {
            track_min_pt_ = pcl::PointXYZ(cur_detected_bbox_.min_pt.x(), 
                                        cur_detected_bbox_.min_pt.y(), 
                                        cur_detected_bbox_.min_pt.z());
            track_max_pt_ = pcl::PointXYZ(cur_detected_bbox_.max_pt.x(), 
                                        cur_detected_bbox_.max_pt.y(), 
                                        cur_detected_bbox_.max_pt.z());
            first_cluster_ = false;
            has_tracked_ = true;
        } else {
            // Update tracker bounds to encompass all detections
            track_min_pt_.x = std::min(track_min_pt_.x, cur_detected_bbox_.min_pt.x());
            track_min_pt_.y = std::min(track_min_pt_.y, cur_detected_bbox_.min_pt.y());
            track_min_pt_.z = std::min(track_min_pt_.z, cur_detected_bbox_.min_pt.z());
            track_max_pt_.x = std::max(track_max_pt_.x, cur_detected_bbox_.max_pt.x());
            track_max_pt_.y = std::max(track_max_pt_.y, cur_detected_bbox_.max_pt.y());
            track_max_pt_.z = std::max(track_max_pt_.z, cur_detected_bbox_.max_pt.z());
        }
        expand_lock_ = true;
        
        NODELET_INFO("Detection: centroid=(%.2f, %.2f, %.2f), volume=%.3f",
                    cur_centroid_.x(), cur_centroid_.y(), cur_centroid_.z(),
                    cur_detected_bbox_.volume());
    }

    void LidarDetector::publishResults()
    {
        ros::Time now = ros::Time::now();            
            if (detected_cluster_ && !detected_cluster_->empty()) {
            visualizer_.publishCloud(detected_cluster_, frame_id_, now, drone_cloud_pub_);
            
            auto marker = Visualizer::createAABBMarker(
                cur_detected_bbox_, 0, "detected_cluster", frame_id_,
                Eigen::Vector4d(1.0, 0.0, 0.0, 0.5));
            marker_pub_.publish(marker);
            
            auto tracker_marker = Visualizer::createAABBMarker(
                track_min_pt_, track_max_pt_, -1, "tracker", frame_id_,
                Eigen::Vector4d(0.0, 0.0, 1.0, 0.3));
            marker_pub_.publish(tracker_marker);
            
            auto centroid_marker = Visualizer::createCentroidMarker(
                cur_centroid_, 100, frame_id_,
                Eigen::Vector4d(1.0, 1.0, 0.0, 1.0), 0.2);
            marker_pub_.publish(centroid_marker);
            
            rm_radar_msgs::DroneDetection detection_msg;
            detection_msg.header.stamp = now;
            detection_msg.header.frame_id = frame_id_;
            detection_msg.id = 255;
            detection_msg.confidence = 0.8;
            detection_msg.distance_to_image_center = -1.0;
            
            detection_msg.pose.position.x = cur_centroid_.x();
            detection_msg.pose.position.y = cur_centroid_.y();
            detection_msg.pose.position.z = cur_centroid_.z();
            detection_msg.pose.orientation.w = 1.0;
            detection_msg.pose.orientation.x = 0.0;
            detection_msg.pose.orientation.y = 0.0;
            detection_msg.pose.orientation.z = 0.0;
            
            detection_msg.centroid.x = cur_centroid_.x();
            detection_msg.centroid.y = cur_centroid_.y();
            detection_msg.centroid.z = cur_centroid_.z();
            
            detection_pub_.publish(detection_msg);
            
            NODELET_DEBUG("Published detection: centroid=(%.2f, %.2f, %.2f)",
                        cur_centroid_.x(), cur_centroid_.y(), cur_centroid_.z());
        }
    }

}  // namespace rm_radar_lidar_detector
