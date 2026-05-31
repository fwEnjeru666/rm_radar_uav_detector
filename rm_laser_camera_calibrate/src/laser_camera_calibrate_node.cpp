#include <ros/ros.h>

#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/Vector3.h>
#include <memory>
#include <rm_laser_camera_calibrate/LaserCameraCalibrateConfig.h>
#include <rm_laser_camera_calibrate/laser_camera_calibrate.h>
#include <rm_radar_msgs/LaserCameraCalib.h>

namespace rm_laser_camera_calibrate
{
class LaserCameraCalibrateNode
{
public:
  LaserCameraCalibrateNode()
    : pnh_("~"), calibrate_nh_(pnh_, "laser_camera_calibrate")
  {
    calibrator_.init(calibrate_nh_);
    calib_cfg_srv_ = std::make_unique<dynamic_reconfigure::Server<rm_laser_camera_calibrate::LaserCameraCalibrateConfig>>(calibrate_nh_);
    calib_cfg_cb_ = [this](rm_laser_camera_calibrate::LaserCameraCalibrateConfig& config, uint32_t level)
    {
      dynamicConfigCB(config, level);
    };
    calib_cfg_srv_->setCallback(calib_cfg_cb_);

    calibrate_nh_.param("sample_topic", sample_topic_, std::string("/processor/laser_calib_extract"));
    calibrate_nh_.param("manual_bias_topic", manual_bias_topic_, std::string("/laser_camera_calibrate/manual_bias"));
    calibrate_nh_.param("status_topic", status_topic_, std::string("/laser_camera_calibrate/status"));
    sample_sub_ = pnh_.subscribe(sample_topic_, 100, &LaserCameraCalibrateNode::sampleCallback, this);
    manual_bias_sub_ = pnh_.subscribe(manual_bias_topic_, 20, &LaserCameraCalibrateNode::manualBiasCallback, this);
    status_pub_ = pnh_.advertise<rm_radar_msgs::LaserCameraCalib>(status_topic_, 20);

    ROS_INFO("[LaserCameraCalibNode] sample_topic=%s manual_bias_topic=%s status_topic=%s",
             sample_topic_.c_str(),
             manual_bias_topic_.c_str(),
             status_topic_.c_str());
  }

private:
  void publishStatus(const ros::Time& stamp)
  {
    rm_radar_msgs::LaserCameraCalib status_msg;
    calibrator_.fillStatusMsg(status_msg, stamp);
    status_pub_.publish(status_msg);
  }

  void sampleCallback(const rm_radar_msgs::LaserCameraCalib::ConstPtr& msg)
  {
    calibrator_.updateArmorLength(*msg);
    publishStatus(msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp);
  }

  void manualBiasCallback(const geometry_msgs::Vector3::ConstPtr& msg)
  {
    const bool ok = calibrator_.recordManualBias(msg->x, msg->y, ros::Time::now());
    if (ok)
    {
      ROS_INFO_THROTTLE(0.5, "[LaserCameraCalibNode] manual sample accepted");
    }
    publishStatus(ros::Time::now());
  }

  void dynamicConfigCB(rm_laser_camera_calibrate::LaserCameraCalibrateConfig& config, uint32_t /*level*/)
  {
    LaserCameraCalibrator::RuntimeConfig runtime_cfg;
    runtime_cfg.enabled = config.enable;
    runtime_cfg.enable_sampling = config.enable_sampling;
    runtime_cfg.h_px_min = config.h_px_min;
    runtime_cfg.h_px_max = config.h_px_max;
    runtime_cfg.min_sample_interval = config.min_sample_interval;
    runtime_cfg.min_h_gap_px = config.min_h_gap_px;
    runtime_cfg.enforce_h_bin_sampling = config.enforce_h_bin_sampling;
    runtime_cfg.h_bin_size_px = config.h_bin_size_px;
    runtime_cfg.max_samples_per_h_bin = config.max_samples_per_h_bin;
    runtime_cfg.min_fit_samples = config.min_fit_samples;
    runtime_cfg.min_fit_groups = config.min_fit_groups;
    runtime_cfg.max_fit_rmse_x = config.max_fit_rmse_x;
    runtime_cfg.max_fit_rmse_y = config.max_fit_rmse_y;
    runtime_cfg.flush_each_sample = config.flush_each_sample;
    calibrator_.setRuntimeConfig(runtime_cfg);
  }

private:
  ros::NodeHandle pnh_;
  ros::NodeHandle calibrate_nh_;
  std::unique_ptr<dynamic_reconfigure::Server<rm_laser_camera_calibrate::LaserCameraCalibrateConfig>> calib_cfg_srv_;
  dynamic_reconfigure::Server<rm_laser_camera_calibrate::LaserCameraCalibrateConfig>::CallbackType calib_cfg_cb_;
  std::string sample_topic_;
  std::string manual_bias_topic_;
  std::string status_topic_;
  ros::Subscriber sample_sub_;
  ros::Subscriber manual_bias_sub_;
  ros::Publisher status_pub_;
  LaserCameraCalibrator calibrator_;
};
}  // namespace rm_laser_camera_calibrate

int main(int argc, char** argv)
{
  ros::init(argc, argv, "laser_camera_calibrate_node");
  rm_laser_camera_calibrate::LaserCameraCalibrateNode node;
  ros::spin();
  return 0;
}
