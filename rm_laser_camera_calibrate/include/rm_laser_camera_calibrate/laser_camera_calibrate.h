#pragma once

#include <fstream>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include <ros/ros.h>
#include <rm_radar_msgs/LaserCameraCalib.h>

namespace rm_laser_camera_calibrate
{
class LaserCameraCalibrator
{
public:
  struct RuntimeConfig
  {
    bool enabled{false};
    bool enable_sampling{false};
    double h_px_min{4.0};
    double h_px_max{300.0};
    double min_sample_interval{0.05};
    double min_h_gap_px{2.0};
    bool enforce_h_bin_sampling{true};
    double h_bin_size_px{6.0};
    int max_samples_per_h_bin{60};
    int min_fit_samples{30};
    int min_fit_groups{8};
    double max_fit_rmse_x{2.5};
    double max_fit_rmse_y{2.5};
    bool flush_each_sample{false};
  };

  LaserCameraCalibrator() = default;
  ~LaserCameraCalibrator();

  bool init(const ros::NodeHandle& nh);
  void setEnableFlags(bool enabled, bool enable_sampling);
  void setRuntimeConfig(const RuntimeConfig& cfg);
  void updateArmorLength(const rm_radar_msgs::LaserCameraCalib& msg);
  bool recordManualBias(double bias_x, double bias_y, const ros::Time& stamp);
  void fillStatusMsg(rm_radar_msgs::LaserCameraCalib& msg, const ros::Time& stamp) const;

private:
  bool openCsvIfNeeded();
  void updateFitMetrics();
  bool inSamplingRange(double h_px) const;
  void rebuildHBinCounts();

private:
  bool enabled_{false};
  bool enable_sampling_{false};

  double h_px_min_{4.0};
  double h_px_max_{300.0};

  std::string csv_path_{"/tmp/laser_camera_samples.csv"};
  bool write_csv_header_{true};
  bool flush_each_sample_{false};
  double min_sample_interval_{0.05};
  bool enforce_h_bin_sampling_{true};
  double h_bin_size_px_{6.0};
  int max_samples_per_h_bin_{60};
  double min_h_gap_px_{2.0};

  ros::Time last_sample_time_;
  bool has_last_sample_h_{false};
  double last_sample_h_px_{0.0};
  std::unordered_map<int, int> h_bin_counts_;

  bool has_latest_h_{false};
  double latest_h_px_{0.0};
  double latest_armor_confidence_{0.0};

  std::vector<double> sample_inv_h_;
  std::vector<double> sample_bias_x_;
  std::vector<double> sample_bias_y_;
  int min_fit_samples_{30};
  int min_fit_groups_{8};
  double max_fit_rmse_x_{2.5};
  double max_fit_rmse_y_{2.5};
  std::array<double, 3> fit_coeff_x_{{ 0.0, 0.0, 0.0 }};
  std::array<double, 3> fit_coeff_y_{{ 0.0, 0.0, 0.0 }};
  bool fit_valid_{false};
  double fit_rmse_x_{0.0};
  double fit_rmse_y_{0.0};
  bool fit_ready_{false};

  std::ofstream csv_;
};
}  // namespace rm_laser_camera_calibrate
