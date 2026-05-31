#include <rm_laser_camera_calibrate/laser_camera_calibrate.h>

#include <algorithm>
#include <cmath>

namespace
{
bool solve3x3(double m[3][3], double b[3], double x[3])
{
  for (int i = 0; i < 3; ++i)
  {
    int pivot = i;
    for (int r = i + 1; r < 3; ++r)
    {
      if (std::abs(m[r][i]) > std::abs(m[pivot][i]))
      {
        pivot = r;
      }
    }
    if (std::abs(m[pivot][i]) < 1e-12)
    {
      return false;
    }
    if (pivot != i)
    {
      for (int c = i; c < 3; ++c)
      {
        std::swap(m[i][c], m[pivot][c]);
      }
      std::swap(b[i], b[pivot]);
    }
    const double diag = m[i][i];
    for (int c = i; c < 3; ++c)
    {
      m[i][c] /= diag;
    }
    b[i] /= diag;
    for (int r = 0; r < 3; ++r)
    {
      if (r == i)
      {
        continue;
      }
      const double factor = m[r][i];
      for (int c = i; c < 3; ++c)
      {
        m[r][c] -= factor * m[i][c];
      }
      b[r] -= factor * b[i];
    }
  }
  x[0] = b[0];
  x[1] = b[1];
  x[2] = b[2];
  return std::isfinite(x[0]) && std::isfinite(x[1]) && std::isfinite(x[2]);
}
}  // namespace

namespace rm_laser_camera_calibrate
{
LaserCameraCalibrator::~LaserCameraCalibrator()
{
  if (csv_.is_open())
  {
    csv_.close();
  }
}

bool LaserCameraCalibrator::init(const ros::NodeHandle& nh)
{
  nh.param("enable", enabled_, false);
  nh.param("enable_sampling", enable_sampling_, false);

  nh.param("h_px_min", h_px_min_, h_px_min_);
  nh.param("h_px_max", h_px_max_, h_px_max_);

  nh.param("csv_path", csv_path_, csv_path_);
  nh.param("write_csv_header", write_csv_header_, write_csv_header_);
  nh.param("flush_each_sample", flush_each_sample_, flush_each_sample_);
  nh.param("min_sample_interval", min_sample_interval_, min_sample_interval_);
  nh.param("enforce_h_bin_sampling", enforce_h_bin_sampling_, enforce_h_bin_sampling_);
  nh.param("h_bin_size_px", h_bin_size_px_, h_bin_size_px_);
  nh.param("max_samples_per_h_bin", max_samples_per_h_bin_, max_samples_per_h_bin_);
  nh.param("min_h_gap_px", min_h_gap_px_, min_h_gap_px_);
  nh.param("min_fit_samples", min_fit_samples_, min_fit_samples_);
  nh.param("min_fit_groups", min_fit_groups_, min_fit_groups_);
  nh.param("max_fit_rmse_x", max_fit_rmse_x_, max_fit_rmse_x_);
  nh.param("max_fit_rmse_y", max_fit_rmse_y_, max_fit_rmse_y_);

  ROS_INFO("[LaserCameraCalib] enable=%d sampling=%d csv=%s",
           static_cast<int>(enabled_),
           static_cast<int>(enable_sampling_),
           csv_path_.c_str());
  return true;
}

void LaserCameraCalibrator::setEnableFlags(bool enabled, bool enable_sampling)
{
  if (enabled_ == enabled && enable_sampling_ == enable_sampling)
  {
    return;
  }
  enabled_ = enabled;
  enable_sampling_ = enable_sampling;
  ROS_INFO("[LaserCameraCalib] runtime update: enable=%d sampling=%d",
           static_cast<int>(enabled_),
           static_cast<int>(enable_sampling_));
}

void LaserCameraCalibrator::setRuntimeConfig(const RuntimeConfig& cfg)
{
  const bool old_enabled = enabled_;
  const bool old_sampling = enable_sampling_;
  const double old_bin_size = h_bin_size_px_;
  const bool old_enforce_h_bin_sampling = enforce_h_bin_sampling_;
  const int old_min_fit_samples = min_fit_samples_;
  const int old_min_fit_groups = min_fit_groups_;
  const double old_max_fit_rmse_x = max_fit_rmse_x_;
  const double old_max_fit_rmse_y = max_fit_rmse_y_;

  enabled_ = cfg.enabled;
  enable_sampling_ = cfg.enable_sampling;
  h_px_min_ = std::max(1e-6, cfg.h_px_min);
  h_px_max_ = std::max(h_px_min_, cfg.h_px_max);
  min_sample_interval_ = std::max(0.0, cfg.min_sample_interval);
  min_h_gap_px_ = std::max(0.0, cfg.min_h_gap_px);
  enforce_h_bin_sampling_ = cfg.enforce_h_bin_sampling;
  h_bin_size_px_ = std::max(1e-6, cfg.h_bin_size_px);
  max_samples_per_h_bin_ = std::max(1, cfg.max_samples_per_h_bin);
  min_fit_samples_ = std::max(3, cfg.min_fit_samples);
  min_fit_groups_ = std::max(1, cfg.min_fit_groups);
  max_fit_rmse_x_ = std::max(1e-6, cfg.max_fit_rmse_x);
  max_fit_rmse_y_ = std::max(1e-6, cfg.max_fit_rmse_y);
  flush_each_sample_ = cfg.flush_each_sample;

  if (std::abs(old_bin_size - h_bin_size_px_) > 1e-9 || old_enforce_h_bin_sampling != enforce_h_bin_sampling_)
  {
    rebuildHBinCounts();
  }

  if (old_min_fit_samples != min_fit_samples_ ||
      old_min_fit_groups != min_fit_groups_ ||
      std::abs(old_max_fit_rmse_x - max_fit_rmse_x_) > 1e-9 ||
      std::abs(old_max_fit_rmse_y - max_fit_rmse_y_) > 1e-9)
  {
    updateFitMetrics();
  }

  if (old_enabled != enabled_ || old_sampling != enable_sampling_)
  {
    ROS_INFO("[LaserCameraCalib] runtime update: enable=%d sampling=%d",
             static_cast<int>(enabled_),
             static_cast<int>(enable_sampling_));
  }
}

bool LaserCameraCalibrator::inSamplingRange(double h_px) const
{
  return std::isfinite(h_px) && h_px >= h_px_min_ && h_px <= h_px_max_;
}

void LaserCameraCalibrator::rebuildHBinCounts()
{
  h_bin_counts_.clear();
  const double bin_size = std::max(1e-6, h_bin_size_px_);
  for (double inv_h : sample_inv_h_)
  {
    if (!std::isfinite(inv_h) || std::abs(inv_h) < 1e-9)
    {
      continue;
    }
    const double h_px = 1.0 / inv_h;
    if (!std::isfinite(h_px) || h_px <= 0.0)
    {
      continue;
    }
    const int h_bin = static_cast<int>(std::floor(h_px / bin_size));
    h_bin_counts_[h_bin] += 1;
  }
}

bool LaserCameraCalibrator::openCsvIfNeeded()
{
  if (csv_.is_open())
  {
    return true;
  }

  bool file_has_content = false;
  {
    std::ifstream ifs(csv_path_, std::ios::in);
    if (ifs.good())
    {
      ifs.seekg(0, std::ios::end);
      file_has_content = (ifs.tellg() > 0);
    }
  }

  csv_.open(csv_path_, std::ios::out | std::ios::app);
  if (!csv_.is_open())
  {
    ROS_ERROR_THROTTLE(1.0, "[LaserCameraCalib] failed to open csv: %s", csv_path_.c_str());
    return false;
  }

  if (write_csv_header_ && !file_has_content)
  {
    csv_ << "stamp,h_px,inv_h,bias_x,bias_y,armor_confidence\n";
    write_csv_header_ = false;
  }
  return true;
}

void LaserCameraCalibrator::updateArmorLength(const rm_radar_msgs::LaserCameraCalib& msg)
{
  if (!msg.armor_valid)
  {
    has_latest_h_ = false;
    return;
  }

  if (!std::isfinite(msg.armor_h_px))
  {
    has_latest_h_ = false;
    return;
  }

  latest_h_px_ = msg.armor_h_px;
  latest_armor_confidence_ = msg.armor_confidence;
  has_latest_h_ = true;
}

void LaserCameraCalibrator::updateFitMetrics()
{
  fit_valid_ = false;
  fit_ready_ = false;
  fit_rmse_x_ = 0.0;
  fit_rmse_y_ = 0.0;
  fit_coeff_x_ = { 0.0, 0.0, 0.0 };
  fit_coeff_y_ = { 0.0, 0.0, 0.0 };

  const int n = static_cast<int>(sample_inv_h_.size());
  if (n < 3)
  {
    return;
  }

  double s0 = 0.0;
  double s1 = 0.0;
  double s2 = 0.0;
  double s3 = 0.0;
  double s4 = 0.0;
  double bx0 = 0.0;
  double bx1 = 0.0;
  double bx2 = 0.0;
  double by0 = 0.0;
  double by1 = 0.0;
  double by2 = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const double inv_h = sample_inv_h_[static_cast<size_t>(i)];
    const double inv_h2 = inv_h * inv_h;
    const double inv_h3 = inv_h2 * inv_h;
    const double inv_h4 = inv_h2 * inv_h2;
    const double yx = sample_bias_x_[static_cast<size_t>(i)];
    const double yy = sample_bias_y_[static_cast<size_t>(i)];

    s0 += 1.0;
    s1 += inv_h;
    s2 += inv_h2;
    s3 += inv_h3;
    s4 += inv_h4;

    bx0 += yx;
    bx1 += yx * inv_h;
    bx2 += yx * inv_h2;

    by0 += yy;
    by1 += yy * inv_h;
    by2 += yy * inv_h2;
  }

  double mx[3][3] = { { s0, s1, s2 }, { s1, s2, s3 }, { s2, s3, s4 } };
  double my[3][3] = { { s0, s1, s2 }, { s1, s2, s3 }, { s2, s3, s4 } };
  double vx[3] = { bx0, bx1, bx2 };
  double vy[3] = { by0, by1, by2 };
  double coeff_x[3] = { 0.0, 0.0, 0.0 };
  double coeff_y[3] = { 0.0, 0.0, 0.0 };
  if (!solve3x3(mx, vx, coeff_x) || !solve3x3(my, vy, coeff_y))
  {
    return;
  }

  double se_x = 0.0;
  double se_y = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const double inv_h = sample_inv_h_[static_cast<size_t>(i)];
    const double inv_h2 = inv_h * inv_h;
    const double pred_x = coeff_x[0] + coeff_x[1] * inv_h + coeff_x[2] * inv_h2;
    const double pred_y = coeff_y[0] + coeff_y[1] * inv_h + coeff_y[2] * inv_h2;
    const double dx = sample_bias_x_[static_cast<size_t>(i)] - pred_x;
    const double dy = sample_bias_y_[static_cast<size_t>(i)] - pred_y;
    se_x += dx * dx;
    se_y += dy * dy;
  }
  const double mse_x = se_x / std::max(1, n);
  const double mse_y = se_y / std::max(1, n);
  if (!std::isfinite(mse_x) || !std::isfinite(mse_y))
  {
    return;
  }

  fit_rmse_x_ = std::sqrt(std::max(0.0, mse_x));
  fit_rmse_y_ = std::sqrt(std::max(0.0, mse_y));
  fit_coeff_x_ = { coeff_x[0], coeff_x[1], coeff_x[2] };
  fit_coeff_y_ = { coeff_y[0], coeff_y[1], coeff_y[2] };
  fit_valid_ = std::isfinite(fit_rmse_x_) && std::isfinite(fit_rmse_y_);
  const bool enough_samples = n >= std::max(3, min_fit_samples_);
  const bool enough_groups = static_cast<int>(h_bin_counts_.size()) >= std::max(1, min_fit_groups_);
  const bool rmse_ok = fit_rmse_x_ <= max_fit_rmse_x_ && fit_rmse_y_ <= max_fit_rmse_y_;
  fit_ready_ = fit_valid_ && enough_samples && enough_groups && rmse_ok;
}

bool LaserCameraCalibrator::recordManualBias(double bias_x, double bias_y, const ros::Time& stamp)
{
  if (!(enabled_ && enable_sampling_))
  {
    return false;
  }

  if (!has_latest_h_)
  {
    return false;
  }

  if (!std::isfinite(bias_x) || !std::isfinite(bias_y))
  {
    return false;
  }

  if (!last_sample_time_.isZero() && (stamp - last_sample_time_).toSec() < min_sample_interval_)
  {
    return false;
  }

  const double h_px = latest_h_px_;
  if (!inSamplingRange(h_px))
  {
    return false;
  }

  if (has_last_sample_h_ && std::abs(h_px - last_sample_h_px_) < std::max(0.0, min_h_gap_px_))
  {
    return false;
  }

  const double bin_size = std::max(1e-6, h_bin_size_px_);
  const int h_bin = static_cast<int>(std::floor(h_px / bin_size));
  if (enforce_h_bin_sampling_)
  {
    const auto it = h_bin_counts_.find(h_bin);
    if (it != h_bin_counts_.end() && it->second >= std::max(1, max_samples_per_h_bin_))
    {
      return false;
    }
  }

  if (!openCsvIfNeeded())
  {
    return false;
  }

  const double inv_h = 1.0 / std::max(h_px, 1e-6);
  csv_ << stamp.toSec() << ','
       << h_px << ','
       << inv_h << ','
       << bias_x << ','
       << bias_y << ','
       << latest_armor_confidence_ << '\n';

  if (flush_each_sample_)
  {
    csv_.flush();
  }

  has_last_sample_h_ = true;
  last_sample_h_px_ = h_px;
  last_sample_time_ = stamp;
  h_bin_counts_[h_bin] += 1;

  sample_inv_h_.push_back(inv_h);
  sample_bias_x_.push_back(bias_x);
  sample_bias_y_.push_back(bias_y);
  updateFitMetrics();
  return true;
}

void LaserCameraCalibrator::fillStatusMsg(rm_radar_msgs::LaserCameraCalib& msg, const ros::Time& stamp) const
{
  msg.header.stamp = stamp;
  msg.armor_valid = has_latest_h_;
  msg.armor_confidence = latest_armor_confidence_;
  msg.armor_h_px = latest_h_px_;

  msg.bias_valid = false;
  msg.bias_x = 0.0;
  msg.bias_y = 0.0;

  msg.status_valid = true;
  msg.h_meets_sampling = has_latest_h_ && inSamplingRange(latest_h_px_);
  msg.sample_count = static_cast<int32_t>(sample_inv_h_.size());
  msg.group_count = static_cast<int32_t>(h_bin_counts_.size());
  msg.fit_valid = fit_valid_;
  msg.fit_rmse_x = fit_rmse_x_;
  msg.fit_rmse_y = fit_rmse_y_;
  msg.calib_ready = fit_ready_;
  msg.fit_bias_valid = false;
  msg.fit_bias_x = 0.0;
  msg.fit_bias_y = 0.0;
  if (fit_ready_ && has_latest_h_ && inSamplingRange(latest_h_px_))
  {
    const double inv_h = 1.0 / std::max(latest_h_px_, 1e-6);
    const double inv_h2 = inv_h * inv_h;
    msg.fit_bias_x = fit_coeff_x_[0] + fit_coeff_x_[1] * inv_h + fit_coeff_x_[2] * inv_h2;
    msg.fit_bias_y = fit_coeff_y_[0] + fit_coeff_y_[1] * inv_h + fit_coeff_y_[2] * inv_h2;
    msg.fit_bias_valid = std::isfinite(msg.fit_bias_x) && std::isfinite(msg.fit_bias_y);
  }
}
}  // namespace rm_laser_camera_calibrate
