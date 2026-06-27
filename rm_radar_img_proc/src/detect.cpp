#include <detect.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <sys/stat.h>

#include <ros/ros.h>

namespace rm_radarplugin
{
    namespace
    {
        std::atomic<unsigned long long> g_yolo_sample_counter{0};

        bool isDirectory(const std::string& path)
        {
            struct stat buffer;
            return !path.empty() && stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode);
        }

        std::string dirname(const std::string& path)
        {
            const auto pos = path.find_last_of("/");
            if (pos == std::string::npos)
            {
                return ".";
            }
            if (pos == 0)
            {
                return "/";
            }
            return path.substr(0, pos);
        }

        bool ensureDirectory(const std::string& path)
        {
            if (path.empty())
            {
                return false;
            }
            if (isDirectory(path))
            {
                return true;
            }

            const std::string parent = dirname(path);
            if (!parent.empty() && parent != path && !isDirectory(parent) && !ensureDirectory(parent))
            {
                return false;
            }

            return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
        }

        cv::Rect clipRect(const cv::Rect& rect, const cv::Size& image_size)
        {
            return rect & cv::Rect(0, 0, image_size.width, image_size.height);
        }

        std::pair<double, double> calcBarColorScores(const Bar& bar, const cv::Mat& bgr_img)
        {
            const cv::Rect image_rect(0, 0, bgr_img.cols, bgr_img.rows);
            const cv::Rect roi_box = bar.bar_rect_.boundingRect() & image_rect;
            if (roi_box.width <= 1 || roi_box.height <= 1)
            {
                return {0.0, 0.0};
            }

            cv::Mat mask = cv::Mat::zeros(roi_box.size(), CV_8UC1);
            std::vector<cv::Point> polygon;
            polygon.reserve(4);
            for (const auto& point : bar.points_)
            {
                polygon.emplace_back(cvRound(point.x) - roi_box.x, cvRound(point.y) - roi_box.y);
            }
            cv::fillConvexPoly(mask, polygon, cv::Scalar(255));

            std::vector<cv::Mat> channels;
            cv::split(bgr_img(roi_box), channels);
            if (channels.size() != 3)
            {
                return {0.0, 0.0};
            }

            cv::Mat red_diff;
            cv::Mat blue_diff;
            cv::subtract(channels[2], channels[1], red_diff);
            cv::subtract(channels[0], channels[1], blue_diff);

            cv::Mat masked_red;
            cv::Mat masked_blue;
            red_diff.copyTo(masked_red, mask);
            blue_diff.copyTo(masked_blue, mask);
            return {cv::sum(masked_red)[0], cv::sum(masked_blue)[0]};
        }
    }  // namespace

    void DetectCore::updateGrayImage(const cv::Mat& image)
    {
        if (image.empty())
        {
            gray_image_.release();
            return;
        }

        if (image.channels() == 3)
        {
            cv::cvtColor(image, gray_image_, cv::COLOR_BGR2GRAY);
        }
        else
        {
            image.copyTo(gray_image_);
        }
    }

    void DetectCore::updateGrayImage(const cv::UMat& image)
    {
        if (image.empty())
        {
            gray_image_.release();
            return;
        }

        if (image.channels() == 3)
        {
            cv::Mat image_mat = image.getMat(cv::ACCESS_READ);
            cv::cvtColor(image_mat, gray_image_, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray_image_ = image.getMat(cv::ACCESS_READ).clone();
        }
    }

    bool DetectCore::shouldSaveYoloFailedSample() const
    {
        return detect_options_.save_yolo_samples;
    }

    bool DetectCore::shouldSaveYoloUncertainSample(float confidence) const
    {
        constexpr float uncertain_confidence_threshold = 0.6f;
        return detect_options_.save_yolo_samples && confidence <= uncertain_confidence_threshold;
    }

    void DetectCore::saveYoloSample(const cv::Mat& image, const std::string& reason, float confidence,
                                    const cv::Rect* roi_box) const
    {
        if (image.empty() || detect_options_.model_config_path.empty())
        {
            return;
        }

        const std::string reason_dir = dirname(detect_options_.model_config_path) + "/samples/" + reason;
        if (!ensureDirectory(reason_dir))
        {
            ROS_WARN_THROTTLE(2, "[DetectCore] Failed to create YOLO sample dir: %s", reason_dir.c_str());
            return;
        }

        std::ostringstream name_builder;
        name_builder << ros::WallTime::now().toNSec() << "_" << g_yolo_sample_counter.fetch_add(1)
                     << "_c" << static_cast<int>(std::round(confidence * 1000.0f));
        const std::string base_path = reason_dir + "/" + name_builder.str();

        cv::imwrite(base_path + "_full.jpg", image);
        if (roi_box && roi_box->width > 1 && roi_box->height > 1)
        {
            const cv::Rect clipped_roi = clipRect(*roi_box, image.size());
            if (clipped_roi.width > 1 && clipped_roi.height > 1)
            {
                cv::imwrite(base_path + "_roi.jpg", image(clipped_roi));
            }
        }
    }

    bool DetectCore::targetFilter(int class_id) const
    {
        if (!detect_options_.enable_target_filter)
        {
            return true;
        }

        switch (class_id)
        {
            case -1:
                return true;
            case 0:
                return true;
            case 1:
                return detect_options_.target_is_red;
            case 2:
                return !detect_options_.target_is_red;
            case 3:
                return true;
            default:
                return false;
        }
    }

    bool DetectCore::targetFilter(const Armor& armor, const cv::Mat& bgr_img) const
    {
        if (!detect_options_.enable_target_filter)
        {
            return true;
        }

        if (bgr_img.empty())
        {
            return false;
        }

        if (!armor.bar_up_ || !armor.bar_bottom_)
        {
            return true;
        }

        constexpr double min_color_score = 800.0;
        constexpr double color_dominance_ratio = 1.25;

        const Bar* selected_bar = armor.bar_up_;
        if (armor.bar_bottom_->length_len_ > armor.bar_up_->length_len_)
        {
            selected_bar = armor.bar_bottom_;
        }

        const std::pair<double, double> color_scores = calcBarColorScores(*selected_bar, bgr_img);
        const bool looks_red = color_scores.first > min_color_score &&
                               color_scores.first > color_scores.second * color_dominance_ratio;
        const bool looks_blue = color_scores.second > min_color_score &&
                                color_scores.second > color_scores.first * color_dominance_ratio;

        if (detect_options_.target_is_red)
        {
            return !looks_blue;
        }
        return !looks_red;
    }

    DetectCore::DetectCore(const DetectOptions& options)
    : detect_options_(options), object_options_(options.object_options), debugger_(options.object_options)
    {
        if (detect_options_.detect_method == DetectMethod::YOLO && detect_options_.model_config_path.empty())
        {
            ROS_WARN("[DetectCore] YOLO detect method selected but model_config_path is empty.");
        }
        if (detect_options_.detect_method == DetectMethod::YOLO && !detect_options_.model_config_path.empty())
        {
            yolo_core_ = std::make_unique<YoloCore>(detect_options_.model_config_path, object_options_.armor_enabled);
        }
    }

    void DetectCore::setOptions(const DetectOptions& options)
    {
        detect_options_ = options;
        object_options_ = options.object_options;
        debugger_.setOptions(object_options_);
    }

    void DetectCore::reset()
    {
        bars_.clear();
        best_armor_.reset();
        debug_armors_.clear();
    }

    void DetectCore::resetBestArmor()
    {
        best_armor_.reset();
    }

    void DetectCore::detect(const cv::Mat& raw_image)
    {
        reset();
        if (!yolo_core_)
        {
            ROS_WARN_THROTTLE(2, "[DetectCore] YOLO core is not initialized.");
            return;
        }

        if (detect_options_.yolo_refine_enabled)
        {
            updateGrayImage(raw_image);
        }
        YoloDetectionOutput detection;
        if (!yolo_core_->detect(raw_image, detection))
        {
            if (detect_options_.process_debug)
            {
                ROS_INFO_THROTTLE(1.0, "[t=%.3f][DetectCore][YOLO] No detections from model output.",
                                  ros::Time::now().toSec());
            }
            if (shouldSaveYoloFailedSample())
            {
                saveYoloSample(raw_image, "no_detection", 0.0f);
            }
            return;
        }

        if (detect_options_.process_debug)
        {
            ROS_INFO_THROTTLE(1.0, "[t=%.3f][DetectCore][YOLO] Model produced 1 detection.",
                              ros::Time::now().toSec());
        }

        const cv::Rect detection_roi(cvRound(detection.bbox.x), cvRound(detection.bbox.y),
                                     cvRound(detection.bbox.width), cvRound(detection.bbox.height));
        if (shouldSaveYoloUncertainSample(detection.confidence))
        {
            saveYoloSample(raw_image, "uncertain", detection.confidence, &detection_roi);
        }

        if (!targetFilter(detection.class_id))
        {
            if (detect_options_.process_debug)
            {
                ROS_INFO_THROTTLE(1.0,
                                  "[t=%.3f][DetectCore][YOLO] Detection filtered by class. class_id=%d conf=%.3f",
                                  ros::Time::now().toSec(), detection.class_id, detection.confidence);
            }
            if (shouldSaveYoloFailedSample())
            {
                saveYoloSample(raw_image, "class_filtered", detection.confidence, &detection_roi);
            }
        }
        else
        {
            auto armor = buildArmorFromYoloDetection(detection, raw_image);
            if (!armor)
            {
                if (detect_options_.process_debug)
                {
                    ROS_INFO_THROTTLE(1.0,
                                      "[t=%.3f][DetectCore][YOLO] Failed to build armor from detection. class_id=%d conf=%.3f roi=(%d,%d,%d,%d)",
                                      ros::Time::now().toSec(), detection.class_id, detection.confidence,
                                      detection_roi.x, detection_roi.y, detection_roi.width, detection_roi.height);
                }
                if (shouldSaveYoloFailedSample())
                {
                    saveYoloSample(raw_image, "armor_build_failed", detection.confidence, &detection_roi);
                }
            }
            else if (detection.class_id == -1 && !targetFilter(*armor, raw_image))
            {
                if (detect_options_.process_debug)
                {
                    ROS_INFO_THROTTLE(1.0,
                                      "[t=%.3f][DetectCore][YOLO] Detection filtered by color. conf=%.3f center=(%.1f, %.1f)",
                                      ros::Time::now().toSec(), detection.confidence, armor->center_.x, armor->center_.y);
                }
                if (shouldSaveYoloFailedSample())
                {
                    saveYoloSample(raw_image, "color_filtered", detection.confidence, &detection_roi);
                }
            }
            else
            {
                best_armor_ = std::move(armor);
                debug_armors_.push_back(*best_armor_);
                if (detect_options_.process_debug)
                {
                    ROS_INFO_THROTTLE(1.0,
                                      "[t=%.3f][DetectCore][YOLO] Selected target center=(%.1f, %.1f) conf=%.3f class_id=%d",
                                      ros::Time::now().toSec(), best_armor_->center_.x, best_armor_->center_.y,
                                      best_armor_->confidence_, best_armor_->class_id_);
                }
            }
        }

        if (!best_armor_)
        {
            if (detect_options_.process_debug)
            {
                ROS_INFO_THROTTLE(1.0, "[t=%.3f][DetectCore][YOLO] No valid target remained after filtering.",
                                  ros::Time::now().toSec());
            }
            if (shouldSaveYoloFailedSample())
            {
                saveYoloSample(raw_image, "no_valid_armor", detection.confidence, &detection_roi);
            }
        }
    }


    //yolo detect
    void DetectCore::detect(const cv::UMat& raw_image)
    {
        detect(raw_image.getMat(cv::ACCESS_READ));
    }

    std::unique_ptr<Armor> DetectCore::buildArmorFromYoloDetection(const YoloDetectionOutput& detection,
                                                                   const cv::Mat& bgr_img)
    {
        constexpr float roi_scale = 0.75f;
        const auto build_fallback_armor = [&](const std::vector<cv::Point2f>& points) {
            auto armor = std::make_unique<Armor>(points);
            armor->confidence_ = detection.confidence;
            armor->class_id_ = detection.class_id;
            return armor;
        };

        if (bgr_img.empty())
        {
            return nullptr;
        }

        std::vector<cv::Point2f> armor_points(detection.armor_points.begin(), detection.armor_points.end());
        const bool points_empty = std::all_of(armor_points.begin(), armor_points.end(), [](const cv::Point2f& point) {
            return std::abs(point.x) < 1e-3f && std::abs(point.y) < 1e-3f;
        });
        if (points_empty)
        {
            if (detection.bbox.width <= 1.0f || detection.bbox.height <= 1.0f)
            {
                return nullptr;
            }
            armor_points = {
                cv::Point2f(detection.bbox.x, detection.bbox.y),
                cv::Point2f(detection.bbox.x + detection.bbox.width, detection.bbox.y),
                cv::Point2f(detection.bbox.x + detection.bbox.width, detection.bbox.y + detection.bbox.height),
                cv::Point2f(detection.bbox.x, detection.bbox.y + detection.bbox.height)};
        }

        if (!detect_options_.yolo_refine_enabled)
        {
            return build_fallback_armor(armor_points);
        }

        cv::Rect roi_box = cv::boundingRect(armor_points);
        roi_box.x -= static_cast<int>(roi_box.width * roi_scale);
        roi_box.y -= static_cast<int>(roi_box.height * roi_scale);
        roi_box.width += static_cast<int>(2 * roi_box.width * roi_scale);
        roi_box.height += static_cast<int>(2 * roi_box.height * roi_scale);
        roi_box &= cv::Rect(0, 0, bgr_img.cols, bgr_img.rows);
        if (roi_box.width <= 1 || roi_box.height <= 1)
        {
            if (detect_options_.process_debug)
            {
                ROS_INFO_THROTTLE(1.0,
                                  "[t=%.3f][DetectCore][YOLO] ROI invalid for light-bar refinement, fallback to YOLO points.",
                                  ros::Time::now().toSec());
            }
            return build_fallback_armor(armor_points);
        }

        const size_t bar_start = bars_.size();
        std::vector<size_t> candidate_indices = findBars(roi_box);

        if (candidate_indices.size() < 2)
        {
            bars_.erase(bars_.begin() + static_cast<std::ptrdiff_t>(bar_start), bars_.end());
            if (detect_options_.process_debug)
            {
                ROS_INFO_THROTTLE(1.0,
                                  "[t=%.3f][DetectCore][YOLO] Only %lu candidate bars found in ROI, fallback to YOLO points.",
                                  ros::Time::now().toSec(), candidate_indices.size());
            }
            return build_fallback_armor(armor_points);
        }

        const auto get_bar_endpoints = [](const Bar& bar) {
            cv::Point2f pts[4];
            bar.bar_rect_.points(pts);
            std::sort(pts, pts + 4, [](const cv::Point2f& lhs, const cv::Point2f& rhs) {
                return lhs.x < rhs.x;
            });
            return std::array<cv::Point2f, 2>{(pts[0] + pts[1]) * 0.5f, (pts[2] + pts[3]) * 0.5f};
        };

        const auto score_bar = [&](const Bar& bar, const cv::Point2f& left_target, const cv::Point2f& right_target) {
            const auto endpoints = get_bar_endpoints(bar);
            return cv::norm(endpoints[0] - left_target) + cv::norm(endpoints[1] - right_target);
        };

        const cv::Point2f& top_left = armor_points[0];
        const cv::Point2f& top_right = armor_points[1];
        const cv::Point2f& bottom_right = armor_points[2];
        const cv::Point2f& bottom_left = armor_points[3];

        size_t best_top_idx = 0;
        size_t best_bottom_idx = 0;
        double best_score = std::numeric_limits<double>::max();
        for (const size_t top_idx : candidate_indices)
        {
            for (const size_t bottom_idx : candidate_indices)
            {
                if (top_idx == bottom_idx || bars_[top_idx].center_point_.y >= bars_[bottom_idx].center_point_.y)
                {
                    continue;
                }

                const double score = score_bar(bars_[top_idx], top_left, top_right) +
                                     score_bar(bars_[bottom_idx], bottom_left, bottom_right);
                if (score < best_score)
                {
                    best_score = score;
                    best_top_idx = top_idx;
                    best_bottom_idx = bottom_idx;
                }
            }
        }

        if (best_score == std::numeric_limits<double>::max())
        {
            bars_.erase(bars_.begin() + static_cast<std::ptrdiff_t>(bar_start), bars_.end());
            if (detect_options_.process_debug)
            {
                ROS_INFO_THROTTLE(1.0,
                                  "[t=%.3f][DetectCore][YOLO] No valid top/bottom bar pair in ROI, fallback to YOLO points.",
                                  ros::Time::now().toSec());
            }
            return build_fallback_armor(armor_points);
        }

        auto armor = std::make_unique<Armor>(bars_[best_top_idx], bars_[best_bottom_idx]);
        armor->confidence_ = detection.confidence;
        armor->class_id_ = detection.class_id;
        if (!refineArmor(*armor))
        {
            armor->bars_4points_.clear();
            armor->bars_4points_.reserve(armor_points.size());
            for (const auto& point : armor_points)
            {
                armor->bars_4points_.emplace_back(point.x, point.y);
            }
            armor->computeGeometry();
            if (shouldSaveYoloFailedSample())
            {
                saveYoloSample(bgr_img, "refine_failed", detection.confidence, &roi_box);
            }
            ROS_DEBUG_THROTTLE(2, "[DetectCore] refineArmor failed, falling back to YOLO armor points.");
        }
        return armor;
    }

    //traditional detect
    void DetectCore::detect(const cv::UMat& raw_image, const cv::UMat* morphology_image)
    {
        reset();
        if (morphology_image)
        {
            findArmor(raw_image, *morphology_image);
        }
    }

    bool DetectCore::isValidBar(const Bar& bar)
    {   
        double angle = static_cast<double>(abs(bar.angle_));
        
        double bar_angle_diff_ = static_cast<double>(abs(abs(angle) - 90.0));
        debugger_.debug(bar);
        if (bar_angle_diff_ > object_options_.max_angle_diff)
            return false;
        if (bar.lw_ratio_ < object_options_.min_lw_ratio || bar.lw_ratio_ > object_options_.max_lw_ratio)
            return false;
        if (bar.pixel_contained_ratio_ < object_options_.min_pixel_contained_ratio)
            return false;
        return true;
    }

    std::vector<size_t> DetectCore::findBars(const cv::Rect& roi_box)
    {
        std::vector<size_t> candidate_indices;
        if (gray_image_.empty() || roi_box.width <= 1 || roi_box.height <= 1)
        {
            return candidate_indices;
        }

        const cv::Mat gray_roi = gray_image_(roi_box);

        cv::Mat binary_roi;
        cv::threshold(gray_roi, binary_roi, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours)
        {
            if (cv::contourArea(contour) < 5.0)
            {
                continue;
            }

            std::vector<cv::Point> global_contour;
            global_contour.reserve(contour.size());
            for (const auto& point : contour)
            {
                global_contour.emplace_back(point.x + roi_box.x, point.y + roi_box.y);
            }

            Bar bar(cv::minAreaRect(global_contour), global_contour);
            if (detect_options_.select_bar && !isValidBar(bar))
            {
                continue;
            }

            bars_.push_back(bar);
            candidate_indices.push_back(bars_.size() - 1);
        }

        return candidate_indices;
    }

    void DetectCore::findBars(const cv::UMat& raw_image, const cv::UMat& morphology_image)
    {
        bars_.clear();
        updateGrayImage(raw_image);

        std::vector<std::vector<cv::Point>> contours;
        cv::Mat morphology_image_mat = morphology_image.getMat(cv::ACCESS_READ);
        findContours(morphology_image_mat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if(object_options_.bar_enabled)
        {
            ROS_INFO_THROTTLE(3, "[findbars] Total contours: %lu, select_bar=%d", contours.size(), detect_options_.select_bar);
        }
        cv::RotatedRect rect;
        for(size_t i=0; i<contours.size(); i++)
        {
            double area = contourArea(contours[i]);
            if (area < 5) continue;
            rect = cv::minAreaRect(contours[i]);
            Bar bar(rect, contours[i]);
            if (detect_options_.select_bar && !isValidBar(bar))
            {
                continue;
            }
            bars_.emplace_back(bar);
        }

        if(object_options_.bar_enabled)
        {
            ROS_INFO_THROTTLE(3, "[findbars] Contours: %lu -> Valid bars: %lu", contours.size(), bars_.size());
        }
    }


    bool DetectCore::calcArmorScore(const Bar& top_bar, const Bar& bottom_bar, double& score)
    {
        const double distance = cv::norm(top_bar.center_point_ - bottom_bar.center_point_);
        const double lens = top_bar.length_len_ + bottom_bar.length_len_;
        const double len_avg = lens * 0.5;
        const double len_ratio = std::max(top_bar.length_len_, bottom_bar.length_len_) /
                                 std::min(top_bar.length_len_, bottom_bar.length_len_);
        const double x_diff = std::fabs(top_bar.center_point_.x - bottom_bar.center_point_.x);
        const double x_ratio = x_diff / len_avg;
        const double angle_diff = std::fabs(std::fabs(top_bar.angle_) - std::fabs(bottom_bar.angle_));
        const double len_diff = std::fabs(top_bar.length_len_ - bottom_bar.length_len_);
        const double center_angle = std::atan2(bottom_bar.center_point_.y - top_bar.center_point_.y,
                                              bottom_bar.center_point_.x - top_bar.center_point_.x) * 180.0 / CV_PI;
        const double vertical_deviation = std::fabs(std::fabs(center_angle) - 90.0);

        debugger_.debug(top_bar, bottom_bar);

        if (len_ratio > object_options_.max_bars_ratio)
        {
            if (object_options_.armor_enabled)
                ROS_INFO("failed in bars ratio, %lf > max bar ratio : %lf", len_ratio, object_options_.max_bars_ratio);
            return false;
        }

        if (distance > lens * object_options_.max_bars_distance || distance < lens * object_options_.min_bars_distance)
        {
            if (object_options_.armor_enabled)
                ROS_INFO("failed in bars distance, %lf > max bars distance : %lf", distance, lens * object_options_.max_bars_distance);
            return false;
        }

        if (x_ratio > object_options_.max_bars_x_dis)
        {
            if (object_options_.armor_enabled)
                ROS_INFO("failed in bars x dis, %lf > max bars x dis : %lf", x_ratio, object_options_.max_bars_x_dis);
            return false;
        }

        if (angle_diff > object_options_.max_bars_angle)
        {
            if (object_options_.armor_enabled)
                ROS_INFO("failed in bars angle, %lf > max bars angle : %lf", angle_diff, object_options_.max_bars_angle);
            return false;
        }
        
        auto get_bar_endpoints = [](const Bar& bar) {
            cv::Point2f pts[4];
            bar.bar_rect_.points(pts);
            std::sort(pts, pts + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
                return a.x < b.x;
            });
            return std::array<cv::Point2f, 2>{(pts[0] + pts[1]) * 0.5f, (pts[2] + pts[3]) * 0.5f};
        };

        const auto top_pts = get_bar_endpoints(top_bar);
        const auto bottom_pts = get_bar_endpoints(bottom_bar);
        const cv::Point2f& top_left = top_pts[0];
        const cv::Point2f& top_right = top_pts[1];
        const cv::Point2f& bottom_left = bottom_pts[0];
        const cv::Point2f& bottom_right = bottom_pts[1];

        const double top_k = (top_left.y - top_right.y) / (top_left.x - top_right.x + 1e-9);
        const double bottom_k = (bottom_right.y - bottom_left.y) / (bottom_right.x - bottom_left.x + 1e-9);
        const double top_b = ((top_left.y + top_right.y) - top_k * (top_left.x + top_right.x)) / 2.0;
        const double bottom_b = ((bottom_right.y + bottom_left.y) - bottom_k * (bottom_right.x + bottom_left.x)) / 2.0;
        const double vertical_dist_bottom_to_top = std::fabs(top_k * bottom_bar.center_point_.x - bottom_bar.center_point_.y + top_b) /
                                                   std::sqrt(top_k * top_k + 1.0);
        const double vertical_dist_top_to_bottom = std::fabs(bottom_k * top_bar.center_point_.x - top_bar.center_point_.y + bottom_b) /
                                                   std::sqrt(bottom_k * bottom_k + 1.0);
        const double parallel_dist_1 = std::sqrt(std::max(0.0, distance * distance - vertical_dist_bottom_to_top * vertical_dist_bottom_to_top));
        const double parallel_dist_2 = std::sqrt(std::max(0.0, distance * distance - vertical_dist_top_to_bottom * vertical_dist_top_to_bottom));
        const double parallel_dist = (parallel_dist_1 + parallel_dist_2) * 0.5;
        if (parallel_dist / distance > object_options_.max_bars_ratio)
        {   
            if (object_options_.armor_enabled)
                ROS_INFO("failed in bars parallel dist, %lf > max bars parallel dist : %lf", parallel_dist / distance, object_options_.max_bars_ratio);
            return false;
        }

        score = 0.0;
        score += (1.0 - x_ratio / object_options_.max_bars_x_dis) * 0.4;
        score += (1.0 - angle_diff / object_options_.max_bars_angle) * 0.3;
        score += (1.0 - len_diff / (len_avg * object_options_.max_bars_ratio)) * 0.2;
        score += (1.0 - vertical_deviation / object_options_.max_bars_angle) * 0.1;
        return true;
    }

    void DetectCore::findArmor(const cv::UMat& raw_image, const cv::UMat& morphology_image)
    {
        best_armor_.reset();
        debug_armors_.clear();
        
        findBars(raw_image, morphology_image);
        if(bars_.size() < 2)
        {
            ROS_WARN_THROTTLE(2, "[findArmor] Only %lu bars found (need >=2). No armor detection possible.", bars_.size());
            return;
        }
        
        if(object_options_.armor_enabled)
        {
            ROS_INFO_THROTTLE(2, "[findArmor] Found %lu bars, searching for armor pairs...", bars_.size());
        }
        std::sort(bars_.begin(), bars_.end(), [](const Bar& a, const Bar& b) {
            return a.center_point_.y < b.center_point_.y;
        });

        double max_bar_length = 0.0;
        for (const auto& bar : bars_)
        {
            if (bar.length_len_ > max_bar_length)
            {
                max_bar_length = bar.length_len_;
            }
        }

        std::vector<ArmorCandidate> candidates;
        candidates.reserve(bars_.size() * (bars_.size() - 1) / 2);

        for (size_t i = 0; i < bars_.size(); i++)
        {
            Bar& bar_top = bars_[i];
            const double max_pair_dy = (bar_top.length_len_ + max_bar_length) * object_options_.max_bars_distance;
            for (size_t j = i + 1; j < bars_.size(); j++)
            {
                Bar& bar_bottom = bars_[j];
                const double dy = bar_bottom.center_point_.y - bar_top.center_point_.y;
                if (dy > max_pair_dy)
                {
                    break;
                }
                double score = 0.0;
                if (calcArmorScore(bar_top, bar_bottom, score))
                {
                    candidates.push_back({i, j, score});
                }
            }
        }

        if (candidates.empty()) {
            ROS_WARN_THROTTLE(2, "[findArmor] %lu bars found but no valid armor pairs matched! ", bars_.size());
            return;
        }

        std::sort(candidates.begin(), candidates.end(), [](const ArmorCandidate& lhs, const ArmorCandidate& rhs) {
            return lhs.score > rhs.score;
        });

        std::vector<bool> bar_used(bars_.size(), false);
        debug_armors_.reserve(candidates.size());
        for (const auto& candidate : candidates)
        {
            if (bar_used[candidate.top_idx] || bar_used[candidate.bottom_idx])
            {
                continue;
            }

            debug_armors_.emplace_back(bars_[candidate.top_idx], bars_[candidate.bottom_idx]);
            debug_armors_.back().confidence_ = candidate.score;
            bar_used[candidate.top_idx] = true;
            bar_used[candidate.bottom_idx] = true;
        }

        if (debug_armors_.empty()) {
            ROS_WARN_THROTTLE(2, "[findArmor] %lu valid pairs found but no drawable armor kept! ", candidates.size());
            return;
        }

        best_armor_ = std::make_unique<Armor>(debug_armors_.front());

        const cv::Size win_size(5, 5);
        const cv::Size zero_zone(-1, -1);
        const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
        std::vector<cv::Point2f> corners;
        corners.reserve(4);
        for (const auto& point : best_armor_->bars_4points_)
        {
            corners.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
        }

        cv::cornerSubPix(gray_image_, corners, win_size, zero_zone, criteria);
        for (int i = 0; i < 4; ++i)
        {
            best_armor_->bars_4points_[i] = cv::Point2d(corners[i].x, corners[i].y);
        }
        best_armor_->computeGeometry();
        debug_armors_.front() = *best_armor_;

        debugger_.debug(*best_armor_);
        if(object_options_.armor_enabled)
        {
            ROS_INFO_THROTTLE(2, "[findArmor] Sub-pixel optimized best armor score=%.3f",
                              best_armor_->confidence_);
        }

    }



bool DetectCore::getCorrectedBarEndpoints(const Bar* bar, const cv::Mat& gray_img,
                                          std::array<cv::Point2f, 2>& endpoints)
{
    if (!bar || gray_img.empty() || bar->length_len_ <= 1.0 || bar->width_len_ <= 1.0)
    {
        return false;
    }

    const float max_brightness = static_cast<float>(std::max(1.0, detect_options_.refine_max_brightness));
    const float roi_scale = static_cast<float>(std::max(0.0, detect_options_.refine_roi_scale));
    const float search_start = static_cast<float>(std::max(0.0, std::min(1.0, detect_options_.refine_search_start)));
    const float search_end = static_cast<float>(std::max(
        static_cast<double>(search_start + 0.01f),
        std::min(1.0, detect_options_.refine_search_end)));

    cv::Rect roi_box = bar->bar_rect_.boundingRect();
    roi_box.x -= static_cast<int>(roi_box.width * roi_scale);
    roi_box.y -= static_cast<int>(roi_box.height * roi_scale);
    roi_box.width += static_cast<int>(2 * roi_box.width * roi_scale);
    roi_box.height += static_cast<int>(2 * roi_box.height * roi_scale);
    roi_box &= cv::Rect(0, 0, gray_img.cols, gray_img.rows);
    if (roi_box.width <= 1 || roi_box.height <= 1)
    {
        return false;
    }

    cv::Mat roi = gray_img(roi_box).clone();
    const float mean_val = static_cast<float>(cv::mean(roi)[0]);
    roi.convertTo(roi, CV_32F);
    cv::normalize(roi, roi, 0, max_brightness, cv::NORM_MINMAX);

    const cv::Moments moments = cv::moments(roi);
    if (std::abs(moments.m00) < 1e-6)
    {
        return false;
    }

    const cv::Point2f centroid(
        static_cast<float>(moments.m10 / moments.m00 + roi_box.x),
        static_cast<float>(moments.m01 / moments.m00 + roi_box.y));

    std::vector<cv::Point2f> points;
    points.reserve(static_cast<size_t>(roi.rows * roi.cols));
    for (int y = 0; y < roi.rows; ++y)
    {
        const float* row = roi.ptr<float>(y);
        for (int x = 0; x < roi.cols; ++x)
        {
            if (row[x] > 1e-3f)
            {
                points.emplace_back(static_cast<float>(x), static_cast<float>(y));
            }
        }
    }

    if (points.size() < 2)
    {
        return false;
    }

    cv::PCA pca(cv::Mat(points).reshape(1), cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
    const float axis_norm = cv::norm(axis);
    if (axis_norm < 1e-6f)
    {
        return false;
    }
    axis /= axis_norm;

    const auto find_corner = [&](int direction) -> cv::Point2f {
        const float dx = axis.x * static_cast<float>(direction);
        const float dy = axis.y * static_cast<float>(direction);
        const float search_length = static_cast<float>(bar->length_len_) * (search_end - search_start);
        const int half_width = std::max(0, static_cast<int>((bar->width_len_ - 2.0) / 2.0));

        std::vector<cv::Point2f> candidates;
        candidates.reserve(static_cast<size_t>(half_width * 2 + 1));

        for (int offset = -half_width; offset <= half_width; ++offset)
        {
            const cv::Point2f start_point(
                centroid.x + static_cast<float>(bar->length_len_) * search_start * dx + offset,
                centroid.y + static_cast<float>(bar->length_len_) * search_start * dy);

            cv::Point2f corner = start_point;
            float max_diff = 0.0f;
            bool found = false;

            for (float step = 1.0f; step < search_length; step += 1.0f)
            {
                const cv::Point2f cur_point(start_point.x + dx * step, start_point.y + dy * step);
                const cv::Point2f prev_point(cur_point.x - dx, cur_point.y - dy);
                if (cur_point.x < 0 || cur_point.x >= gray_img.cols || cur_point.y < 0 || cur_point.y >= gray_img.rows ||
                    prev_point.x < 0 || prev_point.x >= gray_img.cols || prev_point.y < 0 || prev_point.y >= gray_img.rows)
                {
                    break;
                }

                const int cur_x = cvRound(cur_point.x);
                const int cur_y = cvRound(cur_point.y);
                const int prev_x = cvRound(prev_point.x);
                const int prev_y = cvRound(prev_point.y);
                if (cur_x < 0 || cur_x >= gray_img.cols || cur_y < 0 || cur_y >= gray_img.rows ||
                    prev_x < 0 || prev_x >= gray_img.cols || prev_y < 0 || prev_y >= gray_img.rows)
                {
                    break;
                }

                const float prev_val = static_cast<float>(gray_img.at<uchar>(prev_y, prev_x));
                const float cur_val = static_cast<float>(gray_img.at<uchar>(cur_y, cur_x));
                const float diff = prev_val - cur_val;
                if (diff > max_diff && prev_val > mean_val)
                {
                    max_diff = diff;
                    corner = prev_point;
                    found = true;
                }
            }

            if (found)
            {
                candidates.push_back(corner);
            }
        }

        if (candidates.empty())
        {
            return cv::Point2f(-1.0f, -1.0f);
        }

        return std::accumulate(candidates.begin(), candidates.end(), cv::Point2f(0.0f, 0.0f)) /
               static_cast<float>(candidates.size());
    };

    const cv::Point2f end_a = find_corner(1);
    const cv::Point2f end_b = find_corner(-1);
    if (end_a.x < 0.0f || end_b.x < 0.0f)
    {
        return false;
    }

    endpoints = {end_a, end_b};
    return true;
}

bool DetectCore::refineArmor(Armor& armor)
{
    if (gray_image_.empty())
    {
        return false;
    }

    std::array<cv::Point2f, 2> top_endpoints;
    std::array<cv::Point2f, 2> bottom_endpoints;
    if (!getCorrectedBarEndpoints(armor.bar_up_, gray_image_, top_endpoints) ||
        !getCorrectedBarEndpoints(armor.bar_bottom_, gray_image_, bottom_endpoints))
    {
        return false;
    }

    if (top_endpoints[1].x < top_endpoints[0].x)
    {
        std::swap(top_endpoints[0], top_endpoints[1]);
    }
    if (bottom_endpoints[1].x < bottom_endpoints[0].x)
    {
        std::swap(bottom_endpoints[0], bottom_endpoints[1]);
    }

    armor.bars_4points_.clear();
    armor.bars_4points_.emplace_back(top_endpoints[0]);
    armor.bars_4points_.emplace_back(top_endpoints[1]);
    armor.bars_4points_.emplace_back(bottom_endpoints[1]);
    armor.bars_4points_.emplace_back(bottom_endpoints[0]);

    const cv::Size win_size(5, 5);
    const cv::Size zero_zone(-1, -1);
    const cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
    std::vector<cv::Point2f> corners;
    corners.reserve(4);
    for (const auto& point : armor.bars_4points_)
    {
        corners.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }

    cv::cornerSubPix(gray_image_, corners, win_size, zero_zone, criteria);
    for (int i = 0; i < 4; ++i)
    {
        armor.bars_4points_[i] = cv::Point2d(corners[i].x, corners[i].y);
    }
    armor.computeGeometry();
    return true;
}

}  // namespace rm_radarplugin
