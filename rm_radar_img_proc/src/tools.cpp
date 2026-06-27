#include <tools.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <ros/ros.h>

namespace rm_radarplugin
{
namespace tools
{
    Debugger::Debugger(const ObjectOptions& options)
      : options_(options)
    {
    }

    void Debugger::setOptions(const ObjectOptions& options)
    {
        options_ = options;
    }

    void Debugger::debug(const Bar& bar) const
    {
        if (!options_.bar_enabled)
        {
            return;
        }

        const double angle = static_cast<double>(std::abs(bar.angle_));
        const double bar_angle_diff = static_cast<double>(std::abs(std::abs(angle) - 90.0));
        ROS_INFO("bar_angle: %lf",  bar.angle_);
        ROS_INFO("bar_angle_diff: %.2f (max_allowed: %.2f) %s",
                 bar_angle_diff, options_.max_angle_diff,
                 bar_angle_diff > options_.max_angle_diff ? "-> REJECT" : "-> OK");
        ROS_INFO("lw_ratio: %.2f (min: %.2f, max: %.2f) %s",
                 bar.lw_ratio_, options_.min_lw_ratio, options_.max_lw_ratio,
                 (bar.lw_ratio_ < options_.min_lw_ratio || bar.lw_ratio_ > options_.max_lw_ratio) ? "-> REJECT" : "-> OK");
        ROS_INFO("pixel_contained_ratio: %.3f (min: %.3f) %s",
                 bar.pixel_contained_ratio_, options_.min_pixel_contained_ratio,
                 bar.pixel_contained_ratio_ < options_.min_pixel_contained_ratio ? "-> REJECT" : "-> OK");
        ROS_INFO("/////////////////////");
    }

    void Debugger::debug(const Armor& armor) const
    {
        if (!options_.armor_enabled)
        {
            return;
        }

        double bar_ratio = 0.0;
        double bars_dist = 0.0;
        double bars_angle = 0.0;
        double bars_x_dis = 0.0;
        if (armor.bar_up_ != nullptr && armor.bar_bottom_ != nullptr)
        {
            const Bar& top_bar = *armor.bar_up_;
            const Bar& bottom_bar = *armor.bar_bottom_;
            const double top_len = top_bar.length_len_;
            const double bottom_len = bottom_bar.length_len_;
            const double min_len = std::min(top_len, bottom_len);
            const double max_len = std::max(top_len, bottom_len);
            bar_ratio = (min_len > 1e-6) ? (max_len / min_len) : 0.0;

            const double dx = static_cast<double>(top_bar.center_point_.x - bottom_bar.center_point_.x);
            const double dy = static_cast<double>(top_bar.center_point_.y - bottom_bar.center_point_.y);
            bars_dist = std::sqrt(dx * dx + dy * dy);

            bars_angle = std::fabs(static_cast<double>(top_bar.angle_ - bottom_bar.angle_));
            if (bars_angle > 90.0) bars_angle = 180.0 - bars_angle;

            const double lens = top_len + bottom_len;
            bars_x_dis = (lens > 1e-6) ? (std::fabs(dx) / (lens * 0.5)) : 0.0;

            ROS_INFO("max_bars_ratio: %lf > %lf", options_.max_bars_ratio, bar_ratio);
            ROS_INFO("min_bars_distance check: distance=%lf, min_limit=%lf",
                     bars_dist, lens * options_.min_bars_distance);
            ROS_INFO("max_bars_distance check: distance=%lf, max_limit=%lf",
                     bars_dist, lens * options_.max_bars_distance);
            ROS_INFO("top_bar: (%f, %f)", top_bar.center_point_.x, top_bar.center_point_.y);
            ROS_INFO("bottom_bar: (%f, %f)", bottom_bar.center_point_.x, bottom_bar.center_point_.y);
            ROS_INFO("max_bars_angle: %lf > %lf", options_.max_bars_angle, bars_angle);
            ROS_INFO("top_bar_angle: %f", top_bar.angle_);
            ROS_INFO("bottom_bar_angle: %f", bottom_bar.angle_);
            ROS_INFO("max_bars_x_dis: %lf > %lf", options_.max_bars_x_dis, bars_x_dis);
        }

        ROS_INFO("armor center=(%.2f, %.2f), ratio=%.2f, dist=%.2f, angle=%.2f, xdis=%.2f, conf=%.3f",
                 armor.center_.x, armor.center_.y, bar_ratio, bars_dist, bars_angle, bars_x_dis, armor.confidence_);
        ROS_INFO("////////////////////////////////////////////");
    }

    void Debugger::debug(const Bar& top_bar, const Bar& bottom_bar) const
    {
        if (!options_.armor_enabled)
        {
            return;
        }

        Bar top_bar_debug = top_bar;
        Bar bottom_bar_debug = bottom_bar;
        Armor armor_debug(top_bar_debug, bottom_bar_debug);
        debug(armor_debug);
    }

    Visualizer::Visualizer(const VisualizerOptions& options)
      : options_(options)
    {
    }

    void Visualizer::setOptions(const VisualizerOptions& options)
    {
        options_ = options;
    }

    void Visualizer::draw(cv::Mat& image, const Bar& bar) const
    {
        const cv::Scalar line_color = cv::Scalar(0, 255, 0);
        for (int j = 0; j < 4; j++)
        {
            cv::line(image, bar.points_[j], bar.points_[(j + 1) % 4], line_color, options_.line_width);
        }
    }

    void Visualizer::draw(cv::Mat& image, const std::vector<Bar>& bars) const
    {
        for (const auto& bar : bars)
        {
            draw(image, bar);
        }
    }

    void Visualizer::draw(cv::Mat& image, const Armor& armor) const
    {
        const cv::Scalar line_color = cv::Scalar(0, 255, 0);
        for (int j = 0; j < 4; j++)
        {
            cv::line(image, armor.bars_4points_[j], armor.bars_4points_[(j + 1) % 4], line_color, options_.line_width);
        }

        std::ostringstream ss1;
        ss1 << std::fixed << std::setprecision(2) << "s:" << armor.confidence_;
        cv::putText(image, ss1.str(), armor.bars_4points_[0], cv::FONT_HERSHEY_COMPLEX,
                    1.0, cv::Scalar(0, 255, 0), 2);
    }

    void Visualizer::draw(cv::Mat& image, const std::vector<Armor>& armors) const
    {
        for (const auto& armor : armors)
        {
            draw(image, armor);
        }
    }

    void Visualizer::drawVertexes(cv::Mat& image, const Armor& armor, const cv::Point2d& image_center) const
    {
        const cv::Scalar vertex_color = cv::Scalar(0, 255, 0);
        const int cross_size = 15;
        cv::line(image, cv::Point(image_center.x - cross_size, image_center.y),
                 cv::Point(image_center.x + cross_size, image_center.y),
                 cv::Scalar(0, 0, 255), options_.line_width);
        cv::line(image, cv::Point(image_center.x, image_center.y - cross_size),
                 cv::Point(image_center.x, image_center.y + cross_size),
                 cv::Scalar(0, 0, 255), options_.line_width);
        if (!options_.show_centroid_only)
        {
            for (int i = 0; i < 4; i++)
            {
                cv::circle(image, armor.bars_4points_[i], 3, vertex_color, options_.line_width);
                cv::putText(image, std::to_string(i), armor.bars_4points_[i], cv::FONT_HERSHEY_COMPLEX, 1.5, vertex_color);
            }
            cv::line(image, armor.bars_4points_[0], armor.bars_4points_[1], cv::Scalar(0, 255, 0), options_.line_width);
            cv::line(image, armor.bars_4points_[2], armor.bars_4points_[3], cv::Scalar(0, 255, 0), options_.line_width);
            cv::putText(image, std::to_string(5), armor.center_, cv::FONT_HERSHEY_COMPLEX, 1.5, vertex_color);
        }
        cv::circle(image, armor.center_, 3, vertex_color, options_.line_width);
        cv::line(image, armor.center_, image_center, cv::Scalar(255, 0, 0), options_.line_width);
    }

    void Visualizer::drawVertexes(cv::Mat& image, const std::vector<Armor>& armors, const cv::Point2d& image_center) const
    {
        for (const auto& armor : armors)
        {
            drawVertexes(image, armor, image_center);
        }
    }

    void Visualizer::draw(cv::UMat& image, const Bar& bar) const
    {
        const cv::Scalar line_color = cv::Scalar(0, 255, 0);
        for (int j = 0; j < 4; j++)
        {
            cv::line(image, bar.points_[j], bar.points_[(j + 1) % 4], line_color, options_.line_width);
        }
    }

    void Visualizer::draw(cv::UMat& image, const std::vector<Bar>& bars) const
    {
        for (const auto& bar : bars)
        {
            draw(image, bar);
        }
    }

    void Visualizer::draw(cv::UMat& image, const Armor& armor) const
    {
        const cv::Scalar line_color = cv::Scalar(0, 255, 0);
        for (int j = 0; j < 4; j++)
        {
            cv::line(image, armor.bars_4points_[j], armor.bars_4points_[(j + 1) % 4], line_color, options_.line_width);
        }

        std::ostringstream ss1;
        ss1 << std::fixed << std::setprecision(2) << "s:" << armor.confidence_;
        cv::putText(image, ss1.str(), armor.bars_4points_[0], cv::FONT_HERSHEY_COMPLEX,
                    1.0, cv::Scalar(0, 255, 0), 2);
    }

    void Visualizer::draw(cv::UMat& image, const std::vector<Armor>& armors) const
    {
        for (const auto& armor : armors)
        {
            draw(image, armor);
        }
    }

    void Visualizer::drawVertexes(cv::UMat& image, const Armor& armor, const cv::Point2d& image_center) const
    {
        const cv::Scalar vertex_color = cv::Scalar(0, 255, 0);
        const int cross_size = 15;
        cv::line(image, cv::Point(image_center.x - cross_size, image_center.y),
                 cv::Point(image_center.x + cross_size, image_center.y),
                 cv::Scalar(0, 0, 255), options_.line_width);
        cv::line(image, cv::Point(image_center.x, image_center.y - cross_size),
                 cv::Point(image_center.x, image_center.y + cross_size),
                 cv::Scalar(0, 0, 255), options_.line_width);
        if (!options_.show_centroid_only)
        {
            for (int i = 0; i < 4; i++)
            {
                cv::circle(image, armor.bars_4points_[i], 3, vertex_color, options_.line_width);
                cv::putText(image, std::to_string(i), armor.bars_4points_[i], cv::FONT_HERSHEY_COMPLEX, 1.5, vertex_color);
            }
            cv::line(image, armor.bars_4points_[0], armor.bars_4points_[1], cv::Scalar(0, 255, 0), options_.line_width);
            cv::line(image, armor.bars_4points_[2], armor.bars_4points_[3], cv::Scalar(0, 255, 0), options_.line_width);
            cv::putText(image, std::to_string(5), armor.center_, cv::FONT_HERSHEY_COMPLEX, 1.5, vertex_color);
        }
        cv::circle(image, armor.center_, 3, vertex_color, options_.line_width);
        cv::line(image, armor.center_, image_center, cv::Scalar(255, 0, 0), options_.line_width);
    }

    void Visualizer::drawVertexes(cv::UMat& image, const std::vector<Armor>& armors, const cv::Point2d& image_center) const
    {
        for (const auto& armor : armors)
        {
            drawVertexes(image, armor, image_center);
        }
    }

    void Visualizer::resetFps()
    {
        fps_ema_ = 0.0;
        last_fps_wall_ = ros::WallTime();
    }

    void Visualizer::showFps(cv::Mat& image)
    {
        const ros::WallTime now_wall = ros::WallTime::now();
        if (!last_fps_wall_.isZero())
        {
            const double dt = (now_wall - last_fps_wall_).toSec();
            if (dt > 1e-4 && dt < 1.0)
            {
                const double fps_inst = 1.0 / dt;
                const double alpha = 0.1;
                fps_ema_ = (fps_ema_ <= 1e-6) ? fps_inst : (alpha * fps_inst + (1.0 - alpha) * fps_ema_);
            }
        }
        last_fps_wall_ = now_wall;

        char buf[64];
        std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps_ema_);
        cv::putText(image, buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 255), 2);
    }

    void Visualizer::showFps(cv::UMat& image)
    {
        const ros::WallTime now_wall = ros::WallTime::now();
        if (!last_fps_wall_.isZero())
        {
            const double dt = (now_wall - last_fps_wall_).toSec();
            if (dt > 1e-4 && dt < 1.0)
            {
                const double fps_inst = 1.0 / dt;
                const double alpha = 0.1;
                fps_ema_ = (fps_ema_ <= 1e-6) ? fps_inst : (alpha * fps_inst + (1.0 - alpha) * fps_ema_);
            }
        }
        last_fps_wall_ = now_wall;

        char buf[64];
        std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps_ema_);
        cv::putText(image, buf, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 255), 2);
    }

}  // namespace tools
}  // namespace rm_radarplugin
