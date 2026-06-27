#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <ros/time.h>

#include <armor.h>
#include <object_options.h>

namespace rm_radarplugin
{
namespace tools
{
    struct VisualizerOptions
    {
        int line_width{1};
        bool show_centroid_only{false};
    };

    class Debugger
    {
    public:
        explicit Debugger(const ObjectOptions& options);

        void setOptions(const ObjectOptions& options);

        void debug(const Bar& bar) const;
        void debug(const Armor& armor) const;
        void debug(const Bar& top_bar, const Bar& bottom_bar) const;

    private:
        ObjectOptions options_{};
    };

    class Visualizer
    {
    public:
        explicit Visualizer(const VisualizerOptions& options);

        void setOptions(const VisualizerOptions& options);

        void draw(cv::Mat& image, const Bar& bar) const;
        void draw(cv::Mat& image, const std::vector<Bar>& bars) const;
        void draw(cv::Mat& image, const Armor& armor) const;
        void draw(cv::Mat& image, const std::vector<Armor>& armors) const;
        void drawVertexes(cv::Mat& image, const Armor& armor, const cv::Point2d& image_center) const;
        void drawVertexes(cv::Mat& image, const std::vector<Armor>& armors, const cv::Point2d& image_center) const;
        void showFps(cv::Mat& image);

        void draw(cv::UMat& image, const Bar& bar) const;
        void draw(cv::UMat& image, const std::vector<Bar>& bars) const;
        void draw(cv::UMat& image, const Armor& armor) const;
        void draw(cv::UMat& image, const std::vector<Armor>& armors) const;
        void drawVertexes(cv::UMat& image, const Armor& armor, const cv::Point2d& image_center) const;
        void drawVertexes(cv::UMat& image, const std::vector<Armor>& armors, const cv::Point2d& image_center) const;
        void resetFps();
        void showFps(cv::UMat& image);

    private:
        VisualizerOptions options_{};
        double fps_ema_{0.0};
        ros::WallTime last_fps_wall_{};
    };
}  // namespace tools
}  // namespace rm_radarplugin
