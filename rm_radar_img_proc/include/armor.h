#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace rm_radarplugin
{
    class Bar
    {
        public:
        cv::RotatedRect bar_rect_;
        double length_len_;  /// long side
        double width_len_;   /// short side
        double lw_ratio_;
        cv::Point2f points_[4];
        cv::Point2f center_point_;
        double angle_;  /// the angle between short side of bar and x axis
        double pixel_contained_ratio_;

        Bar(cv::RotatedRect bar_rect, const std::vector<cv::Point>& contour) : bar_rect_(std::move(bar_rect))
        {
            length_len_ = std::max(bar_rect_.size.width, bar_rect_.size.height);
            width_len_ = std::min(bar_rect_.size.width, bar_rect_.size.height);
            pixel_contained_ratio_ = cv::contourArea(contour) / bar_rect_.size.area();
            lw_ratio_ = length_len_ / width_len_;
            bar_rect_.points(points_);
            center_point_ = bar_rect_.center;
            if (length_len_ == bar_rect_.size.width)
            {
            angle_ = bar_rect_.angle + 90.0;
            }
            else
            {
            angle_ = bar_rect_.angle;
            }
            // Use an unsigned orientation (0~90 deg) to avoid +/-90 toggling.
            angle_ = std::fabs(angle_);
        };
    };

    class Armor
    {
        public:
        Bar* bar_up_;
        Bar* bar_bottom_;
        std::vector<cv::Point2d> bars_4points_;        /// bl, tl, tr, br ; center point

        cv::Point2d center_;
        double confidence_;
        int class_id_{-1};
        double angle_pca_deg_{0.0};

        Armor()
        {
            bar_up_ = nullptr;
            bar_bottom_ = nullptr;
            confidence_ = 0;
        };

        Armor(Bar& bar_up, Bar& bar_bottom)
        {
            bar_up_ = &bar_up;
            bar_bottom_ = &bar_bottom;
            getBarsPoints();
            computeGeometry();
            confidence_ = 0;
        };

        explicit Armor(const std::vector<cv::Point2f>& armor_points)
        {
            bar_up_ = nullptr;
            bar_bottom_ = nullptr;
            bars_4points_.reserve(armor_points.size());
            for (const auto& point : armor_points)
            {
                bars_4points_.emplace_back(point.x, point.y);
            }
            computeGeometry();
            confidence_ = 0;
        };

        void computeGeometry()
        {
            getCenterPoint();
            getArmorAnglePCA();
        }
        inline double getArmorAnglePcaDeg() const { return angle_pca_deg_; }

        // PCA major-axis heading (more stable under vertex jitter)
        void getArmorAnglePCA()
        {
            if (bars_4points_.size() != 4) {
                angle_pca_deg_ = 0.0;
                return;
            }

            // Compute mean
            double mx = 0.0, my = 0.0;
            for (const auto& p : bars_4points_) {
                mx += p.x;
                my += p.y;
            }
            mx /= 4.0;
            my /= 4.0;

            // 2x2 covariance (unnormalized is fine for eigenvectors)
            double sxx = 0.0, sxy = 0.0, syy = 0.0;
            for (const auto& p : bars_4points_) {
                const double x = p.x - mx;
                const double y = p.y - my;
                sxx += x * x;
                sxy += x * y;
                syy += y * y;
            }

            // Degenerate check
            const double tr = sxx + syy;
            if (tr < 1e-6) {
                angle_pca_deg_ = 0.0;
                return;
            }

            // Eigenvector for the largest eigenvalue of [[sxx,sxy],[sxy,syy]]
            // Solve analytically: angle = 0.5 * atan2(2*sxy, sxx - syy)
            double theta = 0.5 * std::atan2(2.0 * sxy, (sxx - syy));
            double a = theta * 180.0 / CV_PI;

            // Normalize to [-90, 90]
            while (a > 90.0)  a -= 180.0;
            while (a < -90.0) a += 180.0;

            angle_pca_deg_ = a;
        }

        std::vector<cv::Point2f> getHorizontalBarEndpoints(const Bar* bar)
        {
            cv::Point2f pts[4];
            bar->bar_rect_.points(pts);

            std::sort(pts, pts + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
                return a.x < b.x;
            });

            cv::Point2f l_c, r_c; //left center, right center

            l_c = (pts[0] + pts[1]) * 0.5;
            r_c = (pts[2] + pts[3]) * 0.5;
            return {l_c, r_c};

        }

        void getBarsPoints()
        {
            bars_4points_.clear();

            auto up_bar_pts = getHorizontalBarEndpoints(bar_up_);
            auto bottom_bar_pts = getHorizontalBarEndpoints(bar_bottom_);

            if (up_bar_pts[1].x < up_bar_pts[0].x)
            {
                std::swap(up_bar_pts[0], up_bar_pts[1]);
            }
            if (bottom_bar_pts[1].x < bottom_bar_pts[0].x)
            {
                std::swap(bottom_bar_pts[0], bottom_bar_pts[1]);
            }

            cv::Point2f top_left = up_bar_pts[0];
            cv::Point2f top_right = up_bar_pts[1];
            cv::Point2f bottom_left = bottom_bar_pts[0];
            cv::Point2f bottom_right = bottom_bar_pts[1];

            //p0: TL
            bars_4points_.emplace_back(top_left);
            //p1: TR
            bars_4points_.emplace_back(top_right);
            //p2: BR
            bars_4points_.emplace_back(bottom_right);
            //p3: BL
            bars_4points_.emplace_back(bottom_left);
        };

        void getCenterPoint()
        {
            double p1_x = bars_4points_[0].x;
            double p1_y = bars_4points_[0].y;
            double p2_x = bars_4points_[1].x;
            double p2_y = bars_4points_[1].y;
            double p3_x = bars_4points_[2].x;
            double p3_y = bars_4points_[2].y;
            double p4_x = bars_4points_[3].x;
            double p4_y = bars_4points_[3].y;

            double line1_k = (p1_y - p3_y) / (p1_x - p3_x + 0.000000001);
            double line2_k = (p2_y - p4_y) / (p2_x - p4_x + 0.000000001);
            double line1_b = ((p1_y + p3_y) - line1_k * (p1_x + p3_x)) / 2;
            double line2_b = ((p2_y + p4_y) - line2_k * (p2_x + p4_x)) / 2;

            double cross_point_x = (line2_b - line1_b) / (line1_k - line2_k + 0.000000001);
            double cross_point_y = ((line1_k + line2_k) * cross_point_x + line1_b + line2_b) * 0.5;

            center_.x = int(cross_point_x);
            center_.y = int(cross_point_y);
        }
    };
}
