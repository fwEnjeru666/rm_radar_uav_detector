#pragma once

namespace rm_radarplugin
{

struct ObjectOptions
{
    bool bar_enabled{false};
    bool armor_enabled{false};
    double max_angle_diff{0.0};
    double min_lw_ratio{0.0};
    double max_lw_ratio{0.0};
    double min_pixel_contained_ratio{0.0};
    double max_bars_ratio{0.0};
    double min_bars_distance{0.0};
    double max_bars_distance{0.0};
    double max_bars_angle{0.0};
    double max_bars_x_dis{0.0};
};

}  // namespace rm_radarplugin
