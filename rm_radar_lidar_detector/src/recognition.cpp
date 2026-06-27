#include "recognition.h"

#include "dbscan.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <queue>
#include <utility>

#include <Eigen/Eigenvalues>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>

namespace rm_radar_lidar_detector
{

OnlineTemplateRecognition::OnlineTemplateRecognition(const Params& params) : params_(params)
{
    sanitizeParams();
}

void OnlineTemplateRecognition::setParams(const Params& params)
{
    params_ = params;
    sanitizeParams();
    if (initialized_ && static_cast<int>(template_descriptor_.radial_histogram.size()) != params_.radial_bins)
    {
        reset();
    }
}

const OnlineTemplateRecognition::Params& OnlineTemplateRecognition::getParams() const
{
    return params_;
}

bool OnlineTemplateRecognition::initialize(const PointCloudConstPtr& target_cloud, const PointT& marked_point)
{
    reset();
    sanitizeParams();

    if (!target_cloud || target_cloud->empty() || !isFinite(marked_point))
    {
        return false;
    }

    const int anchor_index = findNearestPointIndex(target_cloud, marked_point);
    if (anchor_index < 0)
    {
        return false;
    }

    Descriptor descriptor = buildDescriptor(target_cloud, anchor_index);
    if (!descriptor.valid)
    {
        return false;
    }

    template_anchor_point_ = target_cloud->points[static_cast<std::size_t>(anchor_index)];
    marked_offset_from_anchor_ = subtract(marked_point, template_anchor_point_);
    last_marked_point_ = marked_point;
    template_descriptor_ = descriptor;
    initialized_ = true;
    return true;
}

OnlineTemplateRecognition::Result OnlineTemplateRecognition::match(const PointCloudConstPtr& target_cloud) const
{
    Result result;
    if (!initialized_ || !target_cloud || target_cloud->empty())
    {
        return result;
    }

    const bool use_position_gate = params_.previous_position_gate > 0.0f && isFinite(last_marked_point_);
    const float gate_distance_sq = params_.previous_position_gate * params_.previous_position_gate;

    float best_score = std::numeric_limits<float>::infinity();
    float best_descriptor_distance = std::numeric_limits<float>::infinity();
    Descriptor best_descriptor;
    PointT best_anchor;
    std::size_t best_index = 0;

    for (std::size_t point_index = 0; point_index < target_cloud->size(); ++point_index)
    {
        const PointT& candidate_anchor = target_cloud->points[point_index];
        if (!isFinite(candidate_anchor))
        {
            continue;
        }

        const PointT candidate_marked_point = add(candidate_anchor, marked_offset_from_anchor_);
        if (use_position_gate && squaredDistance(candidate_marked_point, last_marked_point_) > gate_distance_sq)
        {
            continue;
        }

        const Descriptor candidate_descriptor = buildDescriptor(target_cloud, static_cast<int>(point_index));
        if (!candidate_descriptor.valid)
        {
            continue;
        }

        const float descriptor_distance = descriptorDistance(candidate_descriptor);
        float score = descriptor_distance;
        if (params_.previous_position_weight > 0.0f && isFinite(last_marked_point_))
        {
            const float motion_distance = std::sqrt(squaredDistance(candidate_marked_point, last_marked_point_));
            score += params_.previous_position_weight * motion_distance / std::max(params_.support_radius, 1e-3f);
        }

        if (score < best_score)
        {
            best_score = score;
            best_descriptor_distance = descriptor_distance;
            best_descriptor = candidate_descriptor;
            best_anchor = candidate_anchor;
            best_index = point_index;
        }
    }

    if (!std::isfinite(best_score))
    {
        return result;
    }

    result.confidence = confidenceFromDistance(best_descriptor_distance);
    result.matched = best_descriptor_distance <= params_.max_descriptor_distance;
    result.anchor_point = best_anchor;
    result.marked_point = add(best_anchor, marked_offset_from_anchor_);
    result.anchor_index = best_index;
    result.descriptor_distance = best_descriptor_distance;
    result.support_points = best_descriptor.support_points;
    return result;
}

OnlineTemplateRecognition::Result OnlineTemplateRecognition::update(const PointCloudConstPtr& target_cloud)
{
    Result result = match(target_cloud);
    if (!result.matched)
    {
        return result;
    }

    last_marked_point_ = result.marked_point;
    if (result.confidence >= params_.min_update_confidence && params_.template_update_rate > 0.0f)
    {
        const Descriptor candidate_descriptor = buildDescriptor(target_cloud, static_cast<int>(result.anchor_index));
        if (candidate_descriptor.valid)
        {
            blendTemplateDescriptor(candidate_descriptor);
        }
    }
    return result;
}

void OnlineTemplateRecognition::reset()
{
    initialized_ = false;
    template_descriptor_ = Descriptor();
    template_anchor_point_ = PointT();
    marked_offset_from_anchor_ = PointT();
    last_marked_point_ = PointT();
}

bool OnlineTemplateRecognition::initialized() const
{
    return initialized_;
}

const OnlineTemplateRecognition::PointT& OnlineTemplateRecognition::lastMarkedPoint() const
{
    return last_marked_point_;
}

const OnlineTemplateRecognition::PointT& OnlineTemplateRecognition::templateAnchorPoint() const
{
    return template_anchor_point_;
}

bool OnlineTemplateRecognition::isFinite(const PointT& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

float OnlineTemplateRecognition::squaredDistance(const PointT& first, const PointT& second)
{
    const float dx = first.x - second.x;
    const float dy = first.y - second.y;
    const float dz = first.z - second.z;
    return dx * dx + dy * dy + dz * dz;
}

OnlineTemplateRecognition::PointT OnlineTemplateRecognition::add(const PointT& point, const PointT& offset)
{
    PointT result;
    result.x = point.x + offset.x;
    result.y = point.y + offset.y;
    result.z = point.z + offset.z;
    return result;
}

OnlineTemplateRecognition::PointT OnlineTemplateRecognition::subtract(const PointT& first, const PointT& second)
{
    PointT result;
    result.x = first.x - second.x;
    result.y = first.y - second.y;
    result.z = first.z - second.z;
    return result;
}

bool OnlineTemplateRecognition::sanitizeParams()
{
    params_.support_radius = std::max(params_.support_radius, 1e-3f);
    params_.radial_bins = std::max(params_.radial_bins, 1);
    params_.min_support_points = std::max(params_.min_support_points, 1);
    params_.max_descriptor_distance = std::max(params_.max_descriptor_distance, 1e-3f);
    params_.template_update_rate = std::clamp(params_.template_update_rate, 0.0f, 1.0f);
    params_.min_update_confidence = std::clamp(params_.min_update_confidence, 0.0f, 1.0f);
    params_.previous_position_gate = std::max(params_.previous_position_gate, 0.0f);
    params_.previous_position_weight = std::max(params_.previous_position_weight, 0.0f);
    return true;
}

int OnlineTemplateRecognition::findNearestPointIndex(const PointCloudConstPtr& cloud, const PointT& point) const
{
    if (!cloud || cloud->empty())
    {
        return -1;
    }

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<int> nearest_indices(1);
    std::vector<float> nearest_distances(1);
    if (kdtree.nearestKSearch(point, 1, nearest_indices, nearest_distances) <= 0)
    {
        return -1;
    }
    return nearest_indices.front();
}

OnlineTemplateRecognition::Descriptor OnlineTemplateRecognition::buildDescriptor(const PointCloudConstPtr& cloud,
                                                                                  int anchor_index) const
{
    Descriptor descriptor;
    descriptor.radial_histogram.assign(static_cast<std::size_t>(params_.radial_bins), 0.0f);
    if (!cloud || cloud->empty() || anchor_index < 0 || anchor_index >= static_cast<int>(cloud->size()))
    {
        return descriptor;
    }

    const PointT& anchor = cloud->points[static_cast<std::size_t>(anchor_index)];
    if (!isFinite(anchor))
    {
        return descriptor;
    }

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<int> support_indices;
    std::vector<float> support_distances;
    if (kdtree.radiusSearch(anchor, params_.support_radius, support_indices, support_distances) <= 0)
    {
        return descriptor;
    }

    float radius_sum = 0.0f;
    for (std::size_t support_index = 0; support_index < support_indices.size(); ++support_index)
    {
        const float distance_sq = support_distances[support_index];
        if (distance_sq <= 1e-10f)
        {
            continue;
        }

        const float distance = std::sqrt(distance_sq);
        const float normalized_distance = std::min(distance / params_.support_radius, 0.999999f);
        const int bin_index = std::min(static_cast<int>(normalized_distance * params_.radial_bins), params_.radial_bins - 1);
        descriptor.radial_histogram[static_cast<std::size_t>(bin_index)] += 1.0f;
        radius_sum += distance;
        ++descriptor.support_points;
    }

    if (descriptor.support_points < params_.min_support_points)
    {
        descriptor.support_points = 0;
        return descriptor;
    }

    const float support_count = static_cast<float>(descriptor.support_points);
    for (float& histogram_value : descriptor.radial_histogram)
    {
        histogram_value /= support_count;
    }
    descriptor.mean_radius = radius_sum / support_count;
    descriptor.valid = true;
    return descriptor;
}

float OnlineTemplateRecognition::descriptorDistance(const Descriptor& candidate) const
{
    if (!template_descriptor_.valid || !candidate.valid ||
        template_descriptor_.radial_histogram.size() != candidate.radial_histogram.size())
    {
        return std::numeric_limits<float>::infinity();
    }

    float histogram_distance = 0.0f;
    for (std::size_t bin_index = 0; bin_index < template_descriptor_.radial_histogram.size(); ++bin_index)
    {
        histogram_distance += std::abs(template_descriptor_.radial_histogram[bin_index] -
                                       candidate.radial_histogram[bin_index]);
    }

    const float support_max = static_cast<float>(std::max(template_descriptor_.support_points, candidate.support_points));
    const float support_penalty = support_max > 0.0f ?
                                      std::abs(static_cast<float>(template_descriptor_.support_points - candidate.support_points)) /
                                          support_max :
                                      0.0f;
    const float radius_penalty = std::abs(template_descriptor_.mean_radius - candidate.mean_radius) /
                                 std::max(params_.support_radius, 1e-3f);
    return histogram_distance + 0.2f * support_penalty + 0.2f * radius_penalty;
}

float OnlineTemplateRecognition::confidenceFromDistance(float descriptor_distance) const
{
    if (!std::isfinite(descriptor_distance))
    {
        return 0.0f;
    }
    return std::clamp(1.0f - descriptor_distance / params_.max_descriptor_distance, 0.0f, 1.0f);
}

void OnlineTemplateRecognition::blendTemplateDescriptor(const Descriptor& candidate)
{
    if (!template_descriptor_.valid || !candidate.valid ||
        template_descriptor_.radial_histogram.size() != candidate.radial_histogram.size())
    {
        return;
    }

    const float update_rate = params_.template_update_rate;
    for (std::size_t bin_index = 0; bin_index < template_descriptor_.radial_histogram.size(); ++bin_index)
    {
        template_descriptor_.radial_histogram[bin_index] =
            (1.0f - update_rate) * template_descriptor_.radial_histogram[bin_index] +
            update_rate * candidate.radial_histogram[bin_index];
    }
    template_descriptor_.mean_radius =
        (1.0f - update_rate) * template_descriptor_.mean_radius + update_rate * candidate.mean_radius;
    template_descriptor_.support_points = static_cast<int>(std::round(
        (1.0f - update_rate) * static_cast<float>(template_descriptor_.support_points) +
        update_rate * static_cast<float>(candidate.support_points)));
}

FrontViewProjector::FrontViewProjector(const Params& params) : params_(params)
{
    sanitizeParams();
}

void FrontViewProjector::setParams(const Params& params)
{
    params_ = params;
    sanitizeParams();
}

const FrontViewProjector::Params& FrontViewProjector::getParams() const
{
    return params_;
}

bool FrontViewProjector::initialize(const PointCloudConstPtr& target_cloud, const Eigen::Vector3f& sensor_origin)
{
    reset();
    sanitizeParams();

    Eigen::Vector3f centroid;
    int finite_count = 0;
    if (!computeCentroid(target_cloud, centroid, finite_count) || finite_count < params_.min_points)
    {
        return false;
    }

    Eigen::Vector3f view_direction = sensor_origin - centroid;
    if (view_direction.norm() < 1e-4f)
    {
        view_direction = Eigen::Vector3f::UnitY();
    }
    Eigen::Vector3f front_normal = view_direction.normalized();

    origin_ = centroid;
    axis_u_ = Eigen::Vector3f::UnitY();
    axis_v_ = Eigen::Vector3f::UnitZ();
    front_normal_ = front_normal;
    initialized_ = true;
    return true;
}

FrontViewProjector::Projection FrontViewProjector::project(const PointCloudConstPtr& target_cloud,
                                                           const Eigen::Vector3f& sensor_origin) const
{
    Projection projection;
    if (!initialized_ || !target_cloud || target_cloud->empty())
    {
        return projection;
    }

    std::vector<ProjectedPoint> projected_points;
    projected_points.reserve(target_cloud->size());
    float min_u = std::numeric_limits<float>::infinity();
    float max_u = -std::numeric_limits<float>::infinity();
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    std::vector<float> point_ranges;
    point_ranges.reserve(target_cloud->size());

    for (const auto& point : target_cloud->points)
    {
        if (!isFinite(point))
        {
            continue;
        }

        const Eigen::Vector3f point_vec = toEigen(point);
        const Eigen::Vector3f ray = point_vec - sensor_origin;
        const float horizontal_range = std::hypot(ray.x(), ray.y());
        const float range_sq = ray.squaredNorm();
        if (range_sq <= 1e-8f)
        {
            continue;
        }
        ProjectedPoint projected_point;
        projected_point.u = std::atan2(ray.y(), ray.x());
        projected_point.v = std::atan2(ray.z(), horizontal_range);
        projected_point.range_sq = range_sq;
        projected_point.point = point;
        projected_points.push_back(projected_point);
        point_ranges.push_back(std::sqrt(range_sq));

        min_u = std::min(min_u, projected_point.u);
        max_u = std::max(max_u, projected_point.u);
        min_v = std::min(min_v, projected_point.v);
        max_v = std::max(max_v, projected_point.v);
    }

    if (projected_points.size() < static_cast<std::size_t>(params_.min_points) || !std::isfinite(min_u) ||
        !std::isfinite(max_u) || !std::isfinite(min_v) || !std::isfinite(max_v))
    {
        return projection;
    }

    const float span_u = std::max(max_u - min_u, params_.grid_resolution);
    const float span_v = std::max(max_v - min_v, params_.grid_resolution);
    float resolution = params_.grid_resolution;
    if (params_.adaptive_resolution)
    {
        const float median_neighbor_distance = estimateMedianNearestDistance(target_cloud);
        if (std::isfinite(median_neighbor_distance) && median_neighbor_distance > 0.0f && !point_ranges.empty())
        {
            const auto median_range_it = point_ranges.begin() + static_cast<std::ptrdiff_t>(point_ranges.size() / 2);
            std::nth_element(point_ranges.begin(), median_range_it, point_ranges.end());
            const float median_range = std::max(*median_range_it, 1e-3f);
            const float adaptive_resolution = median_neighbor_distance / median_range * params_.adaptive_resolution_scale;
            resolution = std::max(resolution, std::min(adaptive_resolution, params_.max_grid_resolution));
        }
    }
    const float max_span = std::max(span_u, span_v);
    if (max_span / resolution > static_cast<float>(params_.max_image_size))
    {
        resolution = max_span / static_cast<float>(params_.max_image_size);
    }

    PointCloud::Ptr surface_cloud(new PointCloud);
    surface_cloud->header = target_cloud->header;
    surface_cloud->reserve(projected_points.size());
    for (const auto& projected_point : projected_points)
    {
        surface_cloud->push_back(projected_point.point);
    }
    surface_cloud->width = static_cast<uint32_t>(surface_cloud->size());
    surface_cloud->height = 1;
    surface_cloud->is_dense = false;

    const int image_cols = std::max(1, static_cast<int>(std::ceil(span_u / resolution)) + 1);
    const int image_rows = std::max(1, static_cast<int>(std::ceil(span_v / resolution)) + 1);
    cv::Mat image(image_rows, image_cols, CV_8UC1, cv::Scalar(0));
    cv::Mat source_index_image(image_rows, image_cols, CV_32SC1, cv::Scalar(-1));
    for (std::size_t point_index = 0; point_index < projected_points.size(); ++point_index)
    {
        const auto& projected_point = projected_points[point_index];
        const int raw_col = std::clamp(static_cast<int>((projected_point.u - min_u) / resolution), 0, image_cols - 1);
        const int col = image_cols - 1 - raw_col;
        const int row_from_bottom =
            std::clamp(static_cast<int>((projected_point.v - min_v) / resolution), 0, image_rows - 1);
        const int row = image_rows - 1 - row_from_bottom;
        image.at<unsigned char>(row, col) = 255;
        if (source_index_image.at<int>(row, col) < 0)
        {
            source_index_image.at<int>(row, col) = static_cast<int>(point_index);
        }
    }
    const int occupied_cells = cv::countNonZero(image);

    cv::Mat display_image;
    cv::cvtColor(image, display_image, cv::COLOR_GRAY2BGR);
    projection.valid = occupied_cells > 0;
    projection.image = display_image;
    projection.source_index_image = source_index_image;
    projection.surface_cloud = surface_cloud;
    projection.resolution = resolution;
    projection.min_u = min_u;
    projection.max_u = max_u;
    projection.min_v = min_v;
    projection.max_v = max_v;
    projection.occupied_cells = occupied_cells;
    return projection;
}

void FrontViewProjector::reset()
{
    initialized_ = false;
    origin_ = Eigen::Vector3f::Zero();
    axis_u_ = Eigen::Vector3f::UnitX();
    axis_v_ = Eigen::Vector3f::UnitZ();
    front_normal_ = Eigen::Vector3f::UnitY();
}

bool FrontViewProjector::initialized() const
{
    return initialized_;
}

const Eigen::Vector3f& FrontViewProjector::origin() const
{
    return origin_;
}

const Eigen::Vector3f& FrontViewProjector::axisU() const
{
    return axis_u_;
}

const Eigen::Vector3f& FrontViewProjector::axisV() const
{
    return axis_v_;
}

const Eigen::Vector3f& FrontViewProjector::frontNormal() const
{
    return front_normal_;
}

bool FrontViewProjector::isFinite(const PointT& point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

Eigen::Vector3f FrontViewProjector::toEigen(const PointT& point)
{
    return Eigen::Vector3f(point.x, point.y, point.z);
}

bool FrontViewProjector::computeCentroid(const PointCloudConstPtr& cloud, Eigen::Vector3f& centroid, int& finite_count)
{
    centroid = Eigen::Vector3f::Zero();
    finite_count = 0;
    if (!cloud || cloud->empty())
    {
        return false;
    }

    for (const auto& point : cloud->points)
    {
        if (!isFinite(point))
        {
            continue;
        }
        centroid += toEigen(point);
        ++finite_count;
    }

    if (finite_count <= 0)
    {
        return false;
    }
    centroid /= static_cast<float>(finite_count);
    return true;
}

float FrontViewProjector::estimateMedianNearestDistance(const PointCloudConstPtr& cloud) const
{
    if (!cloud || cloud->size() < 2)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    pcl::KdTreeFLANN<PointT> kdtree;
    kdtree.setInputCloud(cloud);

    std::vector<float> nearest_distances;
    nearest_distances.reserve(cloud->size());
    std::vector<int> neighbor_indices;
    std::vector<float> neighbor_distances;
    neighbor_indices.reserve(2);
    neighbor_distances.reserve(2);

    for (const auto& point : cloud->points)
    {
        if (!isFinite(point))
        {
            continue;
        }

        neighbor_indices.clear();
        neighbor_distances.clear();
        if (kdtree.nearestKSearch(point, 2, neighbor_indices, neighbor_distances) >= 2 &&
            neighbor_distances[1] > 0.0f)
        {
            nearest_distances.push_back(std::sqrt(neighbor_distances[1]));
        }
    }

    if (nearest_distances.empty())
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const auto median_it = nearest_distances.begin() + static_cast<std::ptrdiff_t>(nearest_distances.size() / 2);
    std::nth_element(nearest_distances.begin(), median_it, nearest_distances.end());
    return *median_it;
}

bool FrontViewProjector::sanitizeParams()
{
    params_.grid_resolution = std::max(params_.grid_resolution, 1e-3f);
    params_.adaptive_resolution_scale = std::max(params_.adaptive_resolution_scale, 0.1f);
    params_.max_grid_resolution = std::max(params_.max_grid_resolution, params_.grid_resolution);
    params_.max_image_size = std::max(params_.max_image_size, 32);
    params_.min_points = std::max(params_.min_points, 1);
    return true;
}


void TargetCloudExtractor::setParams(const Params& params)
{
    params_ = params;
    params_.min_support_points = std::max(2, params_.min_support_points);
    params_.height_window = std::max(0.02f, params_.height_window);
}

TargetCloudExtractor::PointCloudPtr TargetCloudExtractor::extract(
    const PointCloudConstPtr& detected_cluster) const
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    const auto finalize_target_cloud = [&]() {
        target_cloud->header = detected_cluster->header;
        target_cloud->width = static_cast<uint32_t>(target_cloud->size());
        target_cloud->height = 1;
        target_cloud->is_dense = false;
    };

    if (!detected_cluster || detected_cluster->empty())
    {
        return target_cloud;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr finite_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    finite_cloud->reserve(detected_cluster->size());

    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (const auto& point : detected_cluster->points)
    {
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
        {
            finite_cloud->push_back(point);
            centroid += Eigen::Vector3f(point.x, point.y, point.z);
        }
    }
    if (finite_cloud->size() < static_cast<std::size_t>(params_.min_support_points))
    {
        return target_cloud;
    }
    centroid /= static_cast<float>(finite_cloud->size());

    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (const auto& point : finite_cloud->points)
    {
        const Eigen::Vector3f centered = Eigen::Vector3f(point.x, point.y, point.z) - centroid;
        covariance += centered * centered.transpose();
    }
    covariance /= static_cast<float>(finite_cloud->size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
    if (eigen_solver.info() != Eigen::Success)
    {
        return target_cloud;
    }

    Eigen::Vector3f local_normal = eigen_solver.eigenvectors().col(0).normalized();
    if (local_normal.z() < 0.0f)
    {
        local_normal = -local_normal;
    }

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(finite_cloud);

    std::vector<float> nearest_distances;
    nearest_distances.reserve(finite_cloud->size());
    for (const auto& point : finite_cloud->points)
    {
        std::vector<int> nn_indices;
        std::vector<float> nn_distances;
        if (kdtree.nearestKSearch(point, 2, nn_indices, nn_distances) >= 2 && nn_distances[1] > 0.0f)
        {
            nearest_distances.push_back(std::sqrt(nn_distances[1]));
        }
    }
    if (nearest_distances.empty())
    {
        return target_cloud;
    }

    const auto median_it = nearest_distances.begin() + static_cast<std::ptrdiff_t>(nearest_distances.size() / 2);
    std::nth_element(nearest_distances.begin(), median_it, nearest_distances.end());
    const double point_resolution = std::max(static_cast<double>(*median_it), 1e-3);
    const double local_eps = point_resolution * 2.5;
    const unsigned int local_min_pts = static_cast<unsigned int>(std::max(2, params_.min_support_points));

    std::vector<float> normal_projections;
    normal_projections.reserve(finite_cloud->size());
    for (const auto& point : finite_cloud->points)
    {
        normal_projections.push_back((Eigen::Vector3f(point.x, point.y, point.z) - centroid).dot(local_normal));
    }

    std::vector<std::size_t> sorted_indices;
    sorted_indices.reserve(finite_cloud->size());
    for (std::size_t point_index = 0; point_index < finite_cloud->size(); ++point_index)
    {
        sorted_indices.push_back(point_index);
    }
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](std::size_t lhs, std::size_t rhs) {
        return normal_projections[lhs] > normal_projections[rhs];
    });

    int keypoint_index = -1;
    for (const auto point_index : sorted_indices)
    {
        std::vector<int> support_indices;
        std::vector<float> support_distances;
        if (kdtree.radiusSearch(finite_cloud->points[point_index], local_eps, support_indices, support_distances) >=
            static_cast<int>(local_min_pts))
        {
            keypoint_index = static_cast<int>(point_index);
            break;
        }
    }
    if (keypoint_index < 0)
    {
        return target_cloud;
    }

    const float min_target_projection =
        normal_projections[static_cast<std::size_t>(keypoint_index)] - params_.height_window;
    for (std::size_t point_index = 0; point_index < finite_cloud->size(); ++point_index)
    {
        if (normal_projections[point_index] < min_target_projection)
        {
            continue;
        }

        std::vector<int> support_indices;
        std::vector<float> support_distances;
        if (kdtree.radiusSearch(finite_cloud->points[point_index], local_eps, support_indices, support_distances) >=
            static_cast<int>(local_min_pts))
        {
            target_cloud->push_back(finite_cloud->points[point_index]);
        }
    }
    if (!target_cloud->empty())
    {
        finalize_target_cloud();
    }
    return target_cloud;

}



void TargetFrontViewController::setParams(const Params& params)
{
    const bool was_enabled = params_.enabled;
    params_ = params;
    params_.accumulation_frames = std::max(1, params_.accumulation_frames);
    params_.patch_radius = std::max(1, params_.patch_radius);
    params_.patch_min_score = std::clamp(params_.patch_min_score, 0.0, 1.0);
    params_.patch_update_min_score = std::clamp(params_.patch_update_min_score, params_.patch_min_score, 1.0);
    params_.patch_learning_rate = std::clamp(params_.patch_learning_rate, 0.0, 1.0);
    params_.patch_train_threshold = std::clamp(params_.patch_train_threshold, 0.0, 1.0);
    params_.patch_train_min_points = std::max(1, params_.patch_train_min_points);
    projector_.setParams(params_.projector);
    if (params_.enabled && !was_enabled)
    {
        cv::namedWindow(params_.front_view_window, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(params_.front_view_window, &TargetFrontViewController::onMouse, this);
        cv::namedWindow(params_.template_window, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(params_.template_window, &TargetFrontViewController::onMouse, this);
        cv::startWindowThread();
    }
}

void TargetFrontViewController::setSelectedPointCallback(SelectedPointCallback callback)
{
    selected_point_callback_ = std::move(callback);
}

void TargetFrontViewController::resetAccumulation()
{
    std::lock_guard<std::mutex> lock(mutex_);
    accumulation_queue_.clear();
}

TargetFrontViewController::Output TargetFrontViewController::show(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target_cloud,
    const std::string& frame_id,
    const ros::Time& stamp)
    {
        Output output;
        if (!params_.enabled || !target_cloud || target_cloud->empty())
        {
            return output;
        }

        pcl::PointCloud<pcl::PointXYZ>::ConstPtr projection_cloud = target_cloud;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const int accumulation_frames = std::max(1, params_.accumulation_frames);
            if (accumulation_frames > 1)
            {
                accumulation_queue_.push_back(
                    PointCloudPtr(new PointCloud(*target_cloud)));
                if (accumulation_queue_.size() < static_cast<std::size_t>(accumulation_frames))
                {
                    return output;
                }

                pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_cloud =
                    PointCloudPtr(new PointCloud);
                accumulated_cloud->header = target_cloud->header;
                for (const auto& queued_cloud : accumulation_queue_)
                {
                    if (queued_cloud && !queued_cloud->empty())
                    {
                        *accumulated_cloud += *queued_cloud;
                    }
                }
                accumulation_queue_.clear();
                accumulated_cloud->width = static_cast<uint32_t>(accumulated_cloud->size());
                accumulated_cloud->height = 1;
                accumulated_cloud->is_dense = false;
                if (accumulated_cloud->empty())
                {
                    return output;
                }
                projection_cloud = accumulated_cloud;
            }
            else
            {
                accumulation_queue_.clear();
            }
        }

        const Eigen::Vector3f sensor_origin = Eigen::Vector3f::Zero();
        if (!projector_.initialize(projection_cloud, sensor_origin))
        {
            return output;
        }

        const auto projection = projector_.project(projection_cloud, sensor_origin);
        if (!projection.valid || projection.image.empty())
        {
            return output;
        }

        if (projection.surface_cloud && !projection.surface_cloud->empty())
        {
            output.projected_surface_cloud = projection.surface_cloud;
        }

        const cv::Mat raw_occupancy_mask = buildRawOccupancyMask(projection.source_index_image);
        const cv::Mat processed_mask = buildProcessedTargetMask(projection.source_index_image);
        const cv::Mat selection_index_image = buildSelectionIndexImage(projection.source_index_image, processed_mask);
        cv::Mat template_display_image = projection.image.clone();
        cv::Mat display_image = projection.image.clone();
        std::vector<std::vector<cv::Point>> processed_contours;
        cv::findContours(processed_mask.clone(), processed_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(display_image, processed_contours, -1, cv::Scalar(0, 180, 0), 1);
        const int fixed_display_width = 640;
        const int fixed_display_height = 480;
        double display_scale = std::min(static_cast<double>(fixed_display_width) /
                                            static_cast<double>(std::max(1, display_image.cols)),
                                        static_cast<double>(fixed_display_height) /
                                            static_cast<double>(std::max(1, display_image.rows)));
        display_scale = std::max(display_scale, 1e-6);
        const int scaled_width = std::max(1, static_cast<int>(std::round(static_cast<double>(display_image.cols) * display_scale)));
        const int scaled_height = std::max(1, static_cast<int>(std::round(static_cast<double>(display_image.rows) * display_scale)));
        const int display_offset_x = (fixed_display_width - scaled_width) / 2;
        const int display_offset_y = (fixed_display_height - scaled_height) / 2;
        cv::Mat resized_display_image;
        cv::resize(display_image, resized_display_image, cv::Size(scaled_width, scaled_height), 0.0, 0.0, cv::INTER_NEAREST);
        cv::Mat fixed_display_image(fixed_display_height, fixed_display_width, display_image.type(), cv::Scalar(0, 0, 0));
        resized_display_image.copyTo(fixed_display_image(cv::Rect(display_offset_x,
                                                                  display_offset_y,
                                                                  scaled_width,
                                                                  scaled_height)));
        display_image = fixed_display_image;

        cv::Mat resized_template_display_image;
        cv::resize(template_display_image,
                   resized_template_display_image,
                   cv::Size(scaled_width, scaled_height),
                   0.0,
                   0.0,
                   cv::INTER_NEAREST);
        cv::Mat fixed_template_display_image(fixed_display_height,
                                             fixed_display_width,
                                             template_display_image.type(),
                                             cv::Scalar(0, 0, 0));
        resized_template_display_image.copyTo(fixed_template_display_image(cv::Rect(display_offset_x,
                                                                                   display_offset_y,
                                                                                   scaled_width,
                                                                                   scaled_height)));
        template_display_image = fixed_template_display_image;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (has_selected_template_display_ && !selected_template_display_image_.empty())
            {
                template_display_image = selected_template_display_image_.clone();
            }
        }

        int selected_row = -1;
        int selected_col = -1;
        int tracked_point_index = -1;
        pcl::PointXYZ tracked_point;
        bool publish_tracked_point = false;
        double match_score = 0.0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_projection_ = projection;
            latest_selection_index_image_ = selection_index_image;
            latest_processed_mask_ = processed_mask;
            latest_raw_occupancy_mask_ = raw_occupancy_mask;
            latest_projection_frame_id_ = frame_id;
            latest_projection_stamp_ = stamp;
            latest_display_image_size_ = display_image.size();
            latest_display_scale_ = display_scale;
            latest_display_offset_x_ = static_cast<double>(display_offset_x);
            latest_display_offset_y_ = static_cast<double>(display_offset_y);

            if (has_selected_patch_ && projection.surface_cloud && !projection.surface_cloud->empty())
            {
                int matched_row = -1;
                int matched_col = -1;
                if (matchSelectedTargetPatch(raw_occupancy_mask, processed_mask, matched_row, matched_col, match_score) &&
                    findNearestProjectedPoint(selection_index_image,
                                              matched_row,
                                              matched_col,
                                              8,
                                              tracked_point_index,
                                              selected_row,
                                              selected_col) &&
                    tracked_point_index >= 0 && tracked_point_index < static_cast<int>(projection.surface_cloud->size()))
                {
                    tracked_point = projection.surface_cloud->points[static_cast<std::size_t>(tracked_point_index)];
                    for (int raw_row = 0; raw_row < projection.source_index_image.rows; ++raw_row)
                    {
                        for (int raw_col = 0; raw_col < projection.source_index_image.cols; ++raw_col)
                        {
                            if (projection.source_index_image.at<int>(raw_row, raw_col) == tracked_point_index)
                            {
                                selected_row = raw_row;
                                selected_col = raw_col;
                                raw_row = projection.source_index_image.rows;
                                break;
                            }
                        }
                    }
                    selected_projection_row_ = selected_row;
                    selected_projection_col_ = selected_col;
                    selected_point_ = tracked_point;
                    has_selected_point_ = true;
                    publish_tracked_point = true;

                    if (match_score >= params_.patch_update_min_score)
                    {
                        const int max_patch_radius = std::max(
                            0,
                            (std::min(raw_occupancy_mask.rows, raw_occupancy_mask.cols) - 1) / 2);
                        const int patch_radius = std::min(params_.patch_radius, max_patch_radius);
                        cv::Mat updated_patch = extractPatchWithBorder(raw_occupancy_mask,
                                                                       selected_row,
                                                                       selected_col,
                                                                       patch_radius);
                        trainSelectedTemplate(updated_patch, patch_radius);
                    }
                }
                else
                {
                    selected_row = selected_projection_row_;
                    selected_col = selected_projection_col_;
                    ROS_WARN_THROTTLE(1.0,
                                          "Selected target point 2D patch tracking lost: score=%.3f pixel=(%d,%d)",
                                          match_score,
                                          selected_col,
                                          selected_row);
                }
            }

            if (!publish_tracked_point && has_selected_point_)
            {
                selected_row = selected_projection_row_;
                selected_col = selected_projection_col_;
            }
        }

        if (publish_tracked_point)
        {
            if (selected_point_callback_)
            {
                selected_point_callback_(tracked_point, frame_id, stamp);
            }
            ROS_INFO_THROTTLE(1.0,
                                  "Tracked selected target point 2D score=%.3f idx=%d pixel=(%d,%d) xyz=(%.3f, %.3f, %.3f)",
                                  match_score,
                                  tracked_point_index,
                                  selected_col,
                                  selected_row,
                                  tracked_point.x,
                                  tracked_point.y,
                                  tracked_point.z);
        }

        if (selected_row >= 0 && selected_col >= 0 && selected_row < projection.image.rows && selected_col < projection.image.cols)
        {
            int marker_index = -1;
            int marker_row = selected_row;
            int marker_col = selected_col;
            if (findNearestProjectedPoint(projection.source_index_image,
                                          selected_row,
                                          selected_col,
                                          8,
                                          marker_index,
                                          marker_row,
                                          marker_col))
            {
                const int display_x = display_offset_x +
                                      static_cast<int>(std::round((static_cast<double>(marker_col) + 0.5) * display_scale));
                const int display_y = display_offset_y +
                                      static_cast<int>(std::round((static_cast<double>(marker_row) + 0.5) * display_scale));
                cv::circle(display_image, cv::Point(display_x, display_y), 4, cv::Scalar(0, 0, 255), cv::FILLED);
                cv::drawMarker(display_image,
                               cv::Point(display_x, display_y),
                               cv::Scalar(0, 0, 255),
                               cv::MARKER_CROSS,
                               12,
                               2);
            }
        }

        cv::imshow(params_.template_window, template_display_image);

        cv::putText(display_image,
                    "angular grid: click to select",
                    cv::Point(8, 20),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(0, 255, 0),
                    1);
        cv::imshow(params_.front_view_window, display_image);
        cv::waitKey(1);
        return output;
    }

    void TargetFrontViewController::onMouse(int event, int x, int y, int /*flags*/, void* userdata)
    {
        if (event != cv::EVENT_LBUTTONDOWN || !userdata)
        {
            return;
        }

        auto* controller = static_cast<TargetFrontViewController*>(userdata);
        controller->handleClick(x, y);
    }

    void TargetFrontViewController::handleClick(int display_x, int display_y)
    {
        pcl::PointXYZ selected_point;
        std::string frame_id;
        ros::Time stamp;
        int selected_row = -1;
        int selected_col = -1;
        int point_index = -1;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!latest_projection_.valid || latest_selection_index_image_.empty() ||
                latest_processed_mask_.empty() || latest_raw_occupancy_mask_.empty() ||
                !latest_projection_.surface_cloud ||
                latest_projection_.surface_cloud->empty())
            {
                ROS_WARN_THROTTLE(1.0, "No target projection is available for point selection");
                return;
            }

            const double display_scale = std::max(latest_display_scale_, 1e-6);
            const double display_offset_x = latest_display_offset_x_;
            const double display_offset_y = latest_display_offset_y_;

            selected_click_display_x_ = std::clamp(display_x,
                                                    0,
                                                    std::max(0, latest_display_image_size_.width - 1));
            selected_click_display_y_ = std::clamp(display_y,
                                                    0,
                                                    std::max(0, latest_display_image_size_.height - 1));
            has_selected_click_display_ = true;

            selected_template_display_image_.release();
            has_selected_template_display_ = false;

            double best_distance_sq = std::numeric_limits<double>::infinity();
            for (int source_row = 0; source_row < latest_projection_.source_index_image.rows; ++source_row)
            {
                for (int source_col = 0; source_col < latest_projection_.source_index_image.cols; ++source_col)
                {
                    const int candidate_index = latest_projection_.source_index_image.at<int>(source_row, source_col);
                    if (candidate_index < 0)
                    {
                        continue;
                    }

                    const double candidate_x = display_offset_x + (static_cast<double>(source_col) + 0.5) * display_scale;
                    const double candidate_y = display_offset_y + (static_cast<double>(source_row) + 0.5) * display_scale;
                    const double dx = candidate_x - static_cast<double>(display_x);
                    const double dy = candidate_y - static_cast<double>(display_y);
                    const double distance_sq = dx * dx + dy * dy;
                    if (distance_sq < best_distance_sq)
                    {
                        best_distance_sq = distance_sq;
                        point_index = candidate_index;
                        selected_row = source_row;
                        selected_col = source_col;
                    }
                }
            }

            if (point_index < 0)
            {
                ROS_WARN("No raw projected target point is available for click selection");
                return;
            }

            if (point_index < 0 || point_index >= static_cast<int>(latest_projection_.surface_cloud->size()))
            {
                ROS_WARN("Selected projection index %d is outside surface cloud size %zu",
                             point_index,
                             latest_projection_.surface_cloud->size());
                return;
            }

            selected_point = latest_projection_.surface_cloud->points[static_cast<std::size_t>(point_index)];
            frame_id = latest_projection_frame_id_;
            stamp = latest_projection_stamp_;
            selected_projection_row_ = selected_row;
            selected_projection_col_ = selected_col;
            selected_point_ = selected_point;
            const int max_patch_radius = std::max(
                0,
                (std::min(latest_raw_occupancy_mask_.rows, latest_raw_occupancy_mask_.cols) - 1) / 2);
            const int patch_radius = std::min(params_.patch_radius, max_patch_radius);
            selected_patch_ = extractPatchWithBorder(latest_raw_occupancy_mask_,
                                                            selected_row,
                                                            selected_col,
                                                            patch_radius);
            initializeSelectedTemplate(selected_patch_, patch_radius);
            has_selected_point_ = true;
            has_selected_patch_ = !selected_template_points_.empty();
            if (!has_selected_patch_)
            {
                ROS_WARN("Failed to initialize selected target point sparse 2D template");
            }
        }

        {
            selected_template_display_image_ = buildTemplateDisplayImage(selected_patch_);
            has_selected_template_display_ = !selected_template_display_image_.empty();
        }

        if (selected_point_callback_) { selected_point_callback_(selected_point, frame_id, stamp); }
        ROS_INFO("Selected target point idx=%d click=(%d,%d) pixel=(%d,%d) patch=%dx%d xyz=(%.3f, %.3f, %.3f)",
                     point_index,
                     display_x,
                     display_y,
                     selected_col,
                     selected_row,
                     selected_patch_.cols,
                     selected_patch_.rows,
                     selected_point.x,
                     selected_point.y,
                     selected_point.z);
    }

    bool TargetFrontViewController::findNearestProjectedPoint(const cv::Mat& source_index_image,
                                                  int row,
                                                  int col,
                                                  int search_radius,
                                                  int& point_index,
                                                  int& nearest_row,
                                                  int& nearest_col) const
    {
        if (source_index_image.empty() || source_index_image.type() != CV_32SC1)
        {
            return false;
        }

        search_radius = std::max(0, search_radius);
        int best_distance_sq = std::numeric_limits<int>::max();
        int best_index = -1;
        int best_row = -1;
        int best_col = -1;

        for (int delta_row = -search_radius; delta_row <= search_radius; ++delta_row)
        {
            const int candidate_row = row + delta_row;
            if (candidate_row < 0 || candidate_row >= source_index_image.rows)
            {
                continue;
            }

            for (int delta_col = -search_radius; delta_col <= search_radius; ++delta_col)
            {
                const int candidate_col = col + delta_col;
                if (candidate_col < 0 || candidate_col >= source_index_image.cols)
                {
                    continue;
                }

                const int candidate_index = source_index_image.at<int>(candidate_row, candidate_col);
                if (candidate_index < 0)
                {
                    continue;
                }

                const int distance_sq = delta_row * delta_row + delta_col * delta_col;
                if (distance_sq < best_distance_sq)
                {
                    best_distance_sq = distance_sq;
                    best_index = candidate_index;
                    best_row = candidate_row;
                    best_col = candidate_col;
                }
            }
        }

        if (best_index < 0)
        {
            return false;
        }

        point_index = best_index;
        nearest_row = best_row;
        nearest_col = best_col;
        return true;
    }

    cv::Mat TargetFrontViewController::buildProcessedTargetMask(const cv::Mat& source_index_image) const
    {
        if (source_index_image.empty() || source_index_image.type() != CV_32SC1)
        {
            return cv::Mat();
        }

        cv::Mat mask(source_index_image.rows, source_index_image.cols, CV_8UC1, cv::Scalar(0));
        std::vector<cv::Point> occupied_pixels;
        occupied_pixels.reserve(static_cast<std::size_t>(source_index_image.rows * source_index_image.cols));
        for (int row = 0; row < source_index_image.rows; ++row)
        {
            for (int col = 0; col < source_index_image.cols; ++col)
            {
                if (source_index_image.at<int>(row, col) >= 0)
                {
                    mask.at<unsigned char>(row, col) = 255;
                    occupied_pixels.emplace_back(col, row);
                }
            }
        }

        if (occupied_pixels.empty())
        {
            return mask;
        }

        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        if (occupied_pixels.size() < 3)
        {
            cv::Mat sparse_mask;
            cv::dilate(mask, sparse_mask, kernel, cv::Point(-1, -1), 1);
            return sparse_mask;
        }

        std::vector<cv::Point> hull_pixels;
        cv::convexHull(occupied_pixels, hull_pixels);
        cv::Mat hull_mask(source_index_image.rows, source_index_image.cols, CV_8UC1, cv::Scalar(0));
        cv::fillConvexPoly(hull_mask, hull_pixels, cv::Scalar(255));
        cv::morphologyEx(hull_mask, hull_mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);
        return hull_mask;
    }

    cv::Mat TargetFrontViewController::buildRawOccupancyMask(const cv::Mat& source_index_image) const
    {
        if (source_index_image.empty() || source_index_image.type() != CV_32SC1)
        {
            return cv::Mat();
        }

        cv::Mat mask(source_index_image.rows, source_index_image.cols, CV_8UC1, cv::Scalar(0));
        for (int row = 0; row < source_index_image.rows; ++row)
        {
            for (int col = 0; col < source_index_image.cols; ++col)
            {
                if (source_index_image.at<int>(row, col) >= 0)
                {
                    mask.at<unsigned char>(row, col) = 255;
                }
            }
        }
        return mask;
    }

    cv::Mat TargetFrontViewController::buildSelectionIndexImage(const cv::Mat& source_index_image,
                                                    const cv::Mat& processed_mask) const
    {
        cv::Mat selection_index_image(source_index_image.rows, source_index_image.cols, CV_32SC1, cv::Scalar(-1));
        if (source_index_image.empty() || source_index_image.type() != CV_32SC1 || processed_mask.empty())
        {
            return selection_index_image;
        }

        std::queue<cv::Point> frontier;
        for (int row = 0; row < source_index_image.rows; ++row)
        {
            for (int col = 0; col < source_index_image.cols; ++col)
            {
                const int source_index = source_index_image.at<int>(row, col);
                if (source_index >= 0 && processed_mask.at<unsigned char>(row, col) > 0)
                {
                    selection_index_image.at<int>(row, col) = source_index;
                    frontier.emplace(col, row);
                }
            }
        }

        const std::array<cv::Point, 8> neighbor_offsets = {
            cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1), cv::Point(-1, 0),
            cv::Point(1, 0),   cv::Point(-1, 1), cv::Point(0, 1),  cv::Point(1, 1)};

        while (!frontier.empty())
        {
            const cv::Point current = frontier.front();
            frontier.pop();
            const int current_index = selection_index_image.at<int>(current.y, current.x);
            for (const auto& offset : neighbor_offsets)
            {
                const cv::Point neighbor = current + offset;
                if (neighbor.x < 0 || neighbor.y < 0 || neighbor.x >= processed_mask.cols || neighbor.y >= processed_mask.rows)
                {
                    continue;
                }
                if (processed_mask.at<unsigned char>(neighbor.y, neighbor.x) == 0 ||
                    selection_index_image.at<int>(neighbor.y, neighbor.x) >= 0)
                {
                    continue;
                }
                selection_index_image.at<int>(neighbor.y, neighbor.x) = current_index;
                frontier.push(neighbor);
            }
        }

        return selection_index_image;
    }

    cv::Mat TargetFrontViewController::extractPatchWithBorder(const cv::Mat& image, int center_row, int center_col, int radius) const
    {
        if (image.empty() || radius < 0)
        {
            return cv::Mat();
        }

        const int patch_size = radius * 2 + 1;
        cv::Mat padded_image;
        cv::copyMakeBorder(image, padded_image, radius, radius, radius, radius, cv::BORDER_CONSTANT, cv::Scalar(0));
        const cv::Rect patch_rect(center_col, center_row, patch_size, patch_size);
        if (patch_rect.x < 0 || patch_rect.y < 0 || patch_rect.x + patch_rect.width > padded_image.cols ||
            patch_rect.y + patch_rect.height > padded_image.rows)
        {
            return cv::Mat();
        }
        return padded_image(patch_rect).clone();
    }

    cv::Mat TargetFrontViewController::buildTemplateDisplayImage(const cv::Mat& patch) const
    {
        if (patch.empty())
        {
            return cv::Mat();
        }

        cv::Mat display_patch;
        if (patch.channels() == 1)
        {
            cv::cvtColor(patch, display_patch, cv::COLOR_GRAY2BGR);
        }
        else
        {
            display_patch = patch.clone();
        }

        const int scale = std::max(1, 480 / std::max(display_patch.rows, display_patch.cols));
        cv::Mat scaled_patch;
        cv::resize(display_patch,
                   scaled_patch,
                   cv::Size(display_patch.cols * scale, display_patch.rows * scale),
                   0.0,
                   0.0,
                   cv::INTER_NEAREST);
        return scaled_patch;
    }

    void TargetFrontViewController::initializeSelectedTemplate(const cv::Mat& patch, int patch_radius)
    {
        selected_template_points_.clear();
        selected_template_model_.release();

        if (patch.empty())
        {
            selected_patch_.release();
            selected_template_display_image_.release();
            has_selected_template_display_ = false;
            has_selected_patch_ = false;
            return;
        }

        cv::Mat binary_patch;
        cv::threshold(patch, binary_patch, 0, 255, cv::THRESH_BINARY);
        binary_patch.convertTo(selected_template_model_, CV_32FC1, 1.0 / 255.0);

        selected_patch_ = binary_patch.clone();
        for (int patch_row = 0; patch_row < selected_patch_.rows; ++patch_row)
        {
            for (int patch_col = 0; patch_col < selected_patch_.cols; ++patch_col)
            {
                if (selected_patch_.at<unsigned char>(patch_row, patch_col) == 0)
                {
                    continue;
                }
                selected_template_points_.emplace_back(patch_col - patch_radius, patch_row - patch_radius);
            }
        }

        has_selected_patch_ = !selected_template_points_.empty();
        selected_template_display_image_ = buildTemplateDisplayImage(selected_patch_);
        has_selected_template_display_ = !selected_template_display_image_.empty();
    }

    bool TargetFrontViewController::trainSelectedTemplate(const cv::Mat& patch, int patch_radius)
    {
        if (patch.empty())
        {
            return false;
        }

        if (selected_template_model_.empty() || selected_template_model_.size() != patch.size())
        {
            initializeSelectedTemplate(patch, patch_radius);
            return has_selected_patch_;
        }

        cv::Mat binary_patch;
        cv::threshold(patch, binary_patch, 0, 255, cv::THRESH_BINARY);
        cv::Mat observation;
        binary_patch.convertTo(observation, CV_32FC1, 1.0 / 255.0);

        if (params_.patch_learning_rate > 0.0)
        {
            cv::addWeighted(selected_template_model_,
                            1.0 - params_.patch_learning_rate,
                            observation,
                            params_.patch_learning_rate,
                            0.0,
                            selected_template_model_);
        }

        cv::Mat trained_patch;
        cv::threshold(selected_template_model_, trained_patch, params_.patch_train_threshold, 255.0, cv::THRESH_BINARY);
        trained_patch.convertTo(trained_patch, CV_8UC1);

        std::vector<cv::Point> trained_template_points;
        for (int patch_row = 0; patch_row < trained_patch.rows; ++patch_row)
        {
            for (int patch_col = 0; patch_col < trained_patch.cols; ++patch_col)
            {
                if (trained_patch.at<unsigned char>(patch_row, patch_col) == 0)
                {
                    continue;
                }
                trained_template_points.emplace_back(patch_col - patch_radius, patch_row - patch_radius);
            }
        }

        if (static_cast<int>(trained_template_points.size()) < params_.patch_train_min_points)
        {
            return false;
        }

        selected_patch_ = trained_patch;
        selected_template_points_ = std::move(trained_template_points);
        selected_template_display_image_ = buildTemplateDisplayImage(selected_patch_);
        has_selected_template_display_ = !selected_template_display_image_.empty();
        has_selected_patch_ = true;
        return true;
    }

    bool TargetFrontViewController::matchSelectedTargetPatch(const cv::Mat& raw_occupancy_mask,
                                                 const cv::Mat& processed_mask,
                                                 int& matched_row,
                                                 int& matched_col,
                                                 double& match_score) const
    {
        match_score = 0.0;
        if (raw_occupancy_mask.empty() || processed_mask.empty() || selected_template_points_.empty() ||
            raw_occupancy_mask.rows != processed_mask.rows || raw_occupancy_mask.cols != processed_mask.cols)
        {
            return false;
        }

        cv::Mat distance_source;
        cv::threshold(raw_occupancy_mask, distance_source, 0, 255, cv::THRESH_BINARY_INV);
        cv::Mat distance_image;
        cv::distanceTransform(distance_source, distance_image, cv::DIST_L2, 3);

        const float max_point_distance = static_cast<float>(std::max(2, std::min(8, params_.patch_radius / 2)));
        double best_score = 0.0;
        int best_row = -1;
        int best_col = -1;
        const bool has_previous_position = selected_projection_row_ >= 0 && selected_projection_col_ >= 0;

        for (int candidate_row = 0; candidate_row < raw_occupancy_mask.rows; ++candidate_row)
        {
            for (int candidate_col = 0; candidate_col < raw_occupancy_mask.cols; ++candidate_col)
            {
                if (processed_mask.at<unsigned char>(candidate_row, candidate_col) == 0 ||
                    raw_occupancy_mask.at<unsigned char>(candidate_row, candidate_col) == 0)
                {
                    continue;
                }

                float distance_sum = 0.0F;
                int valid_points = 0;
                for (const auto& template_point : selected_template_points_)
                {
                    const int projected_col = candidate_col + template_point.x;
                    const int projected_row = candidate_row + template_point.y;
                    if (projected_row < 0 || projected_col < 0 ||
                        projected_row >= distance_image.rows || projected_col >= distance_image.cols)
                    {
                        distance_sum += max_point_distance;
                        ++valid_points;
                        continue;
                    }

                    const float point_distance = distance_image.at<float>(projected_row, projected_col);
                    distance_sum += std::min(point_distance, max_point_distance);
                    ++valid_points;
                }

                if (valid_points == 0)
                {
                    continue;
                }

                const float mean_distance = distance_sum / static_cast<float>(valid_points);
                const double candidate_score = std::max(0.0, 1.0 - static_cast<double>(mean_distance / max_point_distance));
                bool is_better = candidate_score > best_score;
                if (!is_better && std::abs(candidate_score - best_score) < 0.03 && has_previous_position)
                {
                    const int current_row_offset = candidate_row - selected_projection_row_;
                    const int current_col_offset = candidate_col - selected_projection_col_;
                    const int best_row_offset = best_row - selected_projection_row_;
                    const int best_col_offset = best_col - selected_projection_col_;
                    const int current_distance_sq = current_row_offset * current_row_offset + current_col_offset * current_col_offset;
                    const int best_distance_sq = best_row_offset * best_row_offset + best_col_offset * best_col_offset;
                    is_better = current_distance_sq < best_distance_sq;
                }

                if (is_better)
                {
                    best_score = candidate_score;
                    best_row = candidate_row;
                    best_col = candidate_col;
                }
            }
        }

        match_score = best_score;
        if (match_score < params_.patch_min_score)
        {
            return false;
        }

        matched_col = std::clamp(best_col, 0, processed_mask.cols - 1);
        matched_row = std::clamp(best_row, 0, processed_mask.rows - 1);
        return true;
    }

}  // namespace rm_radar_lidar_detector
