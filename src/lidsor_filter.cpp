#include "lidsor_filter.hpp"

using namespace nanoflann;

struct PointCloud {
    std::vector<std::array<float, 3>> points;
    
    inline size_t kdtree_get_point_count() const { return points.size(); }
    
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return points[idx][dim];
    }
    
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

std::vector<std::array<float, 4>> filtering_lidsor_cpp(
    const py::array_t<float>& input_points,
    int k,
    float s,
    float i_threshold,
    float d_threshold,
    float scaling_factor
) {
    auto points_buf = input_points.unchecked<2>();
    const size_t num_points = points_buf.shape(0);
    
    // 点群データの準備
    PointCloud cloud;
    std::vector<float> intensities;
    cloud.points.reserve(num_points);
    intensities.reserve(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        cloud.points.push_back({
            points_buf(i, 0),
            points_buf(i, 1),
            points_buf(i, 2)
        });
        float intensity = std::abs(points_buf(i, 3));
        if (intensity <= 1.0f) intensity *= scaling_factor;
        intensities.push_back(intensity);
    }
    
    // KD木の構築
    using KDTree = KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<float, PointCloud>,
        PointCloud,
        3
    >;
    
    KDTree index(3, cloud, KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();
    
    // 各点の平均距離を計算
    std::vector<float> mean_distances(num_points);
    std::vector<float> distances(k);
    std::vector<unsigned int> indices(k);  // ここを修正
    
    for (size_t i = 0; i < num_points; ++i) {
        const float query_pt[3] = {
            cloud.points[i][0],
            cloud.points[i][1],
            cloud.points[i][2]
        };
        
        index.knnSearch(query_pt, k, indices.data(), distances.data());
        
        float sum = 0.0f;
        for (int j = 1; j < k; ++j) {  // j=1から開始して自身を除外
            sum += std::sqrt(distances[j]);
        }
        mean_distances[i] = sum / (k - 1);
    }
    
    // 統計値の計算
    float mean_dist = 0.0f;
    for (float d : mean_distances) mean_dist += d;
    mean_dist /= num_points;
    
    float std_dist = 0.0f;
    for (float d : mean_distances) {
        float diff = d - mean_dist;
        std_dist += diff * diff;
    }
    std_dist = std::sqrt(std_dist / num_points);
    
    float threshold_distance = mean_dist + s * std_dist;
    
    // フィルタリング
    std::vector<std::array<float, 4>> filtered_points;  // 4次元配列に変更
    filtered_points.reserve(num_points);
    
    for (size_t i = 0; i < num_points; ++i) {
        if (mean_distances[i] < threshold_distance && 
            mean_distances[i] < d_threshold && 
            intensities[i] > i_threshold) {
            filtered_points.push_back({
                cloud.points[i][0],
                cloud.points[i][1],
                cloud.points[i][2],
                intensities[i]  // intensityを追加
            });
        }
    }
    
    
    return filtered_points;
}