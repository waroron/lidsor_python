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

std::tuple<std::vector<std::array<float, 4>>, std::vector<size_t>, std::vector<size_t>> filtering_lidsor_cpp(
    const py::array_t<float>& input_points,
    int k_neighbors,           
    float std_multiplier,         
    float range_multiplier,         
    float intensity_threshold,
    float distance_threshold,
    float scaling_factor
){
    auto points_buf = input_points.unchecked<2>();
    const size_t num_points = points_buf.shape(0);
    
    // 一次フィルタリング用の点群Q と 最終出力用の点群F の準備
    PointCloud Q;
    std::vector<std::array<float, 4>> F;
    std::vector<size_t> kept_indices, removed_indices;
    std::vector<size_t> Q_indices;  // Qに保存された点のインデックスを記録
    
    // Step 1: Primary filtering based on distance
    for (size_t i = 0; i < num_points; ++i) {
        float x = points_buf(i, 0);
        float y = points_buf(i, 1);
        float z = points_buf(i, 2);
        
        // distance = √(xi² + yi² + zi²)
        float distance = std::sqrt(x*x + y*y + z*z);
        
        if (distance < distance_threshold) {
            Q.points.push_back({x, y, z});
            Q_indices.push_back(i);
        } else {
            // 距離閾値外の点は直接Fに保存
            F.push_back({x, y, z, points_buf(i, 3)});
            kept_indices.push_back(i);
        }
    }
    
    // Step 2: KD-tree construction and k-nearest neighbor search
    using KDTree = KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<float, PointCloud>,
        PointCloud,
        3
    >;
    
    KDTree index(3, Q, KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();
    
    // 各点の平均距離を計算
    std::vector<float> D(Q.points.size());  // mean distances D
    std::vector<float> distances(k_neighbors);
    std::vector<unsigned int> indices(k_neighbors);
    
    for (size_t i = 0; i < Q.points.size(); ++i) {
        const float query_pt[3] = {
            Q.points[i][0],
            Q.points[i][1],
            Q.points[i][2]
        };
        
        index.knnSearch(query_pt, k_neighbors, indices.data(), distances.data());
        
        float sum = 0.0f;
        for (int j = 1; j < k_neighbors; ++j) {
            sum += std::sqrt(distances[j]);
        }
        D[i] = sum / (k_neighbors - 1);
    }
    
    // Step 3: Calculate statistical parameters
    // mean μ ← D
    float mu = 0.0f;
    for (float d : D) mu += d;
    mu /= D.size();
    
    // standard deviation σ ← D
    float sigma = 0.0f;
    for (float d : D) {
        float diff = d - mu;
        sigma += diff * diff;
    }
    sigma = std::sqrt(sigma / D.size());
    
    // global threshold Tg ← μ + σ × s
    float Tg = mu + sigma * std_multiplier;
    
    // Step 4: Dynamic threshold filtering
    for (size_t i = 0; i < Q.points.size(); ++i) {
        float distance = std::sqrt(
            Q.points[i][0] * Q.points[i][0] + 
            Q.points[i][1] * Q.points[i][1] + 
            Q.points[i][2] * Q.points[i][2]
        );
        
        // dynamic threshold Td ← Tg × range_multiplier × distance
        float Td = Tg * range_multiplier * distance;
        
        size_t original_idx = Q_indices[i];
        float intensity = scaling_factor * std::abs(points_buf(original_idx, 3));
        
        if (D[i] <= Td || intensity >= intensity_threshold) {
        // if (D[i] <= Td || intensity <= intensity_threshold) {
            F.push_back({
                Q.points[i][0],
                Q.points[i][1],
                Q.points[i][2],
                points_buf(original_idx, 3)
            });
            kept_indices.push_back(original_idx);
        } else {
            removed_indices.push_back(original_idx);
        }
    }
    
    return std::make_tuple(F, kept_indices, removed_indices);
}
