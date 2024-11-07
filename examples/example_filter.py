import argparse
import time

import numpy as np
from lidsor_filter import filtering_lidsor_cpp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter point cloud")
    parser.add_argument(
        "--num_points",
        type=int,
        default=10000,
        help="Number of points in the point cloud",
    )
    parser.add_argument(
        "--outliers",
        type=int,
        default=50,
        help="Number of outliers in the point cloud",
    )
    args = parser.parse_args()

    # Generate random point cloud data (-1 to 4 range)
    num_points = args.num_points
    points = np.random.uniform(-1, 4, size=(num_points, 3))

    # Add some outliers
    num_outliers = args.outliers
    outliers = np.random.uniform(5, 10, size=(num_outliers, 3))
    point_cloud = np.vstack([points, outliers])
    # Filter parameters
    k_neighbors = 30
    std_dev_threshold = 2.0

    # Apply LIDSOR filter
    start_time = time.time()
    filtered_points, kept_indices, removed_indices = filtering_lidsor_cpp(
        point_cloud, k=k_neighbors, s=std_dev_threshold
    )
    end_time = time.time()
    print(f"Original points: {point_cloud.shape[0]}")
    print(f"Filtered points: {len(kept_indices)}")
    print(f"Removed points: {len(removed_indices)}")
    print(f"Time taken: {end_time - start_time:.6f} seconds")
