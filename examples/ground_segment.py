import argparse
import time

import numpy as np
import pypatchworkpp
from lidsor_filter import filtering_lidsor_cpp


def segment_and_filter_points(points, scaling_factor=110.0):
    """
    Segment ground points and filter non-ground points using LIDSOR.

    Args:
        points (np.ndarray): Input point cloud data of shape (N, 4) containing x, y, z, intensity
        scaling_factor (float): Scaling factor for intensity values

    Returns:
        np.ndarray: Merged point cloud containing filtered ground and non-ground points
    """
    # Ground point estimation
    params = pypatchworkpp.Parameters()
    patchwork = pypatchworkpp.patchworkpp(params)
    patchwork.estimateGround(points)

    # Get ground and non-ground indices
    ground_idx = patchwork.getGroundIndices()
    nonground_idx = patchwork.getNongroundIndices()

    # Split points
    ground_points = points[ground_idx]
    nonground_points = points[nonground_idx]

    # Filter non-ground points
    filtered_points, kept_indices, removed_indices = filtering_lidsor_cpp(
        nonground_points, scaling_factor=scaling_factor
    )
    filtered_points = np.array(filtered_points, dtype=np.float64)

    # Merge ground and filtered non-ground points
    merged_points = np.concatenate([ground_points, filtered_points], axis=0)

    return merged_points


# ... existing code ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment and filter point cloud")
    parser.add_argument(
        "--num_points",
        type=int,
        default=10000,
        help="Number of points in the point cloud",
    )
    args = parser.parse_args()

    # Generate random point cloud data (1000 points)
    np.random.seed(42)  # For reproducibility
    num_points = args.num_points
    random_points = np.zeros((num_points, 4))
    random_points[:, 0] = np.random.uniform(-50, 50, num_points)  # x coordinates
    random_points[:, 1] = np.random.uniform(-50, 50, num_points)  # y coordinates
    random_points[:, 2] = np.random.uniform(-2, 5, num_points)  # z coordinates
    random_points[:, 3] = np.random.uniform(0, 100, num_points)  # intensity

    # Measure processing time
    start_time = time.time()
    result = segment_and_filter_points(random_points)
    processing_time = time.time() - start_time

    # Display results
    print(f"Input point cloud size: {len(random_points)}")
    print(f"Output point cloud size: {len(result)}")
    print(f"Processing time: {processing_time:.3f} seconds")

    # Point cloud statistics
    print("\nPoint cloud statistics:")
    print(f"X range: [{result[:, 0].min():.2f}, {result[:, 0].max():.2f}]")
    print(f"Y range: [{result[:, 1].min():.2f}, {result[:, 1].max():.2f}]")
    print(f"Z range: [{result[:, 2].min():.2f}, {result[:, 2].max():.2f}]")
    print(f"Intensity range: [{result[:, 3].min():.2f}, {result[:, 3].max():.2f}]")
