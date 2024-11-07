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
    filtered_points = np.array(
        filtering_lidsor_cpp(nonground_points, scaling_factor=scaling_factor), 
        dtype=np.float64
    )
    
    # Merge ground and filtered non-ground points
    merged_points = np.concatenate([ground_points, filtered_points], axis=0)
    
    return merged_points