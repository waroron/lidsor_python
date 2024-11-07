# LIDSOR Filter

Fast Outlier Removal Filter for LiDAR Point Clouds (C++ Implementation)

## Requirements

- CMake (>= 3.10)
- C++14 compliant compiler
- Python 3.x
- pybind11
- nanoflann (included as a submodule)

## Installation(from pip)
```bash
pip install git+https://github.com/waroron/lidsor_python.git

# In case of using patchworkpp
git clone https://github.com/url-kaist/patchwork-plusplus.git
cd patchwork-plusplus
make pyinstall

```

## Installation(Local build)
### 1. Clone the repository
```bash
git clone https://github.com/your-username/lidsor_filter.git
cd lidsor_filter
```

### 2. Initialize submodules
```bash
git submodule init
git submodule update
```

### 3. Install pybind11

For Ubuntu/Debian:
```bash
sudo apt-get install python3-pybind11
```
Or using pip:
```bash
pip install pybind11
```

### 4. Build

```bash
mkdir build
cd build
cmake ..
make
```

### 5. Install (optional)
```bash
sudo make install
```

## Usage

Python example:
```python
import numpy as np
import time
from lidsor_filter import filtering_lidsor_cpp

# Generate random point cloud data (-1 to 4 range)
num_points = 10000
points = np.random.uniform(-1, 4, size=(num_points, 3))

# Add some outliers
num_outliers = 50
outliers = np.random.uniform(5, 10, size=(num_outliers, 3))
point_cloud = np.vstack([points, outliers])
# Filter parameters
k_neighbors = 30
std_dev_threshold = 2.0

# Apply LIDSOR filter
start_time = time.time()
filtered_points = filtering_lidsor_cpp(
    point_cloud,
    k=k_neighbors,
    s=std_dev_threshold
)
end_time = time.time()
print(f"Original points: {point_cloud.shape[0]}")
print(f"Filtered points: {np.asarray(filtered_points).shape[0]}")
print(f"Time taken: {end_time - start_time:.6f} seconds")
```

