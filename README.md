# LIDSOR Filter

Fast Outlier Removal Filter for LiDAR Point Clouds (C++ Implementation)

## Requirements

- CMake (>= 3.10)
- C++14 compliant compiler
- Python 3.x
- pybind11
- nanoflann (included as a submodule)

## Installation
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
from lidsor_filter import filtering_lidsor_cpp
```

