# Heterodyne Analysis API Documentation

## Overview

This directory contains comprehensive API documentation for the heterodyne-analysis
package. The documentation covers all public modules, classes, and functions with
detailed parameter descriptions, return values, usage examples, and implementation
notes.

## Module Organization

### Core Analysis

- **[analysis/core.md](analysis/core.md)**: `HeterodyneAnalysisCore` - Main analysis
  engine
- **[core/io_utils.md](core/io_utils.md)**: I/O utilities and frame counting functions
- **[core/kernels.md](core/kernels.md)**: Computational kernels for correlation
  functions
- **[core/config.md](core/config.md)**: Configuration management and validation

### Optimization

- **[optimization/classical.md](optimization/classical.md)**: `ClassicalOptimizer` -
  Nelder-Mead and Gurobi methods
- **[optimization/robust.md](optimization/robust.md)**: `RobustHeterodyneOptimizer` -
  DRO, scenario-based, ellipsoidal
- **[optimization/utils.md](optimization/utils.md)**: Optimization utility functions

### Data Loading

- **[data/xpcs_loader.md](data/xpcs_loader.md)**: APS and APS-U format data loading

### Command-Line Interface

- **[cli/run_heterodyne.md](cli/run_heterodyne.md)**: Main analysis command
- **[cli/create_config.md](cli/create_config.md)**: Configuration generator
- **[cli/core.md](cli/core.md)**: CLI core functionality

### Visualization

- **[visualization/plotting.md](visualization/plotting.md)**: Core plotting functions
- **[visualization/enhanced_plotting.md](visualization/enhanced_plotting.md)**: Advanced
  visualization

### Performance

- **[performance/monitoring.md](performance/monitoring.md)**: Performance tracking and
  benchmarking
- **[performance/baseline.md](performance/baseline.md)**: Performance baseline
  management

## Quick Start

### Basic Analysis Workflow

```python
from heterodyne.analysis.core import HeterodyneAnalysisCore
from heterodyne.optimization.classical import ClassicalOptimizer
import json
import numpy as np

# Initialize analysis core
config_file = "config_heterodyne.json"
core = HeterodyneAnalysisCore(config_file)

# Load config for optimizer
with open(config_file, 'r') as f:
    config = json.load(f)

# Load experimental data
phi_angles = np.array([0, 36, 72, 108, 144])
c2_data = core.load_experimental_data(phi_angles, len(phi_angles))

# Run optimization (uses 14-parameter heterodyne model)
optimizer = ClassicalOptimizer(core, config)
params, results = optimizer.run_classical_optimization_optimized(
    phi_angles=phi_angles,
    c2_experimental=c2_data
)

# Extract 14 parameters
D0_ref, alpha_ref, D_offset_ref = params[0:3]
D0_sample, alpha_sample, D_offset_sample = params[3:6]
v0, beta, v_offset = params[6:9]
f0, f1, f2, f3 = params[9:13]
phi0 = params[13]

print(f"Reference D₀: {D0_ref:.3e} Å²/s")
print(f"Sample D₀: {D0_sample:.3e} Å²/s")
print(f"Chi-squared: {results.fun:.6e}")
```

### Frame Counting Utilities

```python
from heterodyne.core.io_utils import calculate_time_length, config_frames_to_python_slice

# Calculate time_length from config frames
time_length = calculate_time_length(start_frame=401, end_frame=1000)
# Returns: 600 (not 599!)

# Convert for Python array slicing
python_start, python_end = config_frames_to_python_slice(401, 1000)
# Returns: (400, 1000)

# Use in data slicing
data_slice = full_data[:, python_start:python_end, python_start:python_end]
```

### Robust Optimization

```python
from heterodyne.optimization.robust import RobustHeterodyneOptimizer
import numpy as np

# Initialize robust optimizer with 14-parameter heterodyne model
robust = RobustHeterodyneOptimizer(core, config)

# Initial parameters for 14-parameter heterodyne model
initial_params = np.array([
    100.0, -0.5, 10.0,       # D0_ref, alpha_ref, D_offset_ref
    100.0, -0.5, 10.0,       # D0_sample, alpha_sample, D_offset_sample
    0.1, 0.0, 0.01,          # v0, beta, v_offset
    0.5, 0.0, 50.0, 0.3,     # f0, f1, f2, f3
    0.0                      # phi0
])

# Wasserstein DRO (Distributionally Robust Optimization)
params_dro, info_dro = robust.run_robust_optimization(
    initial_parameters=initial_params,
    phi_angles=phi_angles,
    c2_experimental=c2_data,
    method="wasserstein",
    uncertainty_radius=0.05  # 5% uncertainty radius
)

# Scenario-based optimization
params_scenario, info_scenario = robust.run_robust_optimization(
    initial_parameters=initial_params,
    phi_angles=phi_angles,
    c2_experimental=c2_data,
    method="scenario",
    n_scenarios=15
)

# Ellipsoidal uncertainty
params_ellip, info_ellip = robust.run_robust_optimization(
    initial_parameters=initial_params,
    phi_angles=phi_angles,
    c2_experimental=c2_data,
    method="ellipsoidal"
)

# Access results
print(f"DRO final chi²: {info_dro['final_chi_squared']:.6e}")
print(f"Scenario method: {info_scenario['method']}")
```

### Data Loading (APS and APS-U)

```python
from heterodyne.data.xpcs_loader import load_xpcs_data, load_xpcs_config

# Load configuration from file
config = load_xpcs_config("config.json")
print(f"Loaded config with {len(config)} settings")

# Load XPCS data (auto-detects APS vs APS-U format)
# Returns: (c2_experimental, n_times, phi_angles, n_angles)
c2_experimental, n_times, phi_angles, n_angles = load_xpcs_data("config.json")

# Access loaded data
print(f"Data shape: {c2_experimental.shape}")  # (n_angles, n_times, n_times)
print(f"Number of angles: {n_angles}")
print(f"Number of time points: {n_times}")
print(f"Phi angles: {phi_angles}")
```

## API Conventions

### Naming Conventions

- **Classes**: PascalCase (e.g., `HeterodyneAnalysisCore`, `ClassicalOptimizer`)
- **Functions**: snake_case (e.g., `calculate_time_length`, `load_experimental_data`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_TOLERANCE`, `MAX_ITERATIONS`)
- **Private methods**: Leading underscore (e.g., `_internal_helper`)

### Parameter Types

All functions use type hints for clarity:

```python
def calculate_time_length(start_frame: int, end_frame: int) -> int:
    """Calculate time_length from frame range."""
    return end_frame - start_frame + 1

def config_frames_to_python_slice(
    start_frame: int,
    end_frame: int
) -> tuple[int, int]:
    """Convert config frames to Python slice indices."""
    return start_frame - 1, end_frame
```

### Return Value Conventions

- **Single values**: Return directly
- **Multiple related values**: Return tuple (documented order)
- **Complex results**: Return dictionary with named keys
- **Arrays**: Return NumPy arrays with documented shape

Example:

```python
# Dictionary return for complex results
results = {
    'chi_squared': 1.234e-5,
    'parameters': optimal_params,
    'convergence': convergence_info,
    'uncertainty': uncertainty_bounds
}
```

### Error Handling

All public functions provide comprehensive error handling:

```python
from heterodyne.core.config import ConfigurationError
from heterodyne.data.xpcs_loader import XPCSDataFormatError

try:
    data = load_xpcs_data(config)
except XPCSDataFormatError as e:
    logger.error(f"Data format error: {e}")
    raise
except FileNotFoundError as e:
    logger.error(f"Data file not found: {e}")
    raise
```

## Frame Counting Convention

### Critical Concept

The package uses **1-based inclusive** frame counting in configuration files:

```python
# Config convention (1-based, inclusive)
start_frame = 401  # Start at frame 401
end_frame = 1000   # Include frame 1000
time_length = 600  # Total frames: 1000 - 401 + 1

# Python convention (0-based, exclusive)
python_start = 400  # start_frame - 1
python_end = 1000   # end_frame (unchanged for exclusive slice)

# Array slicing
data[python_start:python_end]  # Gives exactly 600 frames
```

### Utility Functions

```python
from heterodyne.core.io_utils import calculate_time_length, config_frames_to_python_slice

# Always use these utilities for consistency
time_length = calculate_time_length(start_frame, end_frame)
python_start, python_end = config_frames_to_python_slice(start_frame, end_frame)
```

See [core/io_utils.md](core/io_utils.md) for detailed documentation.

## Conditional Angle Subsampling

### Strategy

When the number of angles is small (< 4), preserve all angles:

```python
# In ClassicalOptimizer and RobustHeterodyneOptimizer
if n_angles < 4:
    # Preserve all angles for angular information
    angle_subsample_size = n_angles
else:
    # Subsample for performance
    angle_subsample_size = config.get("n_angles", 4)
```

### Impact

- **Before**: 2 angles → 1 angle (50% loss)
- **After**: 2 angles → 2 angles (100% preservation)
- Time subsampling still applied for ~16x performance improvement

See [optimization/classical.md](optimization/classical.md) and
[optimization/robust.md](optimization/robust.md) for implementation details.

## Performance Optimization

### Memory Limits

Robust optimization includes memory monitoring:

```python
from heterodyne.core.security_performance import monitor_memory

@monitor_memory(max_usage_percent=90.0)  # 90% limit for large datasets
def optimize_ellipsoidal_robust(...):
    # Optimization code
    pass
```

### Numba JIT Compilation

Core calculations use Numba for performance:

```python
import numba as nb

@nb.jit(nopython=True)
def calculate_correlation_numba(time_array, D0, alpha, D_offset):
    """JIT-compiled correlation calculation."""
    # 3-5x speedup over pure Python
    pass
```

### Vectorization

All core operations are vectorized:

```python
# Vectorized chi-squared calculation
residuals = c2_experimental - c2_theoretical
chi_squared = np.sum(residuals**2)
```

## Testing

### Running API Tests

```bash
# Test specific modules
pytest heterodyne/tests/test_analysis_core.py -v
pytest heterodyne/tests/test_time_length_calculation.py -v
pytest heterodyne/tests/test_classical_robust_methods.py -v

# Test with coverage
pytest -v --cov=heterodyne.analysis --cov=heterodyne.optimization

# Performance tests
pytest -v -m performance
```

### Example Tests

See individual module documentation for comprehensive test examples.

## See Also

- **[User Guide](../user-guide/)**: Usage examples and tutorials
- **[Research Documentation](../research/)**: Theoretical framework and validation
- **[Developer Guide](../developer-guide/)**: Contributing and development

## Support

- **GitHub Issues**: https://github.com/imewei/heterodyne/issues
- **Email**: wchen@anl.gov (Wei Chen, Principal Investigator)
- **Documentation**: This API reference

______________________________________________________________________

**Last Updated**: 2025-10-01 **Version**: 1.0.0 **Authors**: Wei Chen, Hongrui He
(Argonne National Laboratory)
