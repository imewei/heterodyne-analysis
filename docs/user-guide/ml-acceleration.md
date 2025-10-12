# ML Acceleration and Performance Optimization

## Overview

The heterodyne-analysis package provides machine learning acceleration for parameter
optimization through intelligent initial value prediction. This guide covers ML
acceleration features, Numba JIT compilation, and performance optimization strategies
for analyzing XPCS data efficiently.

## ML Acceleration

### Purpose

ML acceleration improves optimization convergence by:

- **Smart parameter initialization**: Predicting good starting values based on
  experimental conditions
- **Faster convergence**: 2-5x reduction in optimization time through better initial
  guesses
- **Continuous learning**: Models improve as you analyze more datasets

### How It Works

The ML system uses ensemble learning to predict optimal parameters based on:

- Wavevector magnitude (q)
- Time step (dt)
- Geometric parameters (gap size, frame range)
- Historical optimization results

## Python 3.12+ Compatibility

The heterodyne package requires **Python 3.12 or later** for optimal performance:

```bash
# Check Python version
python --version  # Should show 3.12 or higher

# Create Python 3.12 environment
conda create -n heterodyne python=3.12
conda activate heterodyne

# Install package
pip install heterodyne-analysis[all]
```

**Compatibility Notes:**

- **Numba JIT**: Requires Python 3.12+ for latest performance optimizations
- **Type hints**: Uses modern Python type annotation features
- **Dependencies**: NumPy 1.24+, SciPy 1.9+ optimized for Python 3.12+

## Numba JIT Compilation

### Performance Benefits

Numba provides 3-5x speedup for core computational kernels through just-in-time (JIT)
compilation:

| Operation | Pure Python | With Numba JIT | Speedup |
|-----------|-------------|----------------|---------| | 100 points | 2.3 s | 0.7 s |
3.3× | | 500 points | 12.1 s | 3.2 s | 3.8× | | 1000 points | 45.2 s | 8.9 s | 5.1× | |
5000 points | 892 s | 178 s | 5.0× |

### Automatic JIT Compilation

Numba JIT is enabled by default for all optimized kernels:

```python
from heterodyne.analysis.core import HeterodyneAnalysisCore
import json

# Load configuration
with open("my_config.json", 'r') as f:
    config = json.load(f)

# JIT compilation happens automatically on first call
core = HeterodyneAnalysisCore(config)

# First run includes compilation overhead (~1-2s)
# Subsequent runs use compiled code (fast)
```

### Manual Kernel Warmup

For production workflows, pre-compile kernels to eliminate JIT overhead:

```python
from heterodyne.core.kernels import warmup_numba_kernels

# Warmup all computational kernels (one-time cost)
warmup_results = warmup_numba_kernels()

print(f"Kernels compiled in {warmup_results['total_warmup_time']:.3f}s")
# Kernels compiled in 1.245s

# Now all subsequent analyses run at full speed immediately
```

### Configuration Control

Control Numba behavior in your configuration file:

```json
{
  "performance_settings": {
    "numba_optimization": {
      "enable_numba": true,
      "warmup_numba": true,
      "parallel_numba": true,
      "cache_numba": true,
      "stability_enhancements": {
        "enable_kernel_warmup": true,
        "warmup_iterations": 5,
        "optimize_memory_layout": true,
        "environment_optimization": {
          "auto_configure": true,
          "max_threads": 4
        }
      }
    }
  }
}
```

## Memory Usage Considerations

### Large Dataset Handling

For datasets exceeding 4 million data points, careful memory management is essential:

**Dataset Size Guidelines:**

- **\<100k points**: No special configuration needed
- **100k-1M points**: Enable time subsampling (4x-8x reduction)
- **1M-4M points**: Enable aggressive subsampling (16x-32x reduction)
- **>4M points**: Not recommended - over-subsampling effects reduce accuracy

### Memory Optimization Strategies

**1. Subsampling Configuration:**

```json
{
  "optimization_config": {
    "classical_optimization": {
      "subsampling": {
        "enabled": true,
        "max_data_points": 100000,
        "time_subsample_factor": 4,
        "angle_subsample_factor": 2
      }
    }
  }
}
```

**Impact:**

- 2000×2000 dataset with 2 angles = 8M points
- With factors (4, 2): 8M → ~250k points (32x reduction)
- Speedup: 20-50x faster with \<10% chi-squared degradation

**2. Ellipsoidal Optimization Memory Limit:**

For robust optimization with very large datasets:

```python
from heterodyne.optimization.robust import RobustHeterodyneOptimizer

# Configure memory limit for ellipsoidal method
config['optimization_config']['robust_optimization']['memory_limit'] = 0.9  # 90% limit

robust = RobustHeterodyneOptimizer(core, config)
results = robust.optimize(
    phi_angles=phi_angles,
    c2_experimental=c2_data,
    method="ellipsoidal"
)
```

**3. Low Memory Mode:**

For memory-constrained environments:

```json
{
  "performance_settings": {
    "memory_management": {
      "low_memory_mode": true,
      "garbage_collection_frequency": 5,
      "memory_monitoring": true
    },
    "caching": {
      "cache_size_limit_mb": 500
    }
  }
}
```

### Memory Benchmarks

| Dataset Size | Memory Usage (Default) | Memory Usage (Optimized) | Improvement |
|--------------|------------------------|--------------------------|-------------| | 4M
points | 85% RAM | 75% RAM | 12% reduction | | 8M points | Memory error | 90% limit
success | Enabled large datasets |

## Performance Monitoring

### Built-in Performance Tracking

Monitor optimization performance:

```python
from heterodyne.core.config import performance_monitor

# Time-specific operations
with performance_monitor.time_function("data_loading"):
    c2_data, time_length, phi_angles, num_angles = load_xpcs_data(config_file)

with performance_monitor.time_function("classical_optimization"):
    params, results = classical.run_classical_optimization_optimized(
        phi_angles=phi_angles,
        c2_experimental=c2_data
    )

# Get performance summary
stats = performance_monitor.get_timing_summary()
print(f"Data loading: {stats['data_loading']:.3f}s")
print(f"Optimization: {stats['classical_optimization']:.3f}s")
```

### Stable Benchmarking

For research and reproducibility:

```python
from heterodyne.core.profiler import stable_benchmark, adaptive_stable_benchmark

def my_optimization():
    return classical.run_classical_optimization_optimized(
        phi_angles=phi_angles,
        c2_experimental=c2_data
    )

# Standard benchmarking with outlier filtering
results = stable_benchmark(my_optimization, warmup_runs=5, measurement_runs=15)
cv = results['std'] / results['mean']
print(f"Mean time: {results['mean']:.4f}s ± {cv:.3f} CV")

# Adaptive benchmarking (automatically finds optimal measurement count)
results = adaptive_stable_benchmark(my_optimization, target_cv=0.10)
print(f"Achieved {results['cv']:.3f} CV in {results['total_runs']} runs")
```

**Performance Stability Achievements:**

- 97% reduction in chi-squared calculation variability (CV < 0.31)
- Balanced optimization settings for numerical stability
- Conservative threading (max 4 cores) for consistent results

## Environment Optimization

### Thread Configuration

Set optimal threading for your hardware:

```bash
# Recommended for most systems
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4

# For high-core systems (>16 cores)
export OMP_NUM_THREADS=8
export NUMBA_NUM_THREADS=8

# Run analysis
heterodyne --config my_config.json --method classical
```

### Performance Mode

Enable high-performance mode:

```bash
# Set environment variable
export HETERODYNE_PERFORMANCE_MODE=1

# With Numba parallel optimization
export NUMBA_ENABLE_CUDASIM=0
export NUMBA_DISABLE_JIT=0

# Run analysis
heterodyne --config my_config.json --method all
```

## Troubleshooting Performance Issues

### Slow Optimization

**Problem:** Optimization takes too long (>30 minutes)

**Solutions:**

1. **Enable subsampling:**

   ```json
   {
     "optimization_config": {
       "classical_optimization": {
         "subsampling": {
           "enabled": true,
           "time_subsample_factor": 4,
           "angle_subsample_factor": 2
         }
       }
     }
   }
   ```

2. **Reduce active parameters:** Start with 4-6 essential parameters instead of all 14:

   ```json
   {
     "initial_parameters": {
       "active_parameters": ["D0_ref", "D0_sample", "v0", "f0"]
     }
   }
   ```

3. **Use angle filtering:**

   ```json
   {
     "optimization_config": {
       "angle_filtering": {
         "enabled": true,
         "target_ranges": [
           {"min_angle": -10.0, "max_angle": 10.0},
           {"min_angle": 170.0, "max_angle": 190.0}
         ]
       }
     }
   }
   ```

### Memory Errors

**Problem:** "MemoryError" or system slowdown

**Solutions:**

1. **Enable aggressive subsampling:**

   - Increase `time_subsample_factor` to 8 or 16
   - Increase `angle_subsample_factor` to 4
   - Lower `max_data_points` threshold

2. **Use low memory mode:** Set `low_memory_mode: true` in performance settings

3. **Reduce cache size:** Lower `cache_size_limit_mb` to 500 or less

### Numba Compilation Errors

**Problem:** Numba fails to compile kernels

**Solutions:**

1. **Verify Python version:**

   ```bash
   python --version  # Should be 3.12+
   ```

2. **Update Numba:**

   ```bash
   pip install --upgrade numba
   ```

3. **Disable Numba temporarily:**

   ```json
   {
     "performance_settings": {
       "numba_optimization": {
         "enable_numba": false
       }
     }
   }
   ```

## Best Practices

### For Maximum Performance

1. **Use Python 3.12+**: Latest optimizations and type hints
2. **Enable Numba JIT**: 3-5x speedup for core calculations
3. **Warmup kernels**: Pre-compile for production workflows
4. **Configure threading**: Match OMP_NUM_THREADS to your CPU
5. **Enable subsampling**: For datasets >100k points
6. **Use angle filtering**: 3-5x speedup for anisotropic analysis

### For Maximum Accuracy

1. **Disable subsampling**: For small datasets (\<100k points)
2. **Use all parameters**: Optimize all 14 parameters when data quality permits
3. **Multiple optimization methods**: Run both classical and robust
4. **Increase iteration limits**: For complex parameter spaces
5. **Higher precision**: Use `float64` data type

### For Large Datasets (>1M points)

1. **Mandatory subsampling**: Set `time_subsample_factor` ≥ 4
2. **Lower memory limits**: Set `cache_size_limit_mb` ≤ 1000
3. **Conservative threading**: Use 4-8 threads maximum
4. **Monitor memory**: Enable `memory_monitoring: true`
5. **Consider robust ellipsoidal**: Uses 90% memory limit

## Performance Reference

### Typical Analysis Times

| Dataset Size | Configuration | Time (Classical) | Time (Robust) |
|--------------|---------------|------------------|---------------| | 100k points |
Default | 2-5 min | 5-10 min | | 500k points | With subsampling | 5-10 min | 10-20 min |
| 1M points | Aggressive subsampling | 10-15 min | 20-30 min | | 4M points | Maximum
optimization | 20-30 min | 40-60 min |

**Hardware:** Intel Xeon 8-core, 32GB RAM

### Expected Speedups

| Optimization | Speedup Factor | Notes | |--------------|----------------|-------| |
Numba JIT | 3-5× | Automatic for core kernels | | Subsampling (4×) | 16-20× | \<10%
chi-squared degradation | | Angle filtering | 3-5× | For anisotropic analysis | |
Combined | 50-100× | All optimizations enabled |

## Next Steps

- See [Configuration Guide](configuration.rst) for detailed parameter settings
- Review [API Reference](../api-reference/analysis-core.md) for programmatic access
- Explore [Examples](examples.rst) for real-world workflows
