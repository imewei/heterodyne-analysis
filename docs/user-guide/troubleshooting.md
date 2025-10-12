# Troubleshooting Guide

## Common Issues and Solutions

This guide provides solutions to frequently encountered problems when using the
heterodyne-analysis package.

## Import and Module Errors

### ModuleNotFoundError: No module named 'heterodyne'

**Problem:**

```python
>>> from heterodyne.analysis.core import HeterodyneAnalysisCore
ModuleNotFoundError: No module named 'heterodyne'
```

**Solutions:**

1. **Verify installation:**

   ```bash
   pip list | grep heterodyne
   # Should show: heterodyne-analysis  x.x.x
   ```

2. **Reinstall package:**

   ```bash
   pip install --upgrade heterodyne-analysis[all]
   ```

3. **Check Python environment:**

   ```bash
   which python
   which pip
   # Ensure using correct environment
   ```

4. **Activate correct environment:**

   ```bash
   conda activate heterodyne
   # or
   source heterodyne-env/bin/activate
   ```

### ImportError: cannot import name 'load_xpcs_data'

**Problem:**

```python
from heterodyne.data.xpcs_loader import load_xpcs_data
ImportError: cannot import name 'load_xpcs_data' from 'heterodyne.data.xpcs_loader'
```

**Solutions:**

1. **Verify correct import path:**

   ```python
   # Correct import
   from heterodyne.data.xpcs_loader import load_xpcs_data

   # Usage
   c2_data, time_length, phi_angles, num_angles = load_xpcs_data(config_file)
   ```

2. **Check package version:**

   ```bash
   python -c "import heterodyne; print(heterodyne.__version__)"
   # Should be 1.0.0 or later
   ```

3. **Update to latest version:**

   ```bash
   pip install --upgrade heterodyne-analysis
   ```

### Module has no attribute Error

**Problem:**

```python
AttributeError: module 'heterodyne.analysis.core' has no attribute 'HeterodyneAnalysisCore'
```

**Solutions:**

1. **Verify class name:**

   ```python
   # Correct usage
   from heterodyne.analysis.core import HeterodyneAnalysisCore
   core = HeterodyneAnalysisCore(config)
   ```

2. **Check for circular imports:**

   ```python
   # Don't name your file 'heterodyne.py' - conflicts with package
   # Rename: mv heterodyne.py my_analysis.py
   ```

3. **Clear Python cache:**

   ```bash
   find . -type d -name "__pycache__" -exec rm -rf {} +
   python -c "import heterodyne"  # Re-import
   ```

## Frame Counting Issues

### Incorrect time_length Calculation

**Problem:** Frame counting produces unexpected results:

```python
start_frame = 401
end_frame = 1000
# Expecting 600 frames, getting 599
```

**Solution:**

Use the **1-based inclusive** counting convention:

```python
time_length = end_frame - start_frame + 1  # Correct: 1000 - 401 + 1 = 600
```

**Correct Examples:**

- `start_frame=1, end_frame=100` → `time_length=100` (not 99)
- `start_frame=401, end_frame=1000` → `time_length=600` (not 599)
- `start_frame=1, end_frame=1` → `time_length=1` (single frame)

**Use Utility Functions:**

```python
from heterodyne.core.io_utils import calculate_time_length, config_frames_to_python_slice

# Calculate time_length from config frames
time_length = calculate_time_length(start_frame=401, end_frame=1000)
print(f"Time length: {time_length}")  # Output: 600

# Convert for Python array slicing
python_start, python_end = config_frames_to_python_slice(401, 1000)
# Returns: (400, 1000) for use in data[400:1000]
```

### Cache File Dimension Mismatch

**Problem:**

```
WARNING: Cached data dimension (599) doesn't match config time_length (600)
```

**Solution:**

The package automatically adjusts dimensions:

```python
# Automatic adjustment happens in HeterodyneAnalysisCore
if c2_experimental.shape[1] != self.time_length:
    logger.info(f"Auto-adjusting time_length to match cached data")
    self.time_length = c2_experimental.shape[1]
```

**Manual fix:**

```bash
# Delete old cache files
rm data/cached_c2_isotropic_frames_*.npz

# Regenerate with correct frame counting
heterodyne --config my_config.json --method classical
```

## Data Format Issues

### File Not Found Errors

**Problem:**

```
FileNotFoundError: [Errno 2] No such file or directory: './data/correlation_data.h5'
```

**Solutions:**

1. **Verify file paths:**

   ```bash
   ls -lh data/correlation_data.h5
   ls -lh data/phi_angles.txt
   ```

2. **Use absolute paths in config:**

   ```json
   {
     "experimental_data": {
       "data_folder_path": "/absolute/path/to/data/",
       "data_file_name": "correlation_data.h5"
     }
   }
   ```

3. **Check working directory:**

   ```bash
   pwd
   # Ensure you're in the correct directory
   cd /path/to/project
   ```

### HDF5 Format Incompatibility

**Problem:**

```
OSError: Unable to open file (file signature not found)
```

**Solutions:**

1. **Verify HDF5 file integrity:**

   ```bash
   h5dump -H data/correlation_data.h5
   # Should show file structure
   ```

2. **Check file format:**

   ```python
   import h5py
   with h5py.File("data/correlation_data.h5", 'r') as f:
       print(list(f.keys()))
       # Should show dataset keys
   ```

3. **Convert to NumPy format:**

   ```python
   import numpy as np
   import h5py

   # Load from HDF5
   with h5py.File("data/correlation_data.h5", 'r') as f:
       c2_data = f['exchange/c2_isotropic_average'][:]

   # Save as NPZ
   np.savez_compressed("data/correlation_data.npz", c2_data=c2_data)

   # Update config to use NPZ file
   ```

### Invalid Angle File Format

**Problem:**

```
ValueError: could not convert string to float
```

**Solution:**

Ensure `phi_angles.txt` contains one angle per line:

```text
0.0
45.0
90.0
135.0
180.0
```

**Verify format:**

```bash
cat data/phi_angles.txt
# Should show clean numeric values

# Remove extra whitespace/characters
sed 's/[^0-9.\-]//g' data/phi_angles.txt > data/phi_angles_clean.txt
```

## Optimization Convergence Issues

### Optimization Fails to Converge

**Problem:**

```
Warning: Optimization did not converge (maxiter exceeded)
Chi-squared: 1.234e+05 (very high)
```

**Solutions:**

1. **Improve initial parameter values:**

   ```json
   {
     "initial_parameters": {
       "values": [
         10000.0, -1.5, 0.0,    // Better D0_ref, alpha_ref, D_offset_ref
         10000.0, -1.5, 0.0,    // Better D0_sample, alpha_sample, D_offset_sample
         1.0, 0.0, 0.0,         // Better v0, beta, v_offset
         0.5, 0.0, 50.0, 0.3,   // Better f0, f1, f2, f3
         0.0                     // phi0
       ]
     }
   }
   ```

2. **Reduce number of active parameters:**

   ```json
   {
     "initial_parameters": {
       "active_parameters": ["D0_ref", "D0_sample", "v0", "f0"]
     }
   }
   ```

3. **Increase iteration limit:**

   ```json
   {
     "optimization_config": {
       "classical_optimization": {
         "method_options": {
           "Nelder-Mead": {
             "maxiter": 20000,
             "fatol": 1e-8,
             "xatol": 1e-8
           }
         }
       }
     }
   }
   ```

4. **Try robust optimization:**

   ```bash
   heterodyne --config my_config.json --method robust
   ```

### High Chi-Squared Values

**Problem:**

```
Chi-squared: 5.234e+02 (poor fit)
```

**Solutions:**

1. **Check data quality:**

   ```bash
   heterodyne --config my_config.json --plot-experimental-data
   # Inspect plots for anomalies
   ```

2. **Adjust parameter bounds:**

   ```json
   {
     "parameter_space": {
       "bounds": [
         {"name": "D0_ref", "min": 1.0, "max": 1000000.0},
         {"name": "alpha_ref", "min": -2.0, "max": 2.0}
       ]
     }
   }
   ```

3. **Add more active parameters:**

   ```json
   {
     "initial_parameters": {
       "active_parameters": [
         "D0_ref", "alpha_ref", "D_offset_ref",
         "D0_sample", "alpha_sample", "D_offset_sample",
         "v0", "beta", "f0", "f1"
       ]
     }
   }
   ```

4. **Use all optimization methods:**

   ```bash
   heterodyne --config my_config.json --method all --verbose
   ```

### Parameter Values at Bounds

**Problem:**

```
Warning: Parameters D0_ref, v0 at bounds
```

**Solution:**

Widen parameter bounds:

```json
{
  "parameter_space": {
    "bounds": [
      {
        "name": "D0_ref",
        "min": 0.1,        // Lowered from 1.0
        "max": 10000000.0  // Raised from 1000000.0
      },
      {
        "name": "v0",
        "min": 1e-6,       // Lowered from 1e-5
        "max": 100.0       // Raised from 10.0
      }
    ]
  }
}
```

## Memory and Performance Issues

### Out of Memory Error

**Problem:**

```
MemoryError: Unable to allocate 8.00 GiB for an array
```

**Solutions:**

1. **Enable subsampling:**

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

2. **Use low memory mode:**

   ```json
   {
     "performance_settings": {
       "memory_management": {
         "low_memory_mode": true,
         "memory_monitoring": true
       },
       "caching": {
         "cache_size_limit_mb": 500
       }
     }
   }
   ```

3. **Reduce frame range:**

   ```json
   {
     "analyzer_parameters": {
       "temporal": {
         "start_frame": 500,   // Reduce analysis window
         "end_frame": 1000
       }
     }
   }
   ```

### Slow Optimization Performance

**Problem:** Optimization takes >1 hour for moderate dataset

**Solutions:**

1. **Enable aggressive subsampling:**

   ```json
   {
     "optimization_config": {
       "classical_optimization": {
         "subsampling": {
           "enabled": true,
           "time_subsample_factor": 8,
           "angle_subsample_factor": 4
         }
       }
     }
   }
   ```

2. **Use angle filtering:**

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

3. **Optimize threading:**

   ```bash
   export OMP_NUM_THREADS=4
   export NUMBA_NUM_THREADS=4
   heterodyne --config my_config.json --method classical
   ```

4. **Reduce active parameters:** Start with essential parameters only

### Numba Compilation Warnings

**Problem:**

```
NumbaWarning: Cannot cache compiled function as it uses lifted loops
```

**Solution:**

These warnings are informational and don't affect results:

```python
# Suppress warnings (optional)
import warnings
from numba.core.errors import NumbaWarning
warnings.simplefilter('ignore', category=NumbaWarning)
```

Or disable Numba caching:

```json
{
  "performance_settings": {
    "numba_optimization": {
      "cache_numba": false
    }
  }
}
```

## Configuration Issues

### JSON Syntax Errors

**Problem:**

```
json.decoder.JSONDecodeError: Expecting ',' delimiter: line 15 column 5
```

**Solutions:**

1. **Validate JSON syntax:**

   ```bash
   python -m json.tool my_config.json
   # Reports syntax errors
   ```

2. **Common JSON mistakes:**

   ```json
   // WRONG: Trailing comma
   {
     "key1": "value1",
     "key2": "value2",  // Remove this comma
   }

   // CORRECT:
   {
     "key1": "value1",
     "key2": "value2"
   }
   ```

3. **Use JSON linter:**

   ```bash
   # Online: jsonlint.com
   # Or install jq
   cat my_config.json | jq
   ```

### Missing Required Configuration Fields

**Problem:**

```
KeyError: 'experimental_data'
```

**Solution:**

Ensure all required fields are present:

```json
{
  "metadata": { "config_version": "1.0.0", "analysis_mode": "heterodyne" },
  "experimental_data": { "data_folder_path": "...", "data_file_name": "..." },
  "analyzer_parameters": { "temporal": {...}, "scattering": {...} },
  "initial_parameters": { "parameter_names": [...], "values": [...] },
  "parameter_space": { "bounds": [...] }
}
```

**Or regenerate from template:**

```bash
heterodyne-config --sample my_sample --output new_config.json
# Copy your custom values to new_config.json
```

## Runtime Warnings and Errors

### Singular Matrix Warning

**Problem:**

```
LinAlgWarning: Ill-conditioned matrix (singular matrix detected)
```

**Solutions:**

1. **Improve parameter scaling:**

   ```json
   {
     "advanced_settings": {
       "optimization_controls": {
         "parameter_scaling": "auto"
       }
     }
   }
   ```

2. **Add regularization:**

   ```json
   {
     "optimization_config": {
       "robust_optimization": {
         "regularization_alpha": 0.02,
         "regularization_beta": 0.001
       }
     }
   }
   ```

3. **Use robust optimization:**

   ```bash
   heterodyne --config my_config.json --method robust
   ```

### Division by Zero Warning

**Problem:**

```
RuntimeWarning: divide by zero encountered in true_divide
```

**Solutions:**

1. **Check for zero values in data:**

   ```python
   import numpy as np
   c2_data = np.load("data/correlation_data.npz")['c2_data']
   print(f"Zero values: {np.sum(c2_data == 0)}")
   print(f"NaN values: {np.sum(np.isnan(c2_data))}")
   ```

2. **Enable data validation:**

   ```json
   {
     "validation_rules": {
       "data_quality": {
         "check_data_range": true,
         "check_nan_values": true,
         "nan_handling": "raise"
       }
     }
   }
   ```

3. **Set minimum thresholds:**

   ```json
   {
     "advanced_settings": {
       "chi_squared_calculation": {
         "minimum_sigma": 1e-10
       }
     }
   }
   ```

## Troubleshooting Workflow

### Systematic Debugging

1. **Enable verbose logging:**

   ```bash
   heterodyne --config my_config.json --method classical --verbose
   ```

2. **Check log file:**

   ```bash
   tail -f heterodyne_results/run.log
   ```

3. **Validate data loading:**

   ```bash
   heterodyne --config my_config.json --plot-experimental-data
   ```

4. **Test with minimal configuration:**

   ```json
   {
     "initial_parameters": {
       "active_parameters": ["D0_ref", "D0_sample"]
     }
   }
   ```

5. **Compare methods:**

   ```bash
   heterodyne --config my_config.json --method all
   ```

### Reporting Issues

When reporting problems, include:

1. **Package version:**

   ```bash
   python -c "import heterodyne; print(heterodyne.__version__)"
   ```

2. **Python version:**

   ```bash
   python --version
   ```

3. **Error traceback:**

   ```bash
   heterodyne --config my_config.json --method classical --verbose 2>&1 | tee error.log
   ```

4. **Configuration file:**

   ```bash
   # Sanitize sensitive paths, then share
   cat my_config.json
   ```

5. **System information:**

   ```bash
   uname -a  # Linux/macOS
   # or
   systeminfo  # Windows
   ```

## Getting Additional Help

### Resources

- **Documentation**: https://heterodyne-analysis.readthedocs.io/
- **GitHub Issues**: https://github.com/imewei/heterodyne_analysis/issues
- **Examples**: See [examples.rst](examples.rst)
- **API Reference**: See [../api-reference/overview.md](../api-reference/overview.md)

### Before Asking for Help

1. Check this troubleshooting guide
2. Review error messages carefully
3. Try minimal working example
4. Verify Python and package versions
5. Check GitHub issues for similar problems

### When Opening an Issue

Provide:

- Clear problem description
- Minimal reproducible example
- Full error traceback
- Package and Python versions
- Steps already attempted
