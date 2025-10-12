# Getting Started with Heterodyne Analysis

## Complete Working Example: From Installation to Results

This guide provides a complete, executable workflow for analyzing heterodyne XPCS data using the heterodyne-analysis package.

## Prerequisites

- Python 3.12 or later
- Your XPCS experimental data files
- Basic understanding of X-ray photon correlation spectroscopy

## Step 1: Installation

### Create Python Environment

```bash
# Create isolated environment with Python 3.12+
conda create -n heterodyne python=3.12
conda activate heterodyne

# Or use venv
python3.12 -m venv heterodyne-env
source heterodyne-env/bin/activate  # On Windows: heterodyne-env\Scripts\activate
```

### Install Package

```bash
# Install with all dependencies (recommended)
pip install heterodyne-analysis[all]

# Verify installation
heterodyne --help
heterodyne-config --help
python -c "import heterodyne; print(heterodyne.__version__)"
```

**Optional Dependencies:**

```bash
# For Gurobi optimization (requires license)
pip install heterodyne-analysis[gurobi]

# For development
pip install heterodyne-analysis[dev]
```

## Step 2: Prepare Your Data

### Required Data Files

1. **Correlation data file**: Two-time correlation function `c2(q, φ, t1, t2)`
   - Format: HDF5 (`.h5`, `.hdf`) or NumPy (`.npz`)
   - Shape: `(n_angles, time_length, time_length)`

2. **Angle file**: Scattering angles φ
   - Format: Plain text file
   - Content: One angle per line in degrees

### Example Data Structure

```
my_experiment/
├── data/
│   ├── correlation_data.h5          # Main XPCS data
│   └── phi_angles.txt                # Scattering angles
└── configs/
    └── my_config.json                # Configuration (we'll create this)
```

**Example `phi_angles.txt`:**
```
0.0
45.0
90.0
135.0
180.0
```

## Step 3: Create Configuration File

### Generate Template

```bash
# Create 14-parameter heterodyne configuration
heterodyne-config --sample my_sample --output my_config.json

# Review the template
less my_config.json
```

### Customize Configuration

Edit the generated `my_config.json` file with your experimental parameters:

```json
{
  "metadata": {
    "config_version": "1.0.0",
    "analysis_mode": "heterodyne",
    "description": "My heterodyne XPCS analysis"
  },
  "experimental_data": {
    "data_folder_path": "./data/my_experiment/",
    "data_file_name": "correlation_data.h5",
    "phi_angles_path": "./data/my_experiment/",
    "phi_angles_file": "phi_angles.txt"
  },
  "analyzer_parameters": {
    "temporal": {
      "dt": 0.1,
      "start_frame": 100,
      "end_frame": 1000
    },
    "scattering": {
      "wavevector_q": 0.0045
    },
    "geometry": {
      "stator_rotor_gap": 50000.0
    }
  },
  "initial_parameters": {
    "parameter_names": [
      "D0_ref", "alpha_ref", "D_offset_ref",
      "D0_sample", "alpha_sample", "D_offset_sample",
      "v0", "beta", "v_offset",
      "f0", "f1", "f2", "f3",
      "phi0"
    ],
    "values": [
      1000.0, -0.5, 100.0,
      1000.0, -0.5, 100.0,
      0.01, 0.5, 0.001,
      0.5, 0.0, 50.0, 0.3,
      0.0
    ],
    "active_parameters": ["D0_ref", "D0_sample", "v0", "f0"]
  }
}
```

**Key Parameters to Customize:**

- `dt`: Time step between frames (seconds)
- `wavevector_q`: Scattering wavevector q = 4π sin(θ/2)/λ (Å⁻¹)
- `stator_rotor_gap`: Geometric gap size (Å)
- `start_frame`, `end_frame`: Frame range for analysis (1-based, inclusive)
- `initial_parameters.values`: Starting values for optimization
- `active_parameters`: Which parameters to optimize (start with 4-6)

## Step 4: Validate Data Loading

Before running full optimization, verify your data loads correctly:

```bash
# Plot experimental data for validation
heterodyne --config my_config.json --plot-experimental-data

# Check output in ./heterodyne_results/exp_data/
ls -lh heterodyne_results/exp_data/
```

**Expected outputs:**
- `c2_heatmap_experimental_phi_*.png`: Correlation function heatmaps
- Data validation messages in console

## Step 5: Run Analysis

### Basic Classical Optimization

```bash
# Run classical optimization (Nelder-Mead)
heterodyne --config my_config.json --method classical --verbose

# Results saved to ./heterodyne_results/
```

**Expected runtime:** 2-10 minutes depending on dataset size

### View Results

```bash
# Main results summary
cat heterodyne_results/heterodyne_analysis_results.json

# Classical method results
cat heterodyne_results/classical/nelder_mead/analysis_results_nelder_mead.json

# View fitted parameters
cat heterodyne_results/classical/nelder_mead/parameters.json
```

### Robust Optimization (for Noisy Data)

```bash
# Run robust methods for noise-resistant fitting
heterodyne --config my_config.json --method robust --verbose

# Results in ./heterodyne_results/robust/
```

### Compare All Methods

```bash
# Run both classical and robust optimization
heterodyne --config my_config.json --method all --verbose

# Generate comparison plots
ls heterodyne_results/comparison_plots/
```

## Step 6: Interpret Results

### Understanding Output Structure

```
heterodyne_results/
├── heterodyne_analysis_results.json    # Main summary with all methods
├── run.log                             # Detailed execution log
├── classical/
│   ├── nelder_mead/
│   │   ├── parameters.json             # Optimized parameters
│   │   ├── fitted_data.npz            # Experimental + fitted data
│   │   ├── analysis_results_nelder_mead.json
│   │   └── c2_heatmaps_phi_*.png      # Comparison plots
│   └── gurobi/                         # If Gurobi is available
│       └── ...
├── robust/
│   ├── wasserstein/
│   │   └── ...
│   ├── scenario/
│   │   └── ...
│   └── ellipsoidal/
│       └── ...
└── comparison_plots/                   # Method comparison
    └── ...
```

### Extracting Parameter Values

```python
import json
import numpy as np

# Load results
with open("heterodyne_results/classical/nelder_mead/parameters.json", 'r') as f:
    results = json.load(f)

# Extract optimized parameters
params = results['optimized_parameters']
D0_ref = params[0]
alpha_ref = params[1]
D_offset_ref = params[2]
D0_sample = params[3]
alpha_sample = params[4]
D_offset_sample = params[5]
v0 = params[6]
beta = params[7]
v_offset = params[8]
f0, f1, f2, f3 = params[9:13]
phi0 = params[13]

print(f"Reference transport: D₀_ref = {D0_ref:.3e} Å²/s, α_ref = {alpha_ref:.3f}")
print(f"Sample transport: D₀_sample = {D0_sample:.3e} Å²/s, α_sample = {alpha_sample:.3f}")
print(f"Velocity: v₀ = {v0:.3e} nm/s, β = {beta:.3f}")
print(f"Flow angle: φ₀ = {phi0:.3f}°")

# Load chi-squared
chi_squared = results['chi_squared']
print(f"\nGoodness of fit: χ² = {chi_squared:.6e}")
```

### Loading Fitted Data

```python
# Load fitted data arrays
data = np.load("heterodyne_results/classical/nelder_mead/fitted_data.npz")

c2_experimental = data['c2_experimental']
c2_fitted = data['c2_theoretical_scaled']
residuals = data['residuals']
phi_angles = data['phi_angles']

# Analyze fit quality
print(f"Experimental data shape: {c2_experimental.shape}")
print(f"Mean absolute residual: {np.mean(np.abs(residuals)):.6e}")
print(f"Analyzed angles: {phi_angles}")
```

## Step 7: Python API Usage

For programmatic analysis and custom workflows:

```python
import numpy as np
import json
from heterodyne.analysis.core import HeterodyneAnalysisCore
from heterodyne.optimization.classical import ClassicalOptimizer
from heterodyne.data.xpcs_loader import load_xpcs_data

# Load configuration
config_file = "my_config.json"
with open(config_file, 'r') as f:
    config = json.load(f)

# Initialize analysis core
core = HeterodyneAnalysisCore(config)

# Load experimental data
c2_data, time_length, phi_angles, num_angles = load_xpcs_data(config_file)

print(f"Loaded data shape: {c2_data.shape}")
print(f"Time length: {time_length} frames")
print(f"Number of angles: {num_angles}")
print(f"Phi angles: {phi_angles}")

# Run classical optimization
optimizer = ClassicalOptimizer(core, config)
params, results = optimizer.run_classical_optimization_optimized(
    phi_angles=phi_angles,
    c2_experimental=c2_data
)

# Extract and display results
param_names = [
    "D0_ref", "alpha_ref", "D_offset_ref",
    "D0_sample", "alpha_sample", "D_offset_sample",
    "v0", "beta", "v_offset",
    "f0", "f1", "f2", "f3",
    "phi0"
]

print("\nOptimization Results:")
print("=" * 60)
for name, value in zip(param_names, params):
    print(f"{name:20s}: {value:12.6e}")

print(f"\nChi-squared: {results.chi_squared:.6e}")
print(f"Best method: {results.best_method}")
print(f"Success: {results.success}")
```

## Step 8: Advanced Analysis

### Multi-Parameter Optimization

Start with few parameters, then gradually add more:

```python
# Stage 1: Optimize core parameters
config['initial_parameters']['active_parameters'] = [
    "D0_ref", "D0_sample", "v0", "f0"
]

# Stage 2: Add transport exponents
config['initial_parameters']['active_parameters'] = [
    "D0_ref", "alpha_ref", "D0_sample", "alpha_sample", "v0", "f0"
]

# Stage 3: Full 14-parameter optimization
config['initial_parameters']['active_parameters'] = [
    "D0_ref", "alpha_ref", "D_offset_ref",
    "D0_sample", "alpha_sample", "D_offset_sample",
    "v0", "beta", "v_offset",
    "f0", "f1", "f2", "f3",
    "phi0"
]
```

### Robust Optimization for Noisy Data

```python
from heterodyne.optimization.robust import RobustHeterodyneOptimizer

# Initialize robust optimizer
robust = RobustHeterodyneOptimizer(core, config)

# Run Wasserstein DRO
robust_result = robust.optimize(
    phi_angles=phi_angles,
    c2_experimental=c2_data,
    method="wasserstein",
    epsilon=0.1  # Uncertainty radius
)

# Extract results
robust_params = robust_result['parameters']
robust_chi2 = robust_result['chi_squared']

print(f"\nRobust Optimization Results:")
print(f"D₀_ref: {robust_params[0]:.3e} Å²/s")
print(f"χ²: {robust_chi2:.6e}")
```

## Common Issues and Solutions

### Issue: "File not found" Error

**Problem:**
```
FileNotFoundError: [Errno 2] No such file or directory: './data/correlation_data.h5'
```

**Solution:**
```bash
# Verify file paths
ls -lh data/my_experiment/

# Update paths in config.json to absolute paths
```

### Issue: Optimization Fails to Converge

**Problem:**
```
Warning: Optimization did not converge (maxiter exceeded)
```

**Solutions:**

1. **Better initial values:**
   ```json
   {
     "initial_parameters": {
       "values": [10000.0, -1.5, 0.0, ...]
     }
   }
   ```

2. **Reduce active parameters:**
   ```json
   {
     "initial_parameters": {
       "active_parameters": ["D0_ref", "D0_sample"]
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
             "maxiter": 20000
           }
         }
       }
     }
   }
   ```

### Issue: Out of Memory Error

**Problem:**
```
MemoryError: Unable to allocate array
```

**Solution:**
Enable subsampling for large datasets:

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

### Issue: Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'heterodyne'
```

**Solution:**
```bash
# Verify installation
pip list | grep heterodyne

# Reinstall if needed
pip install --upgrade heterodyne-analysis[all]

# Check Python version
python --version  # Should be 3.12+
```

## Next Steps

### Learn More

- **Configuration Guide**: See [configuration.rst](configuration.rst) for all parameters
- **API Reference**: Explore [analysis-core.md](../api-reference/analysis-core.md) for details
- **Examples**: Review [examples.rst](examples.rst) for advanced workflows
- **Performance**: Read [ml-acceleration.md](ml-acceleration.md) for optimization tips

### Workflow Tips

1. **Always validate data first** with `--plot-experimental-data`
2. **Start with fewer active parameters** (4-6) then gradually add more
3. **Use classical methods first** for clean data, robust for noisy data
4. **Save configurations** for reproducibility
5. **Monitor chi-squared values** to assess fit quality
6. **Compare multiple methods** with `--method all`

### Quality Thresholds

- **Excellent fit**: χ² < 5.0
- **Good fit**: χ² < 10.0
- **Acceptable fit**: χ² < 20.0
- **Poor fit**: χ² > 20.0 (review parameters and data quality)

### Getting Help

- **Documentation**: https://heterodyne-analysis.readthedocs.io/
- **GitHub Issues**: https://github.com/imewei/heterodyne_analysis/issues
- **Citation**: He et al. PNAS 2024, DOI: 10.1073/pnas.2401162121
