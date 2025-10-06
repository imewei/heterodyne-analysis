# Heterodyne Scattering Analysis Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![PyPI version](https://badge.fury.io/py/heterodyne-analysis.svg)](https://badge.fury.io/py/heterodyne-analysis)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://heterodyne-analysis.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-1.9+-green.svg)](https://scipy.org)
[![Numba](https://img.shields.io/badge/Numba-JIT-orange.svg)](https://numba.pydata.org)
[![DOI](https://img.shields.io/badge/DOI-10.1073/pnas.2401162121-blue.svg)](https://doi.org/10.1073/pnas.2401162121)
[![Research](https://img.shields.io/badge/Research-XPCS%20Nonequilibrium-purple.svg)](https://github.com/imewei/heterodyne_analysis)

## Overview

**heterodyne-analysis** is a research-grade Python package for analyzing heterodyne
scattering in X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium
conditions. This package implements theoretical frameworks from
[He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) for characterizing
transport properties in flowing soft matter systems through time-dependent intensity
correlation functions.

The package provides comprehensive analysis of nonequilibrium dynamics by capturing the
interplay between Brownian diffusion and advective shear flow in complex fluids, with
applications to biological systems, colloids, and active matter under flow conditions.

## Key Features

### Analysis Capabilities

- **Heterodyne Scattering Model** (14 parameters): Two-component heterodyne scattering with time-dependent fraction mixing
  - **Diffusion** (3 params): D₀, α, D_offset
  - **Velocity** (3 params): v₀, β, v_offset
  - **Fraction** (4 params): f₀, f₁, f₂, f₃
  - **Flow angle** (1 param): φ₀
- **Multiple optimization methods**: Classical (Nelder-Mead, Powell), Robust
  (Wasserstein DRO, Scenario-based, Ellipsoidal)
- **Parameter validation**: Physical constraints ensure valid heterodyne parameters

### Performance

- **Numba JIT compilation**: 3-5x speedup for core calculations
- **Vectorized operations**: Optimized NumPy array processing
- **Memory optimization**: Ellipsoidal optimization with 90% memory limit for large
  datasets (8M+ data points)
- **Smart caching**: Intelligent data caching with automatic dimension validation

### Data Format Support

- **APS and APS-U formats**: Auto-detection and loading of both old and new synchrotron
  data formats
- **Cached data compatibility**: Automatic cache dimension adjustment and validation
- **Configuration templates**: Comprehensive templates with documented subsampling
  strategies

### Validation and Quality

- **Statistical validation**: Cross-validation and bootstrap analysis for parameter
  reliability
- **Experimental validation**: Tested at synchrotron facilities (APS Sector 8-ID-I)
- **Regression testing**: Comprehensive test suite with performance benchmarks

## Quick Start

```bash
# Install
pip install heterodyne-analysis[all]

# Create heterodyne configuration (14 parameters)
cp heterodyne/config/heterodyne_11param_example.json my_config.json
# Edit my_config.json with your experimental parameters

# Run analysis
heterodyne --config my_config.json --method all

# Run robust optimization for noisy data
heterodyne --config my_config.json --method robust
```

## Quick Reference (v1.0.0)

### Most Common Workflows

**1. First-time Setup:**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with all dependencies
pip install heterodyne-analysis[all]

# Generate configuration template
heterodyne-config --mode laminar_flow --sample my_experiment
```

**2. Standard Analysis (Clean Data):**

```bash
# Edit my_config.json with your experimental parameters
# Then run classical optimization
heterodyne --config my_config.json --method classical

# Results saved to: ./heterodyne_results/
```

**3. Robust Analysis (Noisy Data):**

```bash
# For data with outliers or measurement noise
heterodyne --config my_config.json --method robust --verbose

# Compare both methods
heterodyne --config my_config.json --method all
```

**4. Python API Usage:**

```python
import numpy as np
import json
from heterodyne.analysis.core import HeterodyneAnalysisCore
from heterodyne.optimization.classical import ClassicalOptimizer

# Load config and initialize
with open("my_config.json", 'r') as f:
    config = json.load(f)
core = HeterodyneAnalysisCore(config)

# Load your experimental data
phi_angles = np.array([0, 36, 72, 108, 144])
c2_data = load_your_data()  # Your data loading function

# Run optimization
optimizer = ClassicalOptimizer(core, config)
params, results = optimizer.run_classical_optimization_optimized(
    phi_angles=phi_angles,
    c2_experimental=c2_data
)

print(f"D₀ = {params[0]:.3e} Å²/s")
print(f"χ² = {results.chi_squared:.6e}")
```

**5. Troubleshooting:**

```bash
# Enable verbose logging for debugging
heterodyne --config my_config.json --verbose

# Run test suite to verify installation
pytest -v -m "not slow"

# Check frame counting (v1.0.0 critical fix)
pytest heterodyne/tests/test_time_length_calculation.py -v
```

## Heterodyne Model (11 Parameters)

The package implements the **general two-component heterodyne scattering model** from [He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) **Equation S-95**, which uses time-dependent transport coefficients J(t) for nonequilibrium dynamics.

### Model Foundation (Equation S-95)

This implementation uses **Equation S-95** (general time-dependent form) with transport coefficients:

```
c₂(q⃗, t₁, t₂) = 1 + β/f² [
  [xᵣ(t₁)xᵣ(t₂)]² exp(-q² ∫[t₁ to t₂] J(t) dt) +
  [xₛ(t₁)xₛ(t₂)]² exp(-q² ∫[t₁ to t₂] J(t) dt) +
  2xᵣ(t₁)xᵣ(t₂)xₛ(t₁)xₛ(t₂) exp(-q² ∫[t₁ to t₂] J(t) dt) cos(...)
]
```

where:
- **J(t)**: Time-dependent transport coefficient [Å²/s]
- **x_n(t)**: Time-dependent fraction of component n (reference/sample)
- **𝔼[v(t)]**: Time-dependent mean velocity
- **φ**: Angle between scattering vector and flow direction
- **β**: Contrast factor
- **f²**: Normalization factor

**Relationship to Equilibrium Form (Equation S-98):**
For equilibrium Wiener processes, the transport coefficient reduces to J = 6D, where D is the traditional diffusion coefficient. Equation S-98 is this equilibrium simplification. This package implements the more general S-95 with time-dependent J(t).

### Nonequilibrium Implementation (11-Parameter Model)

This package parameterizes J(t) as **J(t) = J₀·t^α + J_offset** to capture nonequilibrium dynamics including aging, yielding, and shear banding phenomena. Note that parameters labeled "D" in the code (D₀, α, D_offset) are actually transport coefficient parameters (J₀, α, J_offset) for historical compatibility.

### Parameters

**Transport Coefficients (3 parameters):**
- `D₀`: Reference transport coefficient J₀ [Å²/s], range [1.0, 1×10⁶] (labeled "D" for historical compatibility)
- `α`: Transport coefficient time-scaling exponent (dimensionless), range [-2, 2]
- `D_offset`: Baseline transport coefficient J_offset [Å²/s], range [-100, 100]

**Note:** For equilibrium Wiener processes, J = 6D where D is the traditional diffusion coefficient. This implementation uses J(t) directly for nonequilibrium dynamics.

**Velocity (3 parameters):**
- `v₀`: Reference velocity (nm/s), range [-10, 10]
- `β`: Velocity power-law exponent (dimensionless), range [-2, 2]
- `v_offset`: Baseline velocity offset (nm/s), range [-1, 1]

**Fraction (4 parameters):**
- `f₀`: Fraction amplitude (dimensionless), range [0, 1]
- `f₁`: Fraction exponential rate (1/s), range [-1, 1]
- `f₂`: Fraction time offset (s), range [0, 200]
- `f₃`: Fraction baseline (dimensionless), range [0, 1]

**Flow Angle (1 parameter):**
- `φ₀`: Flow direction angle (degrees), range [-360, 360]

### Time-Dependent Fraction

The sample fraction follows:
```
f(t) = f₀ × exp(f₁ × (t - f₂)) + f₃
```

**Constraint**: `0 ≤ f(t) ≤ 1` for all times (enforced during validation)

### Example Configuration

```json
{
  "initial_parameters": {
    "values": [100.0, -0.5, 10.0, 0.1, 0.0, 0.01, 0.5, 0.0, 50.0, 0.3, 0.0],
    "parameter_names": [
      "D0", "alpha", "D_offset",
      "v0", "beta", "v_offset",
      "f0", "f1", "f2", "f3",
      "phi0"
    ]
  },
  "analyzer_parameters": {
    "temporal": {"dt": 0.1, "start_frame": 0, "end_frame": 100},
    "scattering": {"wavevector_q": 0.0054},
    "geometry": {"stator_rotor_gap": 2000000}
  }
}
```

## Installation

### Standard Installation

```bash
pip install heterodyne-analysis[all]
```

### Research Environment Setup

```bash
# Create isolated research environment
conda create -n heterodyne-research python=3.12
conda activate heterodyne-research

# Install with all scientific dependencies
pip install heterodyne-analysis[all]

# For development and method extension
git clone https://github.com/imewei/heterodyne.git
cd heterodyne
pip install -e .[dev]
```

### Optional Dependencies

- **Performance**: `pip install heterodyne-analysis[performance]` (numba, psutil,
  performance monitoring)
- **Robust optimization**: `pip install heterodyne-analysis[robust]` (cvxpy, gurobipy,
  mosek)
- **Development**: `pip install heterodyne-analysis[dev]` (testing, linting, documentation
  tools)

### High-Performance Configuration

```bash
# Optimize for computational performance
export OMP_NUM_THREADS=8
export NUMBA_DISABLE_INTEL_SVML=1
export HETERODYNE_PERFORMANCE_MODE=1

# Enable advanced optimization (requires license)
pip install heterodyne-analysis[gurobi]
```

## Commands

### Main Analysis Command

```bash
heterodyne [OPTIONS]
```

**Key Options:**

- `--method {classical,robust,all}` - Analysis method (default: classical)
- `--config CONFIG` - Configuration file (default: ./heterodyne_config.json)
- `--output-dir DIR` - Output directory (default: ./heterodyne_results)
- `--verbose` - Debug logging
- `--quiet` - File logging only
- `--plot-experimental-data` - Generate data validation plots
- `--plot-simulated-data` - Plot theoretical correlations

**Examples:**

```bash
# Basic analysis
heterodyne --method classical
heterodyne --method robust --verbose
heterodyne --method all

# Data validation
heterodyne --plot-experimental-data
heterodyne --plot-simulated-data --contrast 1.5 --offset 0.1

# Custom configuration
heterodyne --config experiment.json --output-dir ./results
```

### Configuration Generator

```bash
heterodyne-config [OPTIONS]
```

**Options:**

- `--mode {static_isotropic,static_anisotropic,laminar_flow}` - Analysis mode
- `--output OUTPUT` - Output file (default: my_config.json)
- `--sample SAMPLE` - Sample name
- `--author AUTHOR` - Author name
- `--experiment EXPERIMENT` - Experiment description

**Examples:**

```bash
# Default laminar flow config
heterodyne-config

# Static isotropic (fastest)
heterodyne-config --mode static_isotropic --output fast_config.json

# With metadata
heterodyne-config --sample protein --author "Your Name"
```

## Shell Completion

The package supports shell completion for bash, zsh, and fish shells:

```bash
# For bash
heterodyne --help  # Shows all available options

# Tab completion works for:
heterodyne --method <TAB>     # classical, robust, all
heterodyne --config <TAB>     # Completes file paths
heterodyne --output-dir <TAB> # Completes directory paths
```

**Note:** Shell completion is built into the CLI and works automatically in most modern
shells. For advanced completion features, you may need to install optional completion
dependencies.

## Scientific Background

### Physical Model

The package analyzes time-dependent intensity correlation functions in the presence of
laminar flow:

$$c_2(\\vec{q}, t_1, t_2) = 1 + \\beta\\left[e^{-q^2\\int J(t)dt}\\right] \\times
\\text{sinc}^2\\left\[\\frac{1}{2\\pi} qh
\\int\\dot{\\gamma}(t)\\cos(\\phi(t))dt\\right\]$$

where:

- $\\vec{q}$: scattering wavevector [Å⁻¹]
- $h$: gap between stator and rotor [Å]
- $\\phi(t)$: angle between shear/flow direction and $\\vec{q}$ [degrees]
- $\\dot{\\gamma}(t)$: time-dependent shear rate [s⁻¹]
- $J(t)$: time-dependent transport coefficient [Å²/s] (labeled D in code)
- $\\beta$: instrumental contrast parameter

### Analysis Modes

| Mode | Parameters | Physical Description | Computational Complexity | Speed |
|------|------------|---------------------|--------------------------|-------| |
**Static Isotropic** | 3 | Brownian motion only, isotropic systems | O(N) | ⭐⭐⭐ | |
**Static Anisotropic** | 3 | Static systems with angular dependence | O(N log N) | ⭐⭐ |
| **Laminar Flow** | 7 | Full nonequilibrium with flow and shear | O(N²) | ⭐ |

#### Model Parameters

**Static Parameters (All Modes):**

- $D_0$: baseline transport coefficient J₀ [Å²/s] (labeled 'D' for compatibility)
- $\\alpha$: transport coefficient scaling exponent
- $D\_{\\text{offset}}$: additive transport coefficient offset J_offset [Å²/s]

**Flow Parameters (Laminar Flow Mode):**

- $\\dot{\\gamma}\_0$: baseline shear rate [s⁻¹]
- $\\beta$: shear rate scaling exponent
- $\\dot{\\gamma}\_{\\text{offset}}$: additive shear rate offset [s⁻¹]
- $\\phi_0$: flow direction angle [degrees]

## Frame Counting Convention

### Overview

The package uses **1-based inclusive frame counting** in configuration files, which is
then converted to 0-based Python array indices for processing.

### Frame Counting Formula

```python
time_length = end_frame - start_frame + 1  # Inclusive counting
```

**Examples:**

- `start_frame=1, end_frame=100` → `time_length=100` (not 99!)
- `start_frame=401, end_frame=1000` → `time_length=600` (not 599!)
- `start_frame=1, end_frame=1` → `time_length=1` (single frame)

### Convention Details

**Config Convention (1-based, inclusive):**

- `start_frame=1` means "start at first frame"
- `end_frame=100` means "include frame 100"
- Both boundaries are inclusive: `[start_frame, end_frame]`

**Python Slice Convention (0-based, exclusive end):**

- Internally converted using: `python_start = start_frame - 1`
- `python_end = end_frame` (kept as-is for exclusive slice)
- Array slice `[python_start:python_end]` gives exactly `time_length` elements

**Example Conversion:**

```python
# Config values
start_frame = 401  # 1-based
end_frame = 1000   # 1-based

# Convert to Python indices
python_start = 400  # 0-based (401 - 1)
python_end = 1000   # 0-based, exclusive

# Slice gives correct number of frames
data_slice = full_data[:, 400:1000, 400:1000]  # 600 frames
time_length = 1000 - 401 + 1  # = 600 ✓
```

### Cached Data Compatibility

**Cache Filename Convention:**

- Cache files use config values:
  `cached_c2_isotropic_frames_{start_frame}_{end_frame}.npz`
- Example: `cached_c2_isotropic_frames_401_1000.npz` contains 600 frames

**Cache Dimension Validation:** The analysis core automatically detects and adjusts for
dimension mismatches:

```python
# Automatic adjustment if cache dimensions differ
if c2_experimental.shape[1] != self.time_length:
    logger.info(f"Auto-adjusting time_length to match cached data")
    self.time_length = c2_experimental.shape[1]
```

### Utility Functions

Two centralized utility functions ensure consistency:

```python
from heterodyne.core.io_utils import calculate_time_length, config_frames_to_python_slice

# Calculate time_length from config frames
time_length = calculate_time_length(start_frame=401, end_frame=1000)
# Returns: 600

# Convert for Python array slicing
python_start, python_end = config_frames_to_python_slice(401, 1000)
# Returns: (400, 1000) for use in data[400:1000]
```

## Conditional Angle Subsampling

### Strategy

The package automatically preserves angular information when the number of available
angles is small:

```python
# Automatic angle preservation
if n_angles < 4:
    # Use all available angles to preserve angular information
    angle_subsample_size = n_angles
else:
    # Subsample for performance (default: 4 angles)
    angle_subsample_size = config.get("n_angles", 4)
```

### Impact

- **Before fix**: 2 angles → 1 angle (50% angular information loss)
- **After fix**: 2 angles → 2 angles (100% preservation)
- Time subsampling still applied: ~16x performance improvement with \<10% χ² degradation

### Configuration

All configuration templates include subsampling documentation:

```json
{
  "subsampling": {
    "n_angles": 4,
    "n_time_points": 16,
    "comment": "Conditional: n_angles preserved when < 4 for angular info retention"
  }
}
```

## Optimization Methods

### Classical Methods

1. **Nelder-Mead Simplex**: Derivative-free optimization for robust convergence
2. **Gurobi Quadratic Programming**: High-performance commercial solver with trust
   region methods

### Robust Optimization Framework

Advanced uncertainty-aware optimization for noisy experimental data:

1. **Distributionally Robust Optimization (DRO)**:

   - Wasserstein uncertainty sets for data distribution robustness
   - Optimal transport-based uncertainty quantification

2. **Scenario-Based Optimization**:

   - Bootstrap resampling for statistical robustness
   - Monte Carlo uncertainty propagation

3. **Ellipsoidal Uncertainty Sets**:

   - Bounded uncertainty with confidence ellipsoids
   - Analytical uncertainty bounds
   - Memory optimization: 90% limit for large datasets

**Usage Guidelines:**

- Use `--method robust` for noisy data with outliers
- Use `--method classical` for clean, low-noise data
- Use `--method all` to run both and compare results

## Python API

### Basic Usage

```python
import numpy as np
import json
from heterodyne.analysis.core import HeterodyneAnalysisCore
from heterodyne.optimization.classical import ClassicalOptimizer
from heterodyne.data.xpcs_loader import load_xpcs_data

# Load configuration file
config_file = "config_static_isotropic.json"
with open(config_file, 'r') as f:
    config = json.load(f)

# Initialize analysis core
core = HeterodyneAnalysisCore(config)

# Load experimental XPCS data
phi_angles = np.array([0, 36, 72, 108, 144])  # Example angles in degrees
c2_data = load_xpcs_data(
    data_path=config['experimental_data']['data_folder_path'],
    phi_angles=phi_angles,
    n_angles=len(phi_angles)
)

# Run classical optimization
classical = ClassicalOptimizer(core, config)
optimal_params, results = classical.run_classical_optimization_optimized(
    phi_angles=phi_angles,
    c2_experimental=c2_data
)

print(f"Optimal parameters: {optimal_params}")
print(f"Chi-squared: {results.chi_squared:.6e}")
print(f"Method: {results.best_method}")
```

### Research Workflow

```python
import numpy as np
import json
from heterodyne.analysis.core import HeterodyneAnalysisCore
from heterodyne.optimization.classical import ClassicalOptimizer
from heterodyne.optimization.robust import RobustHeterodyneOptimizer
from heterodyne.data.xpcs_loader import load_xpcs_data

# Load experimental configuration
config_file = "config_laminar_flow.json"
with open(config_file, 'r') as f:
    config = json.load(f)

# Initialize analysis core
core = HeterodyneAnalysisCore(config)

# Load XPCS correlation data
phi_angles = np.array([0, 36, 72, 108, 144])  # Scattering angles in degrees
c2_data = load_xpcs_data(
    data_path=config['experimental_data']['data_folder_path'],
    phi_angles=phi_angles,
    n_angles=len(phi_angles)
)

# Classical analysis for clean data
classical = ClassicalOptimizer(core, config)
classical_params, classical_results = classical.run_classical_optimization_optimized(
    phi_angles=phi_angles,
    c2_experimental=c2_data
)

# Robust analysis for noisy data
robust = RobustHeterodyneOptimizer(core, config)
robust_result_dict = robust.optimize(
    phi_angles=phi_angles,
    c2_experimental=c2_data,
    method="wasserstein",  # Options: "wasserstein", "scenario", "ellipsoidal"
    epsilon=0.1  # Uncertainty radius for DRO
)

# Extract results
robust_params = robust_result_dict['optimal_params']
robust_chi2 = robust_result_dict['chi_squared']

print(f"Classical D₀: {classical_params[0]:.3e} Å²/s")
print(f"Classical χ²: {classical_results.chi_squared:.6e}")
print(f"\nRobust D₀: {robust_params[0]:.3e} Å²/s")
print(f"Robust χ²: {robust_chi2:.6e}")
```

### Performance Benchmarking

```python
import time
import numpy as np
import json
from heterodyne.analysis.core import HeterodyneAnalysisCore
from heterodyne.optimization.classical import ClassicalOptimizer
from heterodyne.data.xpcs_loader import load_xpcs_data

# Load configuration
config_file = "config_laminar_flow.json"
with open(config_file, 'r') as f:
    config = json.load(f)

# Initialize
core = HeterodyneAnalysisCore(config)
phi_angles = np.array([0, 36, 72, 108, 144])
c2_data = load_xpcs_data(
    data_path=config['experimental_data']['data_folder_path'],
    phi_angles=phi_angles,
    n_angles=len(phi_angles)
)

# Benchmark classical optimization
classical = ClassicalOptimizer(core, config)
start_time = time.perf_counter()
params, results = classical.run_classical_optimization_optimized(
    phi_angles=phi_angles,
    c2_experimental=c2_data
)
elapsed_time = time.perf_counter() - start_time

print(f"Classical optimization completed in {elapsed_time:.2f} seconds")
print(f"Chi-squared: {results.chi_squared:.6e}")
print(f"Best method: {results.best_method}")
```

## Configuration

### Creating Configurations

```bash
# Generate templates
heterodyne-config --mode static_isotropic --sample protein_01
heterodyne-config --mode laminar_flow --sample microgel
```

### Analysis Mode

The package uses **Laminar Flow Mode** (7 parameters) for all analyses:

```json
{
  "analysis_settings": {
    "model_description": {
      "laminar_flow": "7-parameter heterodyne model with time-dependent transport coefficients and shear"
    }
  }
}
```

**Note:** Static modes have been removed in favor of the more general heterodyne model. If your configuration contains `static_mode` settings, please update to use laminar flow configuration.

### Subsampling Configuration

```json
{
  "subsampling": {
    "n_angles": 4,
    "n_time_points": 16,
    "strategy": "conditional",
    "preserve_angular_info": true
  }
}
```

**Performance Impact:**

- Time subsampling: ~16x speedup
- Angle subsampling: Conditional based on available angles
- Combined impact: 20-50x speedup with \<10% χ² degradation

### Data Formats and Standards

**XPCS Correlation Data Format:**

- Time correlation functions: `c2(q, φ, t1, t2)` as HDF5 or NumPy arrays
- Scattering angles: φ values in degrees \[0°, 360°)
- Time delays: τ = t2 - t1 in seconds
- Wavevector magnitude: q in Å⁻¹

**Configuration Schema:**

```json
{
  "analysis_settings": {
    "angle_filtering": true,
    "optimization_method": "all"
  },
  "experimental_parameters": {
    "q_magnitude": 0.0045,
    "gap_height": 50000.0,
    "temperature": 293.15,
    "viscosity": 1.0e-3
  },
  "frame_settings": {
    "start_frame": 401,
    "end_frame": 1000,
    "time_length_comment": "Calculated as end_frame - start_frame + 1 = 600"
  },
  "optimization_bounds": {
    "D0": [1e-15, 1e-10],
    "alpha": [0.1, 2.0],
    "D_offset": [-1e-12, 1e-12]
  }
}
```

## Output Structure

When running `heterodyne --method all`, the complete analysis produces a comprehensive
results directory with all optimization methods:

```
heterodyne_results/
├── heterodyne_analysis_results.json    # Summary with all methods
├── run.log                           # Detailed execution log
│
├── classical/                        # Classical optimization results
│   ├── nelder_mead/                  # Nelder-Mead simplex method
│   │   ├── parameters.json           # Optimal parameters with metadata
│   │   ├── fitted_data.npz          # Fitted correlation functions + experimental metadata
│   │   ├── analysis_results_nelder_mead.json  # Complete results + chi-squared
│   │   ├── convergence_metrics.json  # Iterations, function evaluations, diagnostics
│   │   └── c2_heatmaps_phi_*.png    # Experimental vs fitted comparison plots
│   └── gurobi/                       # Gurobi quadratic programming (if available)
│       ├── parameters.json
│       ├── fitted_data.npz
│       ├── analysis_results_gurobi.json
│       ├── convergence_metrics.json
│       └── c2_heatmaps_phi_*.png
│
├── robust/                           # Robust optimization results
│   ├── wasserstein/                  # Distributionally Robust Optimization (DRO)
│   │   ├── parameters.json           # Robust optimal parameters
│   │   ├── fitted_data.npz          # Fitted correlations with uncertainty bounds
│   │   ├── analysis_results_wasserstein.json  # DRO results + uncertainty radius
│   │   ├── convergence_metrics.json  # Optimization convergence info
│   │   └── c2_heatmaps_phi_*.png    # Robust fit comparison plots
│   ├── scenario/                     # Scenario-based bootstrap optimization
│   │   ├── parameters.json
│   │   ├── fitted_data.npz
│   │   ├── analysis_results_scenario.json
│   │   ├── convergence_metrics.json
│   │   └── c2_heatmaps_phi_*.png
│   └── ellipsoidal/                  # Ellipsoidal uncertainty sets
│       ├── parameters.json
│       ├── fitted_data.npz
│       ├── analysis_results_ellipsoidal.json
│       ├── convergence_metrics.json
│       └── c2_heatmaps_phi_*.png
│
└── comparison_plots/                 # Method comparison visualizations
    ├── method_comparison_phi_*.png   # Classical vs Robust comparison
    └── parameter_comparison.png      # Parameter values across methods
```

### Key Output Files

**heterodyne_analysis_results.json**: Main summary containing:

- Analysis timestamp and methods run
- Experimental parameters (q, dt, gap size, frames)
- Optimization results for all methods:
  - `classical_nelder_mead`, `classical_gurobi`, `classical_best`
  - `robust_wasserstein`, `robust_scenario`, `robust_ellipsoidal`, `robust_best`

**fitted_data.npz**: NumPy compressed archive with:

- Experimental metadata: `wavevector_q`, `dt`, `stator_rotor_gap`, `start_frame`,
  `end_frame`
- Correlation data: `c2_experimental`, `c2_theoretical_raw`, `c2_theoretical_scaled`
- Scaling parameters: `contrast_params`, `offset_params`
- Quality metrics: `residuals`

**analysis_results\_{method}.json**: Method-specific detailed results:

- Optimized parameters with names
- Chi-squared and reduced chi-squared values
- Experimental metadata
- Scaling parameters for each angle
- Success status and timestamp

**convergence_metrics.json**: Optimization diagnostics:

- Number of iterations
- Function evaluations
- Convergence message
- Final chi-squared value

## Performance

### Environment Optimization

```bash
export OMP_NUM_THREADS=8
export NUMBA_DISABLE_INTEL_SVML=1
export HETERODYNE_PERFORMANCE_MODE=1
```

### Optimizations

- **Numba JIT**: 3-5x speedup for core calculations
- **Vectorized operations**: Optimized array processing
- **Memory efficiency**: Smart caching and allocation
- **Batch processing**: Vectorized chi-squared calculation
- **Conditional subsampling**: 20-50x speedup with minimal accuracy loss

### Benchmarking Results

**Performance Comparison (Intel Xeon, 8 cores):**

| Data Size | Pure Python | Numba JIT | Speedup |
|-----------|-------------|-----------|---------| | 100 points | 2.3 s | 0.7 s | 3.3× |
| 500 points | 12.1 s | 3.2 s | 3.8× | | 1000 points | 45.2 s | 8.9 s | 5.1× | | 5000
points | 892 s | 178 s | 5.0× |

**Memory Optimization:**

| Dataset Size | Before | After | Improvement |
|--------------|--------|-------|-------------| | 8M data points | Memory error | 90%
limit success | Enabled | | 4M data points | 85% usage | 75% usage | 12% reduction |

## Testing

### Quick Test Suite (Development)

```bash
# Fast test suite excluding slow tests (recommended for development)
pytest -v -m "not slow"

# Run frame counting regression tests (v1.0.0 formula)
pytest heterodyne/tests/test_time_length_calculation.py -v

# Run with coverage
pytest -v --cov=heterodyne --cov-report=html -m "not slow"
```

### Comprehensive Test Suite (CI/CD)

```bash
# Full test suite including slow performance tests
pytest heterodyne/tests/ -v

# Performance benchmarks only
pytest heterodyne/tests/ -v -m performance

# Run with parallel execution for speed
pytest -v -n auto
```

### Testing Guide

For comprehensive testing documentation including:

- Frame counting convention (v1.0.0 changes)
- Test markers and categorization
- Temporary file management best practices
- Writing new tests for v1.0.0

See [TESTING.md](TESTING.md) for detailed testing guidelines.

## Citation

If you use this software in your research, please cite the original paper:

```bibtex
@article{he2024transport,
  title={Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter},
  author={He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh and Chen, Wei},
  journal={Proceedings of the National Academy of Sciences},
  volume={121},
  number={31},
  pages={e2401162121},
  year={2024},
  publisher={National Academy of Sciences},
  doi={10.1073/pnas.2401162121}
}
```

**For the software package:**

```bibtex
@software{heterodyne_analysis,
  title={heterodyne-analysis: High-performance XPCS analysis with robust optimization},
  author={Chen, Wei and He, Hongrui},
  year={2024-2025},
  url={https://github.com/imewei/heterodyne_analysis},
  version={1.0.0},
  institution={Argonne National Laboratory}
}
```

## Development

Development setup:

```bash
git clone https://github.com/imewei/heterodyne_analysis.git
cd heterodyne
pip install -e .[dev]

# Run tests
pytest heterodyne/tests/ -v

# Code quality
ruff check heterodyne/
ruff format heterodyne/
black heterodyne/
isort heterodyne/
mypy heterodyne/
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development guidelines.

## License

This research software is distributed under the MIT License, enabling open collaboration
while maintaining attribution requirements for academic use.

**Research Use**: Freely available for academic research with proper citation
**Commercial Use**: Permitted under MIT License terms **Modification**: Encouraged with
contribution back to the community

______________________________________________________________________

**Contact Information:**

- **Primary Investigator**: Wei Chen ([wchen@anl.gov](mailto:wchen@anl.gov))
- **Technical Support**:
  [GitHub Issues](https://github.com/imewei/heterodyne_analysis/issues)
- **Research Collaboration**: Argonne National Laboratory, X-ray Science Division

**Authors:** Wei Chen, Hongrui He (Argonne National Laboratory) **License:** MIT
