Quick Start Guide
=================

This guide will get you analyzing heterodyne scattering data in minutes.

Installation
------------

.. code-block:: bash

   pip install heterodyne-analysis[all]

5-Minute Tutorial
-----------------

**Step 1: Create a Configuration**

.. code-block:: bash

   # Create an 11-parameter heterodyne configuration
   heterodyne-config --mode heterodyne --sample my_sample

   # Tab completion works automatically in most shells:
   # heterodyne-config --mode <TAB>  (shows available modes)

**Step 2: Prepare Your Data**

Ensure your experimental data is in the correct format:

- **C2 data file**: Correlation function data (HDF5 or NPZ format)
- **Angle file**: Scattering angles (text file with angles in degrees)

**Step 3: Run Analysis**

.. code-block:: bash

   # Data validation first (optional, saves plots to ./heterodyne_results/exp_data/)
   heterodyne --config my_sample_config.json --plot-experimental-data

   # Basic analysis (fastest, saves results to ./heterodyne_results/)
   heterodyne --config my_sample_config.json --method classical

   # Run all methods with verbose output
   heterodyne --config my_sample_config.json --method all --verbose

   # Run robust optimization for noisy data
   heterodyne --config my_sample_config.json --method robust

**Step 4: View Results**

Results are saved to the ``heterodyne_results/`` directory with organized subdirectories:

- **Main results**: ``heterodyne_analysis_results.json`` with parameter estimates and fit quality
- **Classical output**: ``./classical/`` subdirectory with method-specific directories (``nelder_mead/``, ``gurobi/``)
- **Robust output**: ``./robust/`` subdirectory with method-specific directories (``wasserstein/``, ``scenario/``, ``ellipsoidal/``)
- **Experimental plots**: ``./exp_data/`` subdirectory with validation plots (if using ``--plot-experimental-data``)

**Method-Specific Outputs**:

- **Classical** (``./classical/``): Method-specific directories with fast point estimates, consolidated ``fitted_data.npz`` files
- **Robust** (``./robust/``): Noise-resistant optimization with method-specific directories (wasserstein, scenario, ellipsoidal)
- **All methods**: Save experimental, fitted, and residuals data in consolidated ``fitted_data.npz`` files per method

Python API Example (11-Parameter Heterodyne Model)
---------------------------------------------------

.. code-block:: python

   import numpy as np
   import json
   from heterodyne.analysis.core import HeterodyneAnalysisCore
   from heterodyne.optimization.classical import ClassicalOptimizer
   from heterodyne.optimization.robust import RobustHeterodyneOptimizer
   from heterodyne.data.xpcs_loader import load_xpcs_data

   # Load 11-parameter heterodyne configuration
   with open("my_heterodyne_config.json", 'r') as f:
       config = json.load(f)

   # Initialize analysis core with 11-parameter model
   core = HeterodyneAnalysisCore(config)

   # Load experimental data
   phi_angles = np.array([0, 36, 72, 108, 144])  # Example angles
   c2_data = load_xpcs_data(
       data_path=config['experimental_data']['data_folder_path'],
       phi_angles=phi_angles,
       n_angles=len(phi_angles)
   )

   # Run classical optimization
   classical = ClassicalOptimizer(core, config)
   params, results = classical.run_classical_optimization_optimized(
       phi_angles=phi_angles,
       c2_experimental=c2_data
   )

   # Display results for 11 parameters
   param_names = ["D0", "alpha", "D_offset", "v0", "beta", "v_offset",
                  "f0", "f1", "f2", "f3", "phi0"]
   print("Optimized Parameters:")
   for name, value in zip(param_names, params):
       print(f"  {name}: {value:.4e}")
   print(f"\nChi-squared: {results.chi_squared:.6e}")
   print(f"Best method: {results.best_method}")

   # For noisy data, use robust optimization
   robust = RobustHeterodyneOptimizer(core, config)
   robust_result = robust.optimize(
       phi_angles=phi_angles,
       c2_experimental=c2_data,
       method="wasserstein",  # Options: wasserstein, scenario, ellipsoidal
       epsilon=0.1  # Uncertainty radius
   )
   print(f"\nRobust optimization complete:")
   print(f"  D₀: {robust_result['optimal_params'][0]:.3e} Å²/s")
   print(f"  v₀: {robust_result['optimal_params'][3]:.3e} nm/s")
   print(f"  f₀: {robust_result['optimal_params'][6]:.3e}")


Heterodyne Model Quick Reference
----------------------------------

The package uses an 11-parameter two-component heterodyne scattering model:

**Parameter Groups:**

- **Diffusion (3)**: D₀, α, D_offset - Controls diffusive dynamics
- **Velocity (3)**: v₀, β, v_offset - Controls flow dynamics
- **Fraction (4)**: f₀, f₁, f₂, f₃ - Controls time-dependent component mixing
- **Flow Angle (1)**: φ₀ - Controls flow direction

**Analysis Strategy:**

1. **Start Simple**: Activate 4-6 essential parameters (D₀, α, v₀, f₀)
2. **Add Complexity**: Gradually add more parameters (β, f₁, f₂)
3. **Full Analysis**: Optimize all relevant parameters
4. **Robust Methods**: Use for noisy experimental data

Configuration Tips
------------------

**Quick 11-Parameter Heterodyne Configuration:**

.. code-block:: javascript

   {
     "metadata": {
       "config_version": "2.0",
       "analysis_mode": "heterodyne"
     },
     "file_paths": {
       "c2_data_file": "path/to/your/data.h5",
       "phi_angles_file": "path/to/angles.txt"
     },
     "initial_parameters": {
       "parameter_names": [
         "D0", "alpha", "D_offset",
         "v0", "beta", "v_offset",
         "f0", "f1", "f2", "f3",
         "phi0"
       ],
       "values": [1000.0, -0.5, 100.0, 0.01, 0.5, 0.001, 0.5, 0.0, 50.0, 0.3, 0.0],
       "active_parameters": ["D0", "alpha", "v0", "f0"]
     }
   }

**Performance Optimization:**

.. code-block:: javascript

   {
     "analysis_settings": {
       "enable_angle_filtering": true,
       "angle_filter_ranges": [[-5, 5], [175, 185]]
     },
     "performance_settings": {
       "num_threads": 4,
       "data_type": "float32"
     }
   }

Logging Control Options
-----------------------

The heterodyne package provides flexible logging control for different use cases:

.. list-table:: Logging Options
   :widths: 20 25 25 30
   :header-rows: 1

   * - Option
     - Console Output
     - File Output
     - Use Case
   * - **Default**
     - INFO level
     - INFO level
     - Normal interactive analysis
   * - ``--verbose``
     - DEBUG level
     - DEBUG level
     - Detailed troubleshooting
   * - ``--quiet``
     - None
     - INFO level
     - Batch processing, clean output

**Examples:**

.. code-block:: bash

   # Normal mode with INFO-level logging
   heterodyne --config my_config.json --method classical

   # Verbose mode with detailed debugging
   heterodyne --config my_config.json --method all --verbose

   # Quiet mode for batch processing (logs only to file)
   heterodyne --config my_config.json --method classical --quiet

   # Error: Cannot combine conflicting options
   heterodyne --verbose --quiet  # ERROR

**Important:** File logging is always enabled and saves to ``output_dir/run.log`` regardless of console settings.

Performance Features
--------------------

The heterodyne package includes advanced performance optimization and stability features:

**JIT Compilation Warmup**

Automatic Numba kernel pre-compilation eliminates JIT overhead:

.. code-block:: python

   from heterodyne.core.kernels import warmup_numba_kernels

   # Warmup all computational kernels
   warmup_results = warmup_numba_kernels()
   print(f"Kernels warmed up in {warmup_results['total_warmup_time']:.3f}s")

**Performance Monitoring**

Built-in performance monitoring with automatic optimization:

.. code-block:: python

   from heterodyne.core.config import performance_monitor

   # Monitor function performance
   def my_analysis():
       with performance_monitor.time_function("my_analysis"):
           # Your analysis code here
           pass

   # Access performance statistics
   stats = performance_monitor.get_timing_summary()
   print(f"Performance stats: {stats}")

**Benchmarking Tools**

Stable and adaptive benchmarking for research:

.. code-block:: python

   from heterodyne.core.profiler import stable_benchmark, adaptive_stable_benchmark

   # Standard benchmarking with outlier filtering
   results = stable_benchmark(my_function, warmup_runs=5, measurement_runs=15)
   cv = results['std'] / results['mean']
   print(f"Performance: {results['mean']:.4f}s ± {cv:.3f} CV")

   # Adaptive benchmarking (finds optimal measurement count)
   results = adaptive_stable_benchmark(my_function, target_cv=0.10)
   print(f"Achieved {results['cv']:.3f} CV in {results['total_runs']} runs")

**Performance Stability Achievements**

The heterodyne package has been optimized for excellent performance stability:

- **97% reduction** in chi-squared calculation variability (CV < 0.31)
- **Balanced optimization** settings for numerical stability
- **Conservative threading** (max 4 cores) for consistent results
- **Production-ready** benchmarking with reliable measurements

**Configuration Options**

Enable advanced performance features in your config:

.. code-block:: json

   {
     "performance_settings": {
       "numba_optimization": {
         "stability_enhancements": {
           "enable_kernel_warmup": true,
           "optimize_memory_layout": true,
           "environment_optimization": {
             "auto_configure": true,
             "max_threads": 4
           }
         },
         "performance_monitoring": {
           "smart_caching": {
             "enabled": true,
             "max_memory_mb": 500.0
           }
         }
       }
     }
   }

Next Steps
----------

- Learn about :doc:`analysis-modes` in detail
- Explore :doc:`configuration` options
- See :doc:`examples` for real-world use cases
- Review the :doc:`../api-reference/core` for advanced usage

Common First-Time Issues
-------------------------

**"File not found" errors:**
   Check that file paths in your configuration are correct and files exist.

**"Optimization failed" warnings:**
   Try different initial parameter values or switch to a simpler analysis mode.

**Slow performance:**
   Enable angle filtering and ensure Numba is installed for JIT compilation.


------------------------


.. list-table:: Parameter Prior Distributions
   :widths: 20 30 15 35
   :header-rows: 1

   * - Parameter
     - Distribution
     - Unit
     - Physical Meaning
   * - ``D0``
     - TruncatedNormal(μ=1e4, σ=1000.0, lower=1.0)
     - [Å²/s]
     - Reference diffusion coefficient
   * - ``alpha``
     - Normal(μ=-1.5, σ=0.1)
     - [dimensionless]
     - Time dependence exponent
   * - ``D_offset``
     - Normal(μ=0.0, σ=10.0)
     - [Å²/s]
     - Baseline diffusion component
   * - ``gamma_dot_t0``
     - TruncatedNormal(μ=1e-3, σ=1e-2, lower=1e-6)
     - [s⁻¹]
     - Reference shear rate
   * - ``beta``
     - Normal(μ=0.0, σ=0.1)
     - [dimensionless]
     - Shear exponent
   * - ``gamma_dot_t_offset``
     - Normal(μ=0.0, σ=1e-3)
     - [s⁻¹]
     - Baseline shear component
   * - ``phi0``
     - Normal(μ=0.0, σ=5.0)
     - [degrees]
     - Angular offset parameter

Scaling Parameters for Physical Constraints
--------------------------------------------


.. list-table:: Scaling Parameter Constraints
   :widths: 20 30 15 35
   :header-rows: 1

   * - Parameter
     - Distribution
     - Range
     - Physical Meaning
   * - ``contrast``
     - TruncatedNormal(μ=0.3, σ=0.1)
     - (0.05, 0.5]
     - Scaling factor for correlation strength
   * - ``offset``
     - TruncatedNormal(μ=1.0, σ=0.2)
     - (0.05, 1.95)
     - Baseline correlation level
   * - ``c2_fitted``
     - -
     - [1.0, 2.0]
     - Final correlation function range
   * - ``c2_theory``
     - -
     - [0.0, 1.0]
     - Theoretical correlation function range

The relationship is: **c2_fitted = c2_theory × contrast + offset**

**Configuration Format:**

.. code-block:: json

   {
     "parameter_space": {
       "bounds": [
         {"name": "D0", "min": 1.0, "max": 1000000, "type": "Normal"},
         {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
         {"name": "D_offset", "min": -100, "max": 100, "type": "Normal"}
       ]
     }
   }
