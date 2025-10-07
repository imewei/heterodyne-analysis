Heterodyne Scattering Analysis Package
====================================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/Python-3.12%2B-blue
   :target: https://www.python.org/
   :alt: Python

.. image:: https://img.shields.io/badge/Numba-JIT%20Accelerated-green
   :target: https://numba.pydata.org/
   :alt: Numba

A high-performance Python package for analyzing heterodyne scattering in X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium conditions. Implements the theoretical framework from `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_ for characterizing transport properties in flowing soft matter systems.

Overview
--------

This package analyzes two-component heterodyne X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium conditions. The implementation provides:

- **Heterodyne Scattering Model** (14 parameters): Two-component heterodyne with separate reference and sample correlations

  - Reference transport (3 params): D₀_ref, α_ref, D_offset_ref
  - Sample transport (3 params): D₀_sample, α_sample, D_offset_sample
  - Velocity (3 params): v₀, β, v_offset
  - Fraction (4 params): f₀, f₁, f₂, f₃
  - Flow angle (1 param): φ₀

- **Multiple optimization methods**: Classical (Nelder-Mead, Powell), Robust (Wasserstein DRO, Scenario-based, Ellipsoidal)
- **High performance**: Numba JIT compilation with 3-5x speedup, vectorized NumPy operations, comprehensive performance monitoring
- **Scientific accuracy**: Automatic c₂ = offset + contrast × c₁ fitting for proper chi-squared calculations

Quick Start
-----------

**Installation:**

.. code-block:: bash

   pip install heterodyne-analysis[all]

**Python API:**

.. code-block:: python

   import numpy as np
   import json
   from heterodyne.analysis.core import HeterodyneAnalysisCore
   from heterodyne.optimization.classical import ClassicalOptimizer
   from heterodyne.data.xpcs_loader import load_xpcs_data

   # Load configuration
   with open("config.json", 'r') as f:
       config = json.load(f)

   # Initialize analysis core
   core = HeterodyneAnalysisCore(config)

   # Load experimental data
   phi_angles = np.array([0, 36, 72, 108, 144])
   c2_data = load_xpcs_data(
       data_path=config['experimental_data']['data_folder_path'],
       phi_angles=phi_angles,
       n_angles=len(phi_angles)
   )

   # Run optimization
   optimizer = ClassicalOptimizer(core, config)
   params, results = optimizer.run_classical_optimization_optimized(
       phi_angles=phi_angles,
       c2_experimental=c2_data
   )

   # Extract parameters (14 total)
   D0_ref, alpha_ref, D_offset_ref = params[0:3]
   D0_sample, alpha_sample, D_offset_sample = params[3:6]
   v0, beta, v_offset = params[6:9]
   f0, f1, f2, f3 = params[9:13]
   phi0 = params[13]

   print(f"D₀_ref = {D0_ref:.3e} Å²/s")
   print(f"D₀_sample = {D0_sample:.3e} Å²/s")
   print(f"χ² = {results.chi_squared:.6e}")

**Command Line Interface:**

.. code-block:: bash

   # Create heterodyne configuration (14 parameters)
   cp heterodyne/config/template.json my_config.json
   # Edit my_config.json with your experimental parameters

   # Main analysis command
   heterodyne --config my_config.json            # Run with 14-parameter heterodyne model
   heterodyne --method robust                    # Robust optimization for noisy data
   heterodyne --method all --verbose             # All methods with debug logging

   # Data visualization
   heterodyne --plot-experimental-data           # Validate experimental data
   heterodyne --plot-simulated-data              # Plot theoretical correlations
   heterodyne --plot-simulated-data --contrast 1.5 --offset 0.1 --phi-angles "0,45,90,135"

   # Configuration and output
   heterodyne --config my_config.json --output-dir ./results --verbose
   heterodyne --quiet                            # File logging only, no console output

Core Features
-------------

**14-Parameter Heterodyne Model (PNAS 2024)**

* **Two-component heterodyne scattering**: Implements He et al. PNAS 2024 **Equation S-95** with separate reference and sample field correlations
* **Separate transport coefficients**: Independent reference and sample transport parameters for comprehensive characterization
* **Comprehensive parameter set**: 14 parameters covering reference transport (3), sample transport (3), velocity (3), fraction (4), and flow angle (1)
* **Time-dependent fraction**: ``f(t) = f₀ × exp(f₁ × (t - f₂)) + f₃`` with physical constraint ``0 ≤ f(t) ≤ 1``
* **Physical constraint enforcement**: Automatic validation during optimization to ensure meaningful results

**Robust Data Handling**

* **Frame counting convention**: 1-based inclusive counting with proper conversion to 0-based Python slicing
* **Conditional angle subsampling**: Preserves angular information when ``n_angles < 4``
* **Memory optimization**: Handles large datasets with 8M+ data points efficiently
* **Smart caching**: Intelligent data caching with automatic dimension validation

**High Performance**

* **Numba JIT compilation**: 3-5x speedup for core calculations
* **Vectorized operations**: Optimized NumPy array processing throughout
* **Computational efficiency**: Optimized algorithms for large-scale XPCS data analysis

Heterodyne Model (14 Parameters)
---------------------------------

The package implements the **two-component heterodyne scattering model** from `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_ **Equation S-95**, with separate reference and sample field correlations.

**Model Equation (Equation S-95):**

The heterodyne correlation g₂ is composed of separate reference and sample field correlations:

.. math::

   g_2 = \text{offset} + \text{contrast} \times |g_1^{\text{ref}} + g_1^{\text{sample}}|^2

where each field correlation has independent time-dependent transport:

.. math::

   g_1^{\text{ref}}(t_1, t_2) &= \exp\left(-q^2 \int_{t_1}^{t_2} D_{\text{ref}}(t) dt\right) \\
   g_1^{\text{sample}}(t_1, t_2) &= \exp\left(-q^2 \int_{t_1}^{t_2} D_{\text{sample}}(t) dt\right)

**Transport Parameterization:**

Each transport coefficient is parameterized as power-law with offset:

.. math::

   D_{\text{ref}}(t) &= D_{0,\text{ref}} \cdot t^{\alpha_{\text{ref}}} + D_{\text{offset,ref}} \\
   D_{\text{sample}}(t) &= D_{0,\text{sample}} \cdot t^{\alpha_{\text{sample}}} + D_{\text{offset,sample}}

**Parameter Categories:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Category
     - Count
     - Parameters
   * - **Reference Transport**
     - 3
     - D₀_ref (reference diffusion coefficient, nm²/s), α_ref (power-law exponent), D_offset_ref (baseline offset, nm²/s)
   * - **Sample Transport**
     - 3
     - D₀_sample (sample diffusion coefficient, nm²/s), α_sample (power-law exponent), D_offset_sample (baseline offset, nm²/s)
   * - **Velocity**
     - 3
     - v₀ (reference velocity, nm/s), β (power-law exponent), v_offset (baseline offset, nm/s)
   * - **Fraction**
     - 4
     - f₀ (amplitude), f₁ (exponential rate, 1/s), f₂ (time offset, s), f₃ (baseline)
   * - **Flow Angle**
     - 1
     - φ₀ (flow direction angle, degrees)

**Time-Dependent Fraction:**

.. math::

   f(t) = f_0 \times \exp(f_1 \times (t - f_2)) + f_3

with physical constraint :math:`0 \leq f(t) \leq 1` for all times (enforced during validation).

Key Features
------------

**Heterodyne Scattering Model (14 Parameters)**
   Two-component heterodyne with separate reference and sample correlations, covering reference transport, sample transport, velocity, fraction dynamics, and flow angle

**Multiple Optimization Methods**
   Classical (Nelder-Mead, Powell) and Robust (Wasserstein DRO, Scenario-based, Ellipsoidal) optimization with comprehensive parameter validation

**High Performance**
   Numba JIT compilation (3-5x speedup), vectorized NumPy operations, and optimized computational kernels

**Scientific Accuracy**
   Automatic c₂ = offset + contrast × c₁ fitting for accurate chi-squared calculations with physical constraint enforcement

**Security and Code Quality**
   Comprehensive security scanning with Bandit, dependency vulnerability checking with pip-audit, and automated code quality tools

**Comprehensive Validation**
   Experimental data validation plots, quality control, and integration testing for all parameter configurations

**Visualization Tools**
   Experimental data validation plots, simulated correlation heatmaps, and method comparison visualizations

**Performance Monitoring**
   Comprehensive performance testing, regression detection, and automated benchmarking

User Guide
----------

.. toctree::
   :maxdepth: 2

   user-guide/installation
   user-guide/quickstart
   user-guide/analysis-modes
   user-guide/configuration
   user-guide/plotting
   user-guide/ml-acceleration
   user-guide/examples

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api-reference/core
   api-reference/utilities

Developer Guide
---------------

.. toctree::
   :maxdepth: 2

   developer-guide/contributing
   developer-guide/testing
   developer-guide/performance
   developer-guide/architecture
   developer-guide/troubleshooting

Theoretical Background
----------------------

The package implements three key equations describing correlation functions in nonequilibrium laminar flow systems:

**Equation 13 - Full Nonequilibrium Laminar Flow:**
   c₂(q⃗, t₁, t₂) = 1 + β[e^(-q²∫J(t)dt)] × sinc²[1/(2π) qh ∫γ̇(t)cos(φ(t))dt]

**Equation S-75 - Equilibrium Under Constant Shear:**
   c₂(q⃗, t₁, t₂) = 1 + β[e^(-6q²D(t₂-t₁))] sinc²[1/(2π) qh cos(φ)γ̇(t₂-t₁)]

**Equation S-76 - One-time Correlation (Siegert Relation):**
   c₂(q⃗, τ) = 1 + β[e^(-6q²Dτ)] sinc²[1/(2π) qh cos(φ)γ̇τ]

**Key Parameters:**

- q⃗: scattering wavevector [Å⁻¹]
- h: gap between stator and rotor [Å]
- φ(t): angle between shear/flow direction and q⃗ [degrees]
- γ̇(t): time-dependent shear rate [s⁻¹]
- D(t): time-dependent diffusion coefficient [Å²/s]
- β: contrast parameter [dimensionless]

Citation
--------

If you use this package in your research, please cite:

.. code-block:: bibtex

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

Support
-------

- **Documentation**: https://heterodyne.readthedocs.io/
- **Issues**: https://github.com/imewei/heterodyne/issues
- **Source Code**: https://github.com/imewei/heterodyne
- **License**: MIT License

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
