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

This package analyzes time-dependent intensity correlation functions c₂(φ,t₁,t₂) in complex fluids under nonequilibrium conditions, capturing the interplay between Brownian diffusion and advective shear flow. The implementation provides:

- **Heterodyne Scattering Model** (11 parameters): Two-component system with time-dependent fraction mixing

  - Diffusion (3 params): D₀, α, D_offset
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

   print(f"D₀ = {params[0]:.3e} Å²/s")
   print(f"χ² = {results.chi_squared:.6e}")

**Command Line Interface:**

.. code-block:: bash

   # Create heterodyne configuration (11 parameters)
   cp heterodyne/config/heterodyne_11param_example.json my_config.json
   # Edit my_config.json with your experimental parameters

   # Main analysis command
   heterodyne --config my_config.json            # Run with 11-parameter heterodyne model
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

**11-Parameter Heterodyne Model (PNAS 2024)**

* **Two-component heterodyne scattering**: Implements He et al. PNAS 2024 **Equation S-95** (general time-dependent form) with transport coefficients
* **Transport coefficient approach**: Uses J(t) directly for nonequilibrium dynamics (J = 6D for equilibrium Wiener processes)
* **Comprehensive parameter set**: 11 parameters covering transport (3), velocity (3), fraction (4), and flow angle (1)
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

Heterodyne Model (11 Parameters)
---------------------------------

The package implements the **general two-component heterodyne scattering model** from `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_ **Equation S-95**, which uses time-dependent transport coefficients J(t) for nonequilibrium dynamics.

**Model Equation (Equation S-95):**

The implementation uses Equation S-95 (general time-dependent form) with transport coefficients:

.. math::

   c_2(\vec{q}, t_1, t_2) = 1 + \frac{\beta}{f^2} \left[
   [x_r(t_1)x_r(t_2)]^2 \exp\left(-q^2 \int_{t_1}^{t_2} J(t) dt\right) +
   [x_s(t_1)x_s(t_2)]^2 \exp\left(-q^2 \int_{t_1}^{t_2} J(t) dt\right) +
   2x_r(t_1)x_r(t_2)x_s(t_1)x_s(t_2) \exp\left(-q^2 \int_{t_1}^{t_2} J(t) dt\right) \cos(...)
   \right]

where:

* :math:`J(t)` - Time-dependent transport coefficient [Å²/s]
* :math:`x_n(t)` - Time-dependent fraction of component n (reference/sample)
* :math:`\mathbb{E}[v(t)]` - Time-dependent mean velocity
* :math:`\phi` - Angle between scattering vector and flow direction
* :math:`\beta` - Contrast factor
* :math:`f^2` - Normalization factor

**Relationship to Equilibrium Form (Equation S-98):**

For equilibrium Wiener processes, the transport coefficient reduces to J = 6D, where D is the traditional diffusion coefficient. Equation S-98 is the equilibrium simplification. This package implements the more general S-95.

**Nonequilibrium Implementation:**

This package parameterizes J(t) as :math:`J(t) = J_0 \cdot t^\alpha + J_{offset}` to capture nonequilibrium dynamics including aging, yielding, and shear banding phenomena. Parameters labeled "D" in the code (D₀, α, D_offset) are actually transport coefficient parameters (J₀, α, J_offset) for historical compatibility.

**Parameter Categories:**

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Category
     - Count
     - Parameters
   * - **Diffusion**
     - 3
     - D₀ (reference diffusion coefficient, nm²/s), α (power-law exponent), D_offset (baseline offset, nm²/s)
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

**Heterodyne Scattering Model (11 Parameters)**
   Two-component system with time-dependent fraction mixing, covering diffusion, velocity, fraction dynamics, and flow angle

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
