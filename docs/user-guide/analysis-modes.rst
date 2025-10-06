Heterodyne Scattering Model
============================

The heterodyne package implements the two-component heterodyne scattering model from `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_ (Equations S-95 to S-98), generalized to nonequilibrium conditions with time-dependent transport coefficients.

Theoretical Foundation
----------------------

The package uses the **commonly used heterodyne equation** (Equation S-98 from He et al. PNAS 2024) for equilibrium systems:

**Model Equation (Equation S-98)**:

.. math::

   g_2(\vec{q}, \tau) = 1 + \beta \left[(1-x)^2 e^{-6q^2 D_r \tau} + x^2 e^{-6q^2 D_s \tau} + 2x(1-x)e^{-3q^2(D_r+D_s)\tau} \cos(q \cos(\phi)\mathbb{E}[v]\tau)\right]

where:

- **x**: Composition fraction (sample intensity ratio)
- **Dᵣ, Dₛ**: Diffusion coefficients for reference and sample components
- **𝔼[v]**: Mean velocity of sample component
- **φ**: Angle between scattering vector and flow direction
- **τ = t₂ - t₁**: Delay time
- **β**: Contrast factor

**Nonequilibrium Extension**: This package extends Equation S-98 to time-dependent nonequilibrium dynamics where transport coefficients evolve with time, capturing aging, yielding, and shear banding phenomena in soft matter systems.

**Time-Dependent Fraction**:

The fraction evolves according to:

.. math::

   f(t) = f_0 \times \exp(f_1 \times (t - f_2)) + f_3

with physical constraint: :math:`0 \leq f(t) \leq 1` for all times.

11-Parameter System
-------------------

The model uses 11 parameters organized into four groups:

**Diffusion Parameters (3)**:

.. list-table::
   :widths: 15 25 15 45
   :header-rows: 1

   * - Parameter
     - Range
     - Units
     - Physical Meaning
   * - **D₀**
     - [1.0, 1×10⁶]
     - Å²/s
     - Reference diffusion coefficient at t=1s
   * - **α**
     - [-2.0, 2.0]
     - dimensionless
     - Diffusion time-scaling exponent
   * - **D_offset**
     - [-100, 100]
     - Å²/s
     - Baseline diffusion contribution

**Velocity Parameters (3)**:

.. list-table::
   :widths: 15 25 15 45
   :header-rows: 1

   * - Parameter
     - Range
     - Units
     - Physical Meaning
   * - **v₀**
     - [1×10⁻⁵, 10.0]
     - nm/s
     - Reference velocity at t=1s
   * - **β**
     - [-2.0, 2.0]
     - dimensionless
     - Velocity time-scaling exponent
   * - **v_offset**
     - [-0.1, 0.1]
     - nm/s
     - Baseline velocity contribution

**Fraction Parameters (4)**:

.. list-table::
   :widths: 15 25 15 45
   :header-rows: 1

   * - Parameter
     - Range
     - Units
     - Physical Meaning
   * - **f₀**
     - [0.0, 1.0]
     - dimensionless
     - Fraction amplitude
   * - **f₁**
     - [-1.0, 1.0]
     - s⁻¹
     - Exponential rate of fraction change
   * - **f₂**
     - [0.0, 200.0]
     - s
     - Time offset for fraction dynamics
   * - **f₃**
     - [0.0, 1.0]
     - dimensionless
     - Baseline fraction value

**Flow Angle (1)**:

.. list-table::
   :widths: 15 25 15 45
   :header-rows: 1

   * - Parameter
     - Range
     - Units
     - Physical Meaning
   * - **φ₀**
     - [-10, 10]
     - degrees
     - Flow direction angle

Physical Interpretation
-----------------------

**Diffusion Contribution**:

The time-dependent diffusion coefficient is:

.. math::

   D(t) = D_0 \times t^\alpha + D_{\text{offset}}

This captures:

- **Anomalous diffusion**: α ≠ 0 indicates sub-diffusive (α < 0) or super-diffusive (α > 0) dynamics
- **Aging effects**: Time-dependent dynamics in nonequilibrium systems
- **Baseline diffusion**: Constant background diffusion contribution

**Velocity Contribution**:

The time-dependent velocity is:

.. math::

   v(t) = v_0 \times t^\beta + v_{\text{offset}}

This describes:

- **Flow acceleration/deceleration**: β controls temporal evolution of flow
- **Steady flow**: β = 0 with v₀ > 0 gives constant flow velocity
- **Transient flow**: Non-zero β captures time-dependent flow dynamics

**Fraction Dynamics**:

The time-dependent fraction mixing describes:

- **Component evolution**: How reference and sample contributions change over time
- **Relaxation dynamics**: f₁ controls the rate of exponential relaxation
- **Equilibrium state**: f₃ represents the long-time steady-state fraction
- **Initial conditions**: f₀ and f₂ control the amplitude and temporal offset

Configuration Examples
-----------------------

**Full 11-Parameter Heterodyne Configuration**:

.. code-block:: javascript

   {
     "metadata": {
       "config_version": "2.0",
       "analysis_mode": "heterodyne"
     },
     "initial_parameters": {
       "parameter_names": [
         "D0", "alpha", "D_offset",
         "v0", "beta", "v_offset",
         "f0", "f1", "f2", "f3",
         "phi0"
       ],
       "values": [1000.0, -0.5, 100.0, 0.01, 0.5, 0.001, 0.5, 0.0, 50.0, 0.3, 0.0],
       "active_parameters": ["D0", "alpha", "v0", "beta", "f0", "f1"]
     },
     "parameter_space": {
       "bounds": [
         {"name": "D0", "min": 1.0, "max": 1000000, "type": "Normal"},
         {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
         {"name": "D_offset", "min": -100, "max": 100, "type": "Normal"},
         {"name": "v0", "min": 1e-5, "max": 10.0, "type": "Normal"},
         {"name": "beta", "min": -2.0, "max": 2.0, "type": "Normal"},
         {"name": "v_offset", "min": -0.1, "max": 0.1, "type": "Normal"},
         {"name": "f0", "min": 0.0, "max": 1.0, "type": "Normal"},
         {"name": "f1", "min": -1.0, "max": 1.0, "type": "Normal"},
         {"name": "f2", "min": 0.0, "max": 200.0, "type": "Normal"},
         {"name": "f3", "min": 0.0, "max": 1.0, "type": "Normal"},
         {"name": "phi0", "min": -10, "max": 10, "type": "Normal"}
       ]
     }
   }

**Simplified Configuration (Fewer Active Parameters)**:

For initial exploration, you can fix some parameters:

.. code-block:: javascript

   {
     "initial_parameters": {
       "parameter_names": [
         "D0", "alpha", "D_offset",
         "v0", "beta", "v_offset",
         "f0", "f1", "f2", "f3",
         "phi0"
       ],
       "values": [1000.0, -0.5, 0.0, 0.01, 0.0, 0.0, 0.5, 0.0, 50.0, 0.3, 0.0],
       "active_parameters": ["D0", "alpha", "v0", "f0"]  // Optimize only 4 parameters
     }
   }

Analysis Workflow
-----------------

**1. Initial Exploration**:

Start with a subset of active parameters:

.. code-block:: bash

   # Optimize only diffusion parameters
   heterodyne --config config.json --method classical

**2. Incremental Complexity**:

Gradually add more parameters:

.. code-block:: bash

   # Add velocity parameters
   # Edit config to include v0, beta in active_parameters
   heterodyne --config config.json --method classical

**3. Full Optimization**:

Optimize all relevant parameters:

.. code-block:: bash

   # Full parameter optimization with robust methods
   heterodyne --config config.json --method all

**4. Robust Optimization for Noisy Data**:

Use robust methods for experimental data with uncertainty:

.. code-block:: bash

   # Wasserstein DRO for outlier resistance
   heterodyne --config config.json --method robust

Parameter Selection Guidelines
-------------------------------

**Start with Essential Parameters**:

- **D₀, α**: Core diffusion dynamics
- **v₀**: Flow velocity (if flow present)
- **f₀**: Reference/sample mixing amplitude

**Add Complexity as Needed**:

- **β**: If flow shows time-dependent behavior
- **f₁, f₂**: If fraction mixing shows temporal dynamics
- **D_offset, v_offset**: For baseline corrections
- **f₃**: For steady-state fraction adjustment
- **φ₀**: For flow direction refinement

**Physical Constraints**:

The package automatically enforces:

- **D(t) ≥ 1×10⁻¹⁰**: Positive diffusion coefficient
- **v(t) ≥ 1×10⁻¹⁰**: Positive velocity
- **0 ≤ f(t) ≤ 1**: Valid fraction range

Best Practices
--------------

**1. Validate Experimental Data**:

.. code-block:: bash

   heterodyne --config config.json --plot-experimental-data

**2. Start Simple**:

Begin with fewer active parameters and add complexity incrementally.

**3. Check Convergence**:

Monitor chi-squared values and parameter uncertainties in results.

**4. Use Robust Methods for Noisy Data**:

Wasserstein DRO, scenario-based, or ellipsoidal methods handle uncertainty better than classical optimization.

**5. Physical Interpretation**:

Ensure fitted parameters have physically meaningful values and interpretations.

Troubleshooting
---------------

**Poor Convergence**:
   - Reduce number of active parameters
   - Adjust initial parameter values
   - Try different optimization methods

**Unphysical Parameters**:
   - Check parameter bounds in configuration
   - Verify experimental data quality
   - Review fraction constraint: 0 ≤ f(t) ≤ 1

**High Chi-Squared**:
   - Increase number of active parameters
   - Use robust optimization methods
   - Check for systematic errors in data

**Fraction Constraint Violations**:
   - Adjust f₀, f₁, f₂, f₃ bounds
   - Ensure f(t) stays within [0, 1] for all times
   - Review fraction dynamics physical interpretation

See Also
--------

- :doc:`configuration` - Detailed configuration guide
- :doc:`../api-reference/analysis-core` - Core analysis API
- :doc:`../developer-guide/optimization` - Optimization strategies
- :doc:`quickstart` - Quick start tutorial
