Theoretical Framework
=====================

This section provides the comprehensive mathematical foundation for the heterodyne scattering analysis
implemented in this package, based on the theoretical framework by He et al. (2024) [1]_.

.. [1] He, H., Liang, H., Chu, M., et al. (2024). Transport coefficient approach for characterizing
   nonequilibrium dynamics in soft matter. *Proceedings of the National Academy of Sciences*,
   121(31), e2401162121. https://doi.org/10.1073/pnas.2401162121

General N-Component Heterogeneous System
-----------------------------------------

Fundamental Correlation Function (Equation 14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a heterogeneous system combining or mixing N components (such as in shear banding scenarios),
each component exhibits time-dependent properties indexed by *n*:

* **Fraction**: :math:`x_n(t)` - time-dependent component fraction
* **Transport coefficient**: :math:`J_n(t)` - characterizes diffusive dynamics
* **Mean velocity**: :math:`\mathbb{E}[v_n(t)]` - advective transport
* **Flow angle**: :math:`\phi_n(t)` - direction relative to scattering vector

The two-time intensity correlation function is given by:

.. math::

   c_2(\vec{q}, t_1, t_2) = 1 + \frac{\beta}{f(t_1,t_2)^2} \sum_{n=1}^{N} \sum_{m=1}^{N} \Bigg[
   x_n(t_1)x_n(t_2)x_m(t_1)x_m(t_2) \times \\
   \exp\left(-\frac{1}{2}q^2 \int_{t_1}^{t_2} [J_n(t)+J_m(t)] dt\right) \times \\
   \cos\left[q \int_{t_1}^{t_2} \mathbb{E}[v_n(t)]\cos(\phi_n(t)) - \mathbb{E}[v_m(t)]\cos(\phi_m(t)) dt\right]
   \Bigg]

where the normalization factor is:

.. math::

   f(t_1,t_2)^2 = \sum_{n=1}^{N} x_n(t_1)^2 \sum_{n=1}^{N} x_n(t_2)^2

This general framework describes complex multi-component systems with heterogeneous dynamics.

Two-Component Heterodyne Scattering
------------------------------------

Specialized Model (Equations S-95 to S-98)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For practical implementation, we consider the widely-used two-component heterodyne configuration
(SI Appendix, Section F.2) where:

* **Reference component (r)**: Static scatterers providing reference signal
* **Sample component (s)**: Dynamic scatterers exhibiting flow and diffusion

The correlation function simplifies to (Equation S-95):

.. math::

   c_2(\vec{q}, t_1, t_2) = 1 + \frac{\beta}{f^2} \Bigg[
   [x_r(t_1)x_r(t_2)]^2 e^{-q^2 \int_{t_1}^{t_2} J_r(t) dt} + \\
   [x_s(t_1)x_s(t_2)]^2 e^{-q^2 \int_{t_1}^{t_2} J_s(t) dt} + \\
   2x_r(t_1)x_r(t_2)x_s(t_1)x_s(t_2)e^{-\frac{1}{2}q^2 \int_{t_1}^{t_2} [J_s(t)+J_r(t)] dt}
   \cos\left[q \cos(\phi) \int_{t_1}^{t_2} \mathbb{E}[v] dt\right]
   \Bigg]

where:

.. math::

   f^2 = [x_s(t_1)^2 + x_r(t_1)^2][x_s(t_2)^2 + x_r(t_2)^2]

Equilibrium Wiener Process Form (Equation S-96)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For equilibrium conditions where all parameters are time-independent and the transport coefficient
follows the Wiener process :math:`J_n(t) = 6D_n`, the equation simplifies to:

.. math::

   c_2(\vec{q}, t_1, t_2) = 1 + \frac{\beta}{f^2} \Bigg[
   x_r^4 e^{-6q^2 D_r \tau} + x_s^4 e^{-6q^2 D_s \tau} + \\
   2x_r^2 x_s^2 e^{-3q^2(D_r+D_s)\tau} \cos[q \cos(\phi)\mathbb{E}[v]\tau]
   \Bigg]

where :math:`\tau = t_2 - t_1` is the delay time and:

.. math::

   f^2 = [x_s^2 + x_r^2]^2

One-Time Correlation Form (Equation S-98)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defining the composition fraction :math:`x = \frac{I_s}{I_s + I_r} = \frac{x_s^2 \mathbb{E}[I]}{x_s^2\mathbb{E}[I] + x_r^2\mathbb{E}[I]} = \frac{x_s^2}{f}`,
the **commonly used heterodyne equation** becomes:

.. math::

   g_2(\vec{q}, \tau) = 1 + \beta \Bigg[
   (1-x)^2 e^{-6q^2 D_r \tau} + x^2 e^{-6q^2 D_s \tau} + \\
   2x(1-x)e^{-3q^2(D_r+D_s)\tau} \cos[q \cos(\phi)\mathbb{E}[v]\tau]
   \Bigg]

This form (Equation S-98) has been widely applied in heterodyne X-ray photon correlation spectroscopy (XPCS) studies.

Package Implementation: Equation S-95 with Transport Coefficients
-------------------------------------------------------------------

What This Package Implements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This package implements **Equation S-95** (the general time-dependent two-component form), NOT the equilibrium
Equation S-98. The key differences are:

**Equation S-95 (Implemented):**
   Uses time-dependent transport coefficients :math:`J(t)` with integrals :math:`\int J(t) dt`

**Equation S-98 (Reference form):**
   Uses equilibrium diffusion coefficients :math:`D` with :math:`J = 6D` relationship

Transport Coefficient vs Diffusion Coefficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Transport Coefficient J(t):**
   - General parameterization for nonequilibrium dynamics
   - Units: [Å²/s] (same as diffusion)
   - Direct implementation: :math:`\exp(-q^2 \int J(t) dt)`
   - Code uses: :math:`J(t) = J_0 \cdot t^\alpha + J_{\text{offset}}`

**Diffusion Coefficient D:**
   - Traditional equilibrium concept
   - For Wiener process: :math:`J = 6D`
   - Equilibrium form: :math:`\exp(-6q^2 D \tau)`

**Important:** Parameters labeled "D₀", "α", "D_offset" in the code are actually transport coefficient
parameters (J₀, α, J_offset) for historical compatibility.

Implementation: Separate Reference and Sample Transport
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full Equation S-95 allows different transport coefficients for reference and sample:

.. math::

   J_r(t) \neq J_s(t)

This package implements the **full two-component model** with separate transport coefficients for reference and sample fields:

.. math::

   D_{\text{ref}}(t) &= D_{0,\text{ref}} \cdot t^{\alpha_{\text{ref}}} + D_{\text{offset,ref}} \\
   D_{\text{sample}}(t) &= D_{0,\text{sample}} \cdot t^{\alpha_{\text{sample}}} + D_{\text{offset,sample}}

This enables comprehensive characterization of the distinct transport properties of both components,
which is essential for heterodyne measurements where reference and sample exhibit different dynamics.

14-Parameter Nonequilibrium Extension
--------------------------------------

Time-Dependent Parameterization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As described above, this package implements **Equation S-95** with **separate time-dependent transport coefficients** for reference and sample fields.
The model uses 14 parameters organized into five groups:

**1. Reference Transport Dynamics (3 parameters)**

.. math::

   D_{\text{ref}}(t) = D_{0,\text{ref}} \cdot t^{\alpha_{\text{ref}}} + D_{\text{offset,ref}}

* :math:`D_{0,\text{ref}}` [Å²/s]: Reference field transport coefficient
* :math:`\alpha_{\text{ref}}` [dimensionless]: Reference time-scaling exponent
* :math:`D_{\text{offset,ref}}` [Å²/s]: Reference baseline transport component

**2. Sample Transport Dynamics (3 parameters)**

.. math::

   D_{\text{sample}}(t) = D_{0,\text{sample}} \cdot t^{\alpha_{\text{sample}}} + D_{\text{offset,sample}}

* :math:`D_{0,\text{sample}}` [Å²/s]: Sample field transport coefficient
* :math:`\alpha_{\text{sample}}` [dimensionless]: Sample time-scaling exponent
* :math:`D_{\text{offset,sample}}` [Å²/s]: Sample baseline transport component

**3. Velocity Dynamics (3 parameters)**

.. math::

   v(t) = v_0 \cdot (t-t_0)^{\beta} + v_{\text{offset}}

* :math:`v_0` [nm/s]: Reference velocity
* :math:`\beta` [dimensionless]: Velocity scaling exponent
* :math:`v_{\text{offset}}` [nm/s]: Baseline velocity component

**4. Time-Dependent Fraction (4 parameters)**

.. math::

   f(t) = f_0 \cdot \exp[f_1(t - f_2)] + f_3

with constraint :math:`0 \leq f(t) \leq 1`

* :math:`f_0` [dimensionless]: Amplitude of exponential component
* :math:`f_1` [1/s]: Exponential rate constant
* :math:`f_2` [s]: Time shift parameter
* :math:`f_3` [dimensionless]: Constant offset

**5. Flow Geometry (1 parameter)**

* :math:`\phi_0` [degrees]: Flow direction angle relative to scattering vector

Nonequilibrium Correlation Function (As Implemented)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implemented two-time correlation function uses **Equation S-95** with **separate transport coefficients**
for reference and sample fields:

.. math::

   g_2 = \text{offset} + \text{contrast} \times |g_1^{\text{ref}} + g_1^{\text{sample}}|^2

where each field correlation has independent transport:

.. math::

   g_1^{\text{ref}}(t_1, t_2) &= \exp\left(-q^2 \int_{t_1}^{t_2} D_{\text{ref}}(t) dt\right) \\
   g_1^{\text{sample}}(t_1, t_2) &= \exp\left(-q^2 \int_{t_1}^{t_2} D_{\text{sample}}(t) dt\right)

with separate power-law transport:

.. math::

   D_{\text{ref}}(t) &= D_{0,\text{ref}} \cdot t^{\alpha_{\text{ref}}} + D_{\text{offset,ref}} \\
   D_{\text{sample}}(t) &= D_{0,\text{sample}} \cdot t^{\alpha_{\text{sample}}} + D_{\text{offset,sample}}

**Key Implementation Features:**

* **Separate transport coefficients**: Independent D_ref(t) and D_sample(t) for comprehensive characterization
* **Two-component heterodyne**: Full heterodyne correlation with distinct reference and sample dynamics
* **Time-dependent fraction**: :math:`f(t) = f_0 \cdot \exp[f_1(t - f_2)] + f_3`
* **Aging dynamics**: Power-law time dependence of both transport coefficients
* **Transient flow**: Time-evolving velocity fields v(t)
* **Component evolution**: Dynamic changes in composition fractions f(t)
* **Nonequilibrium structure**: Departure from equilibrium Wiener process

Physical Interpretation
------------------------

Transport Coefficient Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transport coefficient :math:`J(t)` generalizes the diffusion coefficient to nonequilibrium
conditions. For standard Brownian motion, :math:`J(t) = 6D`, but under nonequilibrium conditions
(aging, yielding, shear banding), :math:`J(t)` can exhibit complex time dependence.

**Key Features:**

* **Aging systems**: :math:`\alpha < 0` indicates slowing dynamics (approaching glass transition)
* **Rejuvenation**: :math:`\alpha > 0` indicates accelerating dynamics (shear rejuvenation)
* **Steady state**: :math:`\alpha = 0` recovers time-independent diffusion

Component Mixing Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~

The time-dependent fraction :math:`f(t)` describes the evolution of the intensity ratio between
reference and sample components:

* **Shear banding**: Rapid changes in :math:`f(t)` indicate band formation/destruction
* **Steady shear**: Constant :math:`f(t)` indicates stable two-phase flow
* **Yielding transition**: Monotonic change in :math:`f(t)` tracks yield dynamics

Flow Orientation
~~~~~~~~~~~~~~~~

The angle :math:`\phi_0` characterizes the flow direction relative to the scattering geometry:

* :math:`\phi_0 = 0°`: Flow parallel to scattering vector (maximum Doppler effect)
* :math:`\phi_0 = 90°`: Flow perpendicular to scattering vector (no advective contribution)

Scattering Geometry
-------------------

Wavevector Definition
~~~~~~~~~~~~~~~~~~~~~

The scattering wavevector magnitude is:

.. math::

   q = \frac{4\pi}{\lambda} \sin\left(\frac{\theta}{2}\right)

where :math:`\lambda` is the X-ray wavelength and :math:`\theta` is the scattering angle.

Multi-Angle Analysis
~~~~~~~~~~~~~~~~~~~~~

The correlation function is measured at multiple scattering angles :math:`\phi_i` to capture
the angular dependence of the dynamics. This enables:

* **Flow characterization**: Extracting velocity magnitude and direction
* **Anisotropy quantification**: Measuring directional variations in dynamics
* **Component separation**: Distinguishing reference and sample contributions

Optimization Framework
----------------------

Parameter Estimation
~~~~~~~~~~~~~~~~~~~~

Optimal parameters are determined by minimizing the chi-squared objective:

.. math::

   \chi^2(\boldsymbol{\theta}) = \sum_{i,j} \frac{[c_2^{\text{exp}}(\phi_i, t_j) - c_2^{\text{model}}(\phi_i, t_j; \boldsymbol{\theta})]^2}{\sigma_{ij}^2}

where:

* :math:`\boldsymbol{\theta}` = [D₀_ref, α_ref, D_offset_ref, D₀_sample, α_sample, D_offset_sample, v₀, β, v_offset, f₀, f₁, f₂, f₃, φ₀] is the 14-parameter vector
* :math:`c_2^{\text{exp}}` is experimental data
* :math:`c_2^{\text{model}}` is the theoretical prediction
* :math:`\sigma_{ij}` is measurement uncertainty

Classical Optimization Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package implements multiple optimization algorithms:

* **Nelder-Mead**: Derivative-free simplex method for robust convergence
* **L-BFGS-B**: Quasi-Newton method with box constraints for efficiency
* **Basin-hopping**: Global optimization to avoid local minima
* **Differential Evolution**: Evolutionary algorithm for complex landscapes

Robust Optimization
~~~~~~~~~~~~~~~~~~~

For noisy experimental data, robust methods provide stability:

**Distributionally Robust Optimization (DRO)**

.. math::

   \min_{\boldsymbol{\theta}} \max_{\mathbb{P} \in \mathcal{U}} \mathbb{E}_{\mathbb{P}}[\chi^2(\boldsymbol{\theta}, \boldsymbol{\xi})]

where :math:`\mathcal{U}` is a Wasserstein uncertainty set.

**Scenario-Based Robust Optimization**

.. math::

   \min_{\boldsymbol{\theta}} \max_{s \in S} \chi^2(\boldsymbol{\theta}, \boldsymbol{\xi}_s)

using bootstrap-generated scenarios :math:`S`.

Physical Constraints
~~~~~~~~~~~~~~~~~~~~

Optimization is subject to physical constraints:

* **Positivity**: :math:`D_0 > 0`, :math:`f_0 \geq 0`
* **Fraction bounds**: :math:`0 \leq f(t) \leq 1` for all :math:`t`
* **Angular range**: :math:`0° \leq \phi_0 < 360°`
* **Scaling bounds**: :math:`-2 \leq \alpha, \beta \leq 2` for physical time dependence

Numerical Implementation
------------------------

Computational Kernels
~~~~~~~~~~~~~~~~~~~~~

The package uses JIT-compiled Numba kernels for performance:

**1. Integral Computation**

.. code-block:: python

   @numba.jit(nopython=True, fastmath=True)
   def compute_transport_integral(t1, t2, D0, alpha, D_offset):
       """Compute ∫[t1 to t2] D(t) dt analytically."""
       return D0/(1+alpha) * (t2**(1+alpha) - t1**(1+alpha)) + D_offset*(t2-t1)

**2. Correlation Function**

.. code-block:: python

   @numba.jit(nopython=True, parallel=True)
   def compute_heterodyne_correlation(time_grid, phi_angles, params):
       """Vectorized heterodyne correlation computation."""
       # Parallel evaluation over angles and time points
       return c2_matrix

**3. Chi-Squared Objective**

.. code-block:: python

   @numba.jit(nopython=True)
   def chi_squared_objective(params, experimental_data, phi_angles, time_grid):
       """Fast chi-squared evaluation for optimization."""
       # Optimized residual calculation
       return chi_squared

Performance Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Vectorization**: SIMD operations for array computations
* **Memory layout**: Contiguous arrays for cache efficiency
* **Parallel execution**: Multi-threaded angle evaluations
* **Smart caching**: Precomputed matrices for repeated calculations

Error Analysis
--------------

Parameter Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~

Confidence intervals computed from the Hessian matrix:

.. math::

   \boldsymbol{\theta}_{\text{CI}} = \boldsymbol{\theta}_{\text{opt}} \pm t_{\alpha/2} \sqrt{\text{diag}(\mathbf{H}^{-1})}

where :math:`\mathbf{H}` is the Hessian at the optimum.

Goodness of Fit
~~~~~~~~~~~~~~~

Reduced chi-squared assesses fit quality:

.. math::

   \chi^2_{\text{red}} = \frac{\chi^2}{N - p}

where :math:`N` is the number of data points and :math:`p = 11` is the number of parameters.

Residual Analysis
~~~~~~~~~~~~~~~~~

Normalized residuals identify systematic deviations:

.. math::

   r_{ij} = \frac{c_2^{\text{exp}}(\phi_i, t_j) - c_2^{\text{model}}(\phi_i, t_j)}{\sigma_{ij}}

Well-distributed residuals (:math:`|r_{ij}| < 3`) indicate good model fit.

Validation Protocols
--------------------

Cross-Validation
~~~~~~~~~~~~~~~~

* **K-fold validation**: Assess parameter stability across data subsets
* **Leave-one-out**: Validate with small datasets

Bootstrap Analysis
~~~~~~~~~~~~~~~~~~

* **Non-parametric bootstrap**: Quantify parameter uncertainties
* **Parametric bootstrap**: Test model assumptions

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

* **Parameter perturbation**: Measure response to small changes
* **Robustness testing**: Evaluate stability against noise levels

References
----------

.. [1] He, H., Liang, H., Chu, M., et al. (2024). Transport coefficient approach for characterizing
   nonequilibrium dynamics in soft matter. *Proceedings of the National Academy of Sciences*,
   121(31), e2401162121. https://doi.org/10.1073/pnas.2401162121

See :doc:`publications` for additional references and applications.
