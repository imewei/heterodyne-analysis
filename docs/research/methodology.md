# Research Methodology and Theoretical Framework

## Overview

This document provides comprehensive documentation of the theoretical framework,
computational methods, and validation procedures for the heterodyne-analysis package.
The methodology is based on the transport coefficient approach for characterizing
nonequilibrium dynamics in soft matter systems via X-ray Photon Correlation Spectroscopy
(XPCS).

**Authors**: Wei Chen, Hongrui He (Argonne National Laboratory) **Reference**: He et al.
(2024), PNAS 121(31):e2401162121

## Theoretical Framework

### Physical Model

The heterodyne-analysis package implements the **14-parameter two-component heterodyne
scattering model** from He et al. PNAS 2024 (Equation S-95) for analyzing time-dependent
intensity correlation functions in X-ray Photon Correlation Spectroscopy (XPCS)
measurements of soft matter systems under nonequilibrium conditions.

#### Two-Component Heterodyne Scattering

The heterodyne model describes scattering from a system with two distinct components:

- **Reference component**: Typically a tracer or reference material with known
  properties
- **Sample component**: The material of interest undergoing nonequilibrium dynamics

Each component has:

- Independent transport coefficients (Jᵣ and Jₛ)
- Time-dependent field correlation functions (g₁_ref and g₁_sample)
- Distinct fractions (xᵣ and xₛ) that may vary with time

#### Equation S-95: Heterodyne Correlation Function

The complete two-time correlation function for heterodyne scattering:

$$c_2(\\vec{q}, t_1, t_2, \\phi) = 1 + \\frac{\\beta}{f^2} \\left\[
\\left[x_r(t_1)x_r(t_2)\\right]^2 \\exp(-q^2\\int\_{t_1}^{t_2} J_r(t)dt) +
\\left[x_s(t_1)x_s(t_2)\\right]^2 \\exp(-q^2\\int\_{t_1}^{t_2} J_s(t)dt) +
2x_r(t_1)x_r(t_2)x_s(t_1)x_s(t_2)
\\exp\\left(-\\frac{1}{2}q^2\\int\_{t_1}^{t_2}[J_s(t)+J_r(t)]dt\\right) \\cos\\left\[q
\\cos(\\phi)\\int\_{t_1}^{t_2} v(t)dt\\right\] \\right\]$$

where the normalization factor is:

$$f^2 = \\left[x_s(t_1)^2 + x_r(t_1)^2\\right]\\left[x_s(t_2)^2 + x_r(t_2)^2\\right]$$

**Key Features of Equation S-95:**

1. **Two-time correlation structure**: c₂(t₁,t₂) is a full matrix, not just τ = t₂ - t₁

   - Each element represents correlation between times t₁ and t₂
   - Captures memory effects and non-stationary dynamics
   - Essential for nonequilibrium systems

2. **Three correlation terms**:

   - **Reference autocorrelation**: Pure reference component dynamics
   - **Sample autocorrelation**: Pure sample component dynamics
   - **Cross-correlation**: Interference between reference and sample, modulated by
     velocity

3. **Separate transport coefficients**: Jᵣ(t) and Jₛ(t) allow different diffusive
   behaviors

   - Reference may exhibit normal diffusion while sample shows anomalous transport
   - Captures distinct physical processes in each component

4. **Time-dependent fractions**: xₛ(t) and xᵣ(t) = 1 - xₛ(t)

   - Compositional changes during measurement
   - Mixing, phase separation, or aggregation dynamics

5. **Velocity cross-term**: cos[q cos(φ)∫v(t)dt]

   - Flow-induced decorrelation between components
   - Angular dependence through φ (flow direction relative to scattering vector)
   - Captures shear-induced effects

**Physical Interpretation:**

The heterodyne correlation arises from coherent scattering between reference and sample
fields. Unlike homodyne scattering (intensity-intensity correlation), heterodyne
scattering provides direct access to field correlations through the cross-term, enabling
measurement of both amplitude and phase information.

For equilibrium systems with no flow (v=0) and static fractions, Equation S-95 reduces
to simpler forms, but retains separate transport dynamics for each component.

### 14-Parameter Heterodyne Model

The complete heterodyne model uses 14 parameters organized into 5 groups:

#### 1. Reference Transport (3 parameters)

**Power-law transport coefficient:** $$J_r(t) = J\_{0,\\text{ref}} \\cdot
t^{\\alpha\_{\\text{ref}}} + J\_{\\text{offset},\\text{ref}}$$

| Parameter | Symbol | Units | Physical Meaning |
|-----------|--------|-------|------------------| | D₀_ref | J₀_ref | Å²/s | Reference
transport coefficient at t=1s | | α_ref | α_ref | - | Reference transport exponent
(anomalous diffusion) | | D_offset_ref | J_offset_ref | Å²/s | Reference baseline
transport |

**Physical interpretation:**

- α_ref = 0: Normal diffusion (Brownian motion)
- α_ref > 0: Superdiffusion (active transport, convection)
- α_ref < 0: Subdiffusion (caging, crowding)

**Relationship to diffusion:** For equilibrium systems, J = 6D where D is the diffusion
coefficient. Parameters labeled "D" in the code represent transport coefficients J.

#### 2. Sample Transport (3 parameters)

**Power-law transport coefficient:** $$J_s(t) = J\_{0,\\text{sample}} \\cdot
t^{\\alpha\_{\\text{sample}}} + J\_{\\text{offset},\\text{sample}}$$

| Parameter | Symbol | Units | Physical Meaning |
|-----------|--------|-------|------------------| | D₀_sample | J₀_sample | Å²/s |
Sample transport coefficient at t=1s | | α_sample | α_sample | - | Sample transport
exponent (anomalous diffusion) | | D_offset_sample | J_offset_sample | Å²/s | Sample
baseline transport |

**Key insight:** Independent transport allows reference and sample to exhibit different
dynamics:

- Reference: Normal diffusion (α_ref = 0)
- Sample: Subdiffusion (α_sample < 0) due to crowding
- Captures heterogeneous dynamics in complex fluids

#### 3. Velocity (3 parameters)

**Time-dependent velocity coefficient:** $$v(t) = v_0 \\cdot t^\\beta +
v\_{\\text{offset}}$$

| Parameter | Symbol | Units | Physical Meaning |
|-----------|--------|-------|------------------| | v₀ | v₀ | nm/s | Velocity
coefficient at t=1s | | β | β | - | Velocity time-dependence exponent | | v_offset |
v_offset | nm/s | Baseline velocity |

**Physical significance:**

- β = 0: Constant flow velocity
- β > 0: Accelerating flow
- β < 0: Decelerating flow

**Role in cross-correlation:** Velocity appears only in the cross-term as cos\[q
cos(φ)∫v(t)dt\], modulating interference between reference and sample components.

#### 4. Fraction (4 parameters)

**Time-dependent sample fraction:** $$x_s(t) = f_0 \\cdot \\exp\\left\[f_1 \\cdot (t -
f_2)\\right\] + f_3$$

with constraint: 0 ≤ xₛ(t) ≤ 1, and xᵣ(t) = 1 - xₛ(t)

| Parameter | Symbol | Units | Physical Meaning |
|-----------|--------|-------|------------------| | f₀ | f₀ | - | Fraction amplitude
(exponential term) | | f₁ | f₁ | s⁻¹ | Fraction rate (exponential growth/decay) | | f₂ |
f₂ | s | Fraction time offset (delay/shift) | | f₃ | f₃ | - | Fraction baseline
(steady-state value) |

**Physical interpretation:**

- f₁ > 0: Sample fraction increases (aggregation, phase separation)
- f₁ < 0: Sample fraction decreases (dissolution, dispersal)
- f₁ = 0: Static fractions (no compositional change)
- f₃: Asymptotic fraction as t → ∞

**Two-time structure:** Critical for Equation S-95:

- Fractions evaluated at both t₁ and t₂ independently
- Normalization f² depends on fractions at both times
- Captures non-stationary compositional dynamics

#### 5. Flow Angle (1 parameter)

**Flow direction relative to scattering:** $$\\phi = \\phi_0 -
\\phi\_{\\text{scattering}}$$

| Parameter | Symbol | Units | Physical Meaning |
|-----------|--------|-------|------------------| | φ₀ | φ₀ | degrees | Flow direction
angle |

**Physical significance:**

- Appears in cross-term: cos[q cos(φ)∫v(t)dt]
- φ = 0°: Flow parallel to scattering vector (maximum effect)
- φ = 90°: Flow perpendicular to scattering vector (no effect)
- Angular dependence reveals flow geometry

### Implementation of Field Correlations

**Field correlation functions for each component:**

$$g\_{1,r}(t_1, t_2) = \\exp\\left(-\\frac{q^2}{2} \\int\_{t_1}^{t_2} J_r(t)
dt\\right)$$

$$g\_{1,s}(t_1, t_2) = \\exp\\left(-\\frac{q^2}{2} \\int\_{t_1}^{t_2} J_s(t)
dt\\right)$$

**Key relationships:**

- Reference autocorrelation: g₁_r² = exp(-q²∫Jᵣdt)
- Sample autocorrelation: g₁_s² = exp(-q²∫Jₛdt)
- Cross-correlation: g₁_r·g₁_s = exp(-½q²∫[Jₛ+Jᵣ]dt)

**Implementation note:** The code computes g₁ functions (with factor of 1/2 in exponent)
and then squares them for autocorrelation terms, naturally producing the correct factors
in Equation S-95.

### Physical Constraints

**Parameter bounds ensure physical validity:**

1. **Transport positivity**: J(t) > 0 for all t

   - Implemented via optimization bounds
   - Ensures positive diffusion coefficients

2. **Fraction constraints**: 0 ≤ xₛ(t) ≤ 1

   - Enforced by np.clip() in implementation
   - Physical requirement for mixing fractions

3. **Numerical stability**:

   - Exponential arguments clipped to prevent overflow
   - Time zero handling for negative exponents (β < 0, α < 0)
   - Threshold values for numerical stability

### Comparison with Legacy Models

**14-parameter heterodyne vs. simpler models:**

| Feature | 14-param Heterodyne | 7-param Laminar Flow | 3-param Static |
|---------|---------------------|----------------------|----------------| | Components |
2 (ref + sample) | 1 | 1 | | Transport | Independent Jᵣ, Jₛ | Single J | Single J | |
Fractions | Time-dependent xₛ(t) | Static | N/A | | Cross-correlation | Yes (with
velocity) | N/A | N/A | | Velocity | Time-dependent v(t) | Time-dependent | Zero | |
Parameters | 14 | 7 | 3 |

**When to use 14-parameter model:**

- Two-component systems (tracer + sample)
- Nonequilibrium dynamics
- Time-varying composition
- Flow-induced decorrelation
- Heterogeneous transport dynamics

### Analysis Modes

The package supports three analysis modes with different parameter sets and
computational complexities:

#### 1. Static Isotropic Mode

**Parameters**: 3 (D₀, α, D_offset) **Complexity**: O(N) **Physical System**: Brownian
motion only, isotropic systems

**Model Equations:**

- Flow parameters set to zero: $\\dot{\\gamma}_0 = \\beta =
  \\dot{\\gamma}_{\\text{offset}} = 0$
- Angular dependence removed: $\\phi_0$ irrelevant
- Correlation function simplifies to:

$$c_2(q, t_1, t_2) = 1 + \\beta \\exp\\left[-q^2 \\int\_{t_1}^{t_2} D(t') dt'\\right]$$

**Use Cases:**

- Equilibrium colloidal suspensions
- Protein solutions without flow
- Brownian motion characterization

**Optimization Strategy:**

- Identical correlation functions for all angles (static, isotropic)
- Vectorized calculation for all angles simultaneously
- Fastest mode: O(N) complexity

#### 2. Static Anisotropic Mode

**Parameters**: 3 (D₀, α, D_offset) **Complexity**: O(N log N) **Physical System**:
Static systems with angular dependence

**Model Equations:**

- Flow parameters zero like isotropic mode
- Angular dependence preserved in experimental data
- Different correlation functions for each angle

**Use Cases:**

- Anisotropic particles (rods, ellipsoids)
- Systems with preferred orientations
- Crystalline or liquid crystalline phases

**Optimization Strategy:**

- Angle-dependent data fitting
- Preserves angular information
- Conditional subsampling: Use all angles when n_angles < 4

#### 3. Laminar Flow Mode

**Parameters**: 7 (D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀) **Complexity**: O(N²)
**Physical System**: Full nonequilibrium with flow and shear

**Model Equations:**

- Complete correlation function with both diffusion and flow terms
- Angular dependence through flow direction φ₀
- Shear-induced decorrelation through sinc² term

**Use Cases:**

- Colloidal suspensions under shear
- Microfluidic flow systems
- Rheological measurements with XPCS
- Active matter in flow

**Optimization Strategy:**

- Full parameter space optimization
- Angle-dependent correlation functions
- Most computationally intensive mode

### Parameter Space and Physical Constraints

#### Static Parameters (All Modes)

| Parameter | Symbol | Units | Typical Range | Physical Meaning |
|-----------|--------|-------|---------------|------------------| | Reference diffusion
| D₀ | Å²/s | 10⁻¹⁵ to 10⁻¹⁰ | Diffusion coefficient at t=1s | | Diffusion exponent | α
| - | -1.0 to 2.0 | Anomalous diffusion exponent | | Diffusion offset | D_offset | Å²/s
| -10⁻¹² to 10⁻¹² | Baseline diffusion coefficient |

#### Flow Parameters (Laminar Flow Mode Only)

| Parameter | Symbol | Units | Typical Range | Physical Meaning |
|-----------|--------|-------|---------------|------------------| | Reference shear rate
| γ̇₀ | s⁻¹ | 0.1 to 1000 | Shear rate at t=1s | | Shear exponent | β | - | -1.0 to 2.0
| Shear rate time-dependence | | Shear offset | γ̇_offset | s⁻¹ | -100 to 100 | Baseline
shear rate | | Flow direction | φ₀ | degrees | 0 to 360 | Angle of flow direction |

#### Physical Constraints

1. **Diffusion Positivity**: D(t) > 0 for all t

   - Constraint: $D_0 t^\\alpha + D\_{\\text{offset}} > 0$
   - Implementation: Bounds checking during optimization

2. **Parameter Bounds**: Configured in JSON files

   - Example: `"D0": [1e-15, 1e-10]`
   - Enforced by optimization algorithms

3. **Numerical Stability**: Avoid extreme parameter values

   - Logarithmic parameter space for wide-range parameters
   - Bounded optimization with trust regions

## Computational Methods

### Frame Counting Convention

#### Overview

The package uses **1-based inclusive** frame counting in configuration files, which is
then converted to 0-based Python array indices for processing.

#### Mathematical Formula

```python
time_length = end_frame - start_frame + 1  # Inclusive counting
```

**Examples:**

| start_frame | end_frame | time_length | Explanation |
|-------------|-----------|-------------|-------------| | 1 | 100 | 100 | Full range
from first to 100th frame | | 401 | 1000 | 600 | 600 frames from 401 to 1000 (NOT 599!)
| | 1 | 1 | 1 | Single frame edge case |

#### Convention Details

**Configuration Convention (1-based, inclusive):**

- `start_frame=1`: Start at first frame (not zero)
- `end_frame=100`: Include frame 100 in analysis
- Both boundaries inclusive: [start_frame, end_frame]

**Python Slice Convention (0-based, exclusive end):**

- Internal conversion: `python_start = start_frame - 1`
- `python_end = end_frame` (unchanged for exclusive slice)
- Array slice `[python_start:python_end]` gives exactly `time_length` elements

**Example Conversion:**

```python
# Configuration values
start_frame = 401  # 1-based
end_frame = 1000   # 1-based

# Calculate time_length
time_length = 1000 - 401 + 1  # = 600

# Convert to Python indices
python_start = 401 - 1  # = 400 (0-based)
python_end = 1000       # = 1000 (exclusive)

# Array slicing
data_slice = full_data[:, 400:1000, 400:1000]  # 600 frames
assert data_slice.shape[1] == 600  # Correct!
```

#### Time Array Construction

The time array always starts at 0 for the t₁=t₂=0 correlation:

```python
dt = 0.5  # Time step [s/frame]
time_length = 600  # Number of frames

# Time array: [0, dt, 2*dt, ..., dt*(time_length-1)]
time_array = np.linspace(0, dt * (time_length - 1), time_length)

# Example for time_length=600, dt=0.5:
# time_array = [0.0, 0.5, 1.0, ..., 299.5] seconds
# 600 points total
```

#### Cached Data Compatibility

**Cache Filename Convention:**

```
cached_c2_isotropic_frames_{start_frame}_{end_frame}.npz
```

Example: `cached_c2_isotropic_frames_401_1000.npz` contains 600 frames.

**Cache Dimension Validation:**

The analysis core automatically detects and adjusts for dimension mismatches:

```python
# In HeterodyneAnalysisCore.load_experimental_data()
if c2_experimental.shape[1] != self.time_length:
    logger.info(f"Auto-adjusting time_length to match cached data")
    self.time_length = c2_experimental.shape[1]
    # Recreate time_array with correct dimensions
    self.time_array = np.linspace(0, self.dt * (self.time_length - 1), self.time_length)
```

#### Utility Functions

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

### Conditional Angle Subsampling

#### Strategy

The package implements conditional angle subsampling to preserve angular information
when the number of available angles is small:

```python
# In ClassicalOptimizer and RobustHeterodyneOptimizer
if n_angles < 4:
    # Preserve all angles for angular information
    angle_subsample_size = n_angles
else:
    # Subsample for performance (default: 4 angles)
    angle_subsample_size = config.get("n_angles", 4)
```

#### Impact

**Before Conditional Subsampling (v < 0.6.5):**

- Input: 2 angles available
- Subsampling: 2 → 1 angle
- Result: 50% angular information loss

**After Conditional Subsampling (v ≥ 0.6.5):**

- Input: 2 angles available
- Conditional: n_angles < 4 → Use all angles
- Result: 100% angular information preservation

**Performance Trade-off:**

- Time subsampling still applied: ~16x performance improvement
- Combined impact: 20-50x speedup with \<10% χ² degradation

#### Configuration

All configuration templates include subsampling documentation:

```json
{
  "subsampling": {
    "n_angles": 4,
    "n_time_points": 16,
    "strategy": "conditional",
    "preserve_angular_info": true,
    "comment": "Conditional: n_angles preserved when < 4 for angular info retention"
  }
}
```

### Chi-Squared Optimization

#### Objective Function

The optimization objective is the chi-squared goodness-of-fit:

$$\\chi^2(\\mathbf{p}) = \\sum\_{i=1}^{n\_{\\text{angles}}}
\\sum\_{j=1}^{n\_{\\text{time}}} \\sum\_{k=1}^{n\_{\\text{time}}}
\\left\[\\frac{c\_{2,\\text{exp}}^{(i)}(j,k) - (A_i \\cdot
c\_{2,\\text{theo}}^{(i)}(j,k; \\mathbf{p}) +
B_i)}{c\_{2,\\text{exp}}^{(i)}(j,k)}\\right\]^2$$

where:

- $\\mathbf{p}$ is the parameter vector
- $i$ indexes scattering angles
- $j, k$ index time points
- $A_i, B_i$ are contrast and offset scaling parameters for each angle
- $c\_{2,\\text{exp}}^{(i)}$ is experimental correlation for angle $i$
- $c\_{2,\\text{theo}}^{(i)}$ is theoretical correlation for angle $i$

#### Scaling Parameter Optimization

For each angle, optimal scaling parameters $(A_i, B_i)$ are determined by least-squares:

$$\\min\_{A_i, B_i} \\sum\_{j,k} \\left\[\\frac{c\_{2,\\text{exp}}^{(i)}(j,k) - (A_i
\\cdot c\_{2,\\text{theo}}^{(i)}(j,k) +
B_i)}{c\_{2,\\text{exp}}^{(i)}(j,k)}\\right\]^2$$

This is a linear least squares problem solved independently for each angle.

**Solution (v0.6.1+ optimization):**

```python
# For each angle i
# Build system: [c_theo, ones] @ [A, B]^T = c_exp
# Solve using np.linalg.solve() instead of lstsq() for 2x speedup
```

#### Performance Optimizations

**Version 0.6.1+ Improvements:**

1. **Chi-Squared Calculation**: 38% faster (1.33ms → 0.82ms)

   - Vectorized reshaping instead of list comprehensions
   - Pre-allocated result arrays
   - Cached configuration to avoid dict lookups

2. **Least Squares**: 2x faster for 2x2 systems

   - Direct solve() instead of lstsq() for small matrices
   - Explicit matrix inversion avoided

3. **Memory Efficiency**: Reduced allocation overhead

   - Memory pooling for result arrays
   - In-place operations where possible

### Numerical Integration

#### Diffusion Integral

The diffusion contribution requires integration of the time-dependent diffusion
coefficient:

$$I_D(t_1, t_2) = \\int\_{t_1}^{t_2} D(t') dt' = \\int\_{t_1}^{t_2} (D_0 t'^\\alpha +
D\_{\\text{offset}}) dt'$$

**Analytical Solution:**

$$I_D(t_1, t_2) = \\begin{cases} D_0 \\frac{t_2^{\\alpha+1} -
t_1^{\\alpha+1}}{\\alpha+1} + D\_{\\text{offset}}(t_2 - t_1) & \\text{if } \\alpha \\neq
-1 \\ D_0 \\ln\\left(\\frac{t_2}{t_1}\\right) + D\_{\\text{offset}}(t_2 - t_1) &
\\text{if } \\alpha = -1 \\end{cases}$$

**Implementation:**

```python
# Vectorized for all time pairs (t_i, t_j)
time_diff = time_array[:, np.newaxis] - time_array[np.newaxis, :]

if alpha != -1:
    integral = D0 / (alpha + 1) * (time_array[:, np.newaxis]**(alpha + 1) -
                                    time_array[np.newaxis, :]**(alpha + 1))
else:
    integral = D0 * np.log(time_array[:, np.newaxis] / time_array[np.newaxis, :])

integral += D_offset * time_diff
```

#### Shear Rate Integral

Similarly, the flow contribution requires:

$$I\_\\gamma(t_1, t_2) = \\int\_{t_1}^{t_2} \\dot{\\gamma}(t') dt' = \\int\_{t_1}^{t_2}
(\\dot{\\gamma}_0 t'^\\beta + \\dot{\\gamma}_{\\text{offset}}) dt'$$

**Analytical Solution:**

$$I\_\\gamma(t_1, t_2) = \\begin{cases} \\dot{\\gamma}_0 \\frac{t_2^{\\beta+1} -
t_1^{\\beta+1}}{\\beta+1} + \\dot{\\gamma}_{\\text{offset}}(t_2 - t_1) & \\text{if }
\\beta \\neq -1 \\ \\dot{\\gamma}_0 \\ln\\left(\\frac{t_2}{t_1}\\right) +
\\dot{\\gamma}_{\\text{offset}}(t_2 - t_1) & \\text{if } \\beta = -1 \\end{cases}$$

### Optimization Algorithms

#### Classical Methods

**1. Nelder-Mead Simplex**

- **Type**: Derivative-free optimization
- **Advantages**: Robust to noisy functions, no gradient required
- **Disadvantages**: Slower convergence than gradient-based methods
- **Use Case**: Default method for general XPCS analysis

**Implementation:**

```python
from scipy.optimize import minimize

result = minimize(
    fun=objective_function,
    x0=initial_params,
    method='Nelder-Mead',
    bounds=parameter_bounds,
    options={
        'maxiter': 10000,
        'xatol': 1e-8,
        'fatol': 1e-8
    }
)
```

**2. Gurobi Quadratic Programming**

- **Type**: Quadratic approximation with commercial solver
- **Advantages**: Fast convergence for smooth problems, handles constraints
- **Disadvantages**: Requires Gurobi license, may fail for non-quadratic objectives
- **Use Case**: High-performance optimization with Gurobi license

**Implementation:**

```python
# Approximate objective as quadratic using finite differences
# Build quadratic program: minimize (1/2)x^T Q x + c^T x
# Subject to: lb <= x <= ub

model = gp.Model("heterodyne_optimization")
x = model.addVars(n_params, lb=lower_bounds, ub=upper_bounds)
model.setObjective(quadratic_objective, GRB.MINIMIZE)
model.optimize()
```

#### Robust Optimization Methods

**1. Distributionally Robust Optimization (DRO)**

Wasserstein uncertainty sets for data distribution robustness:

$$\\min\_{\\mathbf{p}} \\max\_{\\mathbb{P} \\in \\mathcal{U}_\\epsilon}
\\mathbb{E}_{\\mathbb{P}}[\\chi^2(\\mathbf{p}; c_2)]$$

where $\\mathcal{U}\_\\epsilon$ is a Wasserstein ball of radius $\\epsilon$ around the
empirical distribution.

**Use Case**: Noisy data with distribution uncertainty

**2. Scenario-Based Optimization**

Bootstrap resampling for statistical robustness:

$$\\min\_{\\mathbf{p}} \\frac{1}{N\_{\\text{scenarios}}}
\\sum\_{s=1}^{N\_{\\text{scenarios}}} \\chi^2(\\mathbf{p}; c_2^{(s)})$$

where $c_2^{(s)}$ are bootstrap-resampled datasets.

**Use Case**: Data with outliers and measurement uncertainty

**3. Ellipsoidal Uncertainty Sets**

Bounded uncertainty with confidence ellipsoids:

$$\\min\_{\\mathbf{p}} \\max\_{|\\delta| \\leq \\rho} \\chi^2(\\mathbf{p}; c_2 +
\\delta)$$

where $\\rho$ is the uncertainty radius.

**Use Case**: Large datasets with bounded uncertainty (up to 8M+ data points)

**Memory Optimization:**

```python
@monitor_memory(max_usage_percent=90.0)  # 90% limit for large datasets
def optimize_ellipsoidal_robust(...):
    # Optimization with memory monitoring
    pass
```

## Validation and Quality Assurance

### Regression Testing

#### Frame Counting Validation

Comprehensive regression tests ensure correct frame counting:

```python
# Test case from test_time_length_calculation.py
start_frame = 401
end_frame = 1000
expected_time_length = 600  # NOT 599!

time_length = end_frame - start_frame + 1
assert time_length == expected_time_length
```

**Test Suite:**

- `test_time_length_calculation.py`: Frame counting formula validation
- Coverage: All 9 modules using frame counting
- Run: `pytest heterodyne/tests/test_time_length_calculation.py -v`

#### Performance Regression

Performance baselines track optimization improvements:

```bash
# Run performance regression tests
pytest heterodyne/tests/ -v -m performance

# Update baselines after verified improvements
python heterodyne/tests/establish_baselines.py
```

**Tracked Metrics:**

- Chi-squared calculation time
- Memory usage
- Optimization convergence speed
- JIT compilation overhead

### Statistical Validation

#### Cross-Validation

K-fold cross-validation for parameter reliability:

```python
# Pseudo-code for k-fold validation
for fold in range(k_folds):
    train_data, test_data = split_data(c2_experimental, fold, k_folds)
    params = optimize(train_data)
    chi2_test = calculate_chi_squared(params, test_data)
    validation_scores.append(chi2_test)

# Compute mean and std of validation scores
mean_score = np.mean(validation_scores)
std_score = np.std(validation_scores)
```

#### Bootstrap Analysis

Statistical uncertainty quantification:

```python
# Bootstrap parameter uncertainty
bootstrap_params = []
for i in range(n_bootstrap):
    # Resample data with replacement
    c2_bootstrap = resample_data(c2_experimental)
    params = optimize(c2_bootstrap)
    bootstrap_params.append(params)

# Compute confidence intervals
param_means = np.mean(bootstrap_params, axis=0)
param_stds = np.std(bootstrap_params, axis=0)
confidence_intervals = compute_percentile_intervals(bootstrap_params, alpha=0.05)
```

#### Residual Analysis

Goodness-of-fit assessment:

```python
# Calculate residuals
residuals = c2_experimental - c2_theoretical_scaled

# Reduced chi-squared
dof = total_data_points - n_params
chi2_reduced = chi2 / dof

# Residual statistics
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)

# Quality criteria
quality_good = (
    chi2_reduced < 2.0 and
    abs(residual_mean) < 0.01 and
    residual_std < 0.1
)
```

### Experimental Validation

#### Synchrotron Facilities

- **Facility**: Advanced Photon Source (APS) Sector 8-ID-I
- **Beamline**: 8-ID-I (XPCS beamline)
- **Energy**: 7-12 keV
- **Detector**: Pixel array detectors with μs time resolution

#### Sample Systems

1. **Colloidal Suspensions**

   - Silica particles (20-500 nm diameter)
   - Concentration: 0.1-10% volume fraction
   - Solvents: Water, glycerol-water mixtures

2. **Protein Solutions**

   - Model proteins (lysozyme, BSA)
   - Concentration: 1-100 mg/mL
   - Buffers: Phosphate, Tris-HCl

3. **Active Matter**

   - Self-propelled colloids
   - Biological systems (bacteria)
   - Light-activated swimmers

#### Flow Conditions

1. **Laminar Shear**

   - Couette geometry (rotating cylinder)
   - Shear rates: 0.1-1000 s⁻¹
   - Gap sizes: 10-100 μm

2. **Pressure-Driven Flow**

   - Microfluidic channels
   - Flow rates: 0.01-10 μL/min
   - Channel dimensions: 10-500 μm

3. **Microfluidic Geometries**

   - Straight channels
   - Cross-junction mixers
   - Expansion/contraction regions

## References

### Primary Publications

1. **He, H., et al. (2024)**. "Transport coefficient approach for characterizing
   nonequilibrium dynamics in soft matter." *Proceedings of the National Academy of
   Sciences*, 121(31), e2401162121. https://doi.org/10.1073/pnas.2401162121

   *Primary reference for the theoretical framework and methodology.*

### X-ray Photon Correlation Spectroscopy

2. **Grübel, G., & Zontone, F. (2004)**. "Correlation spectroscopy with coherent
   X-rays." *Journal of Alloys and Compounds*, 362(1-2), 3-11.

3. **Sandy, A. R., et al. (2018)**. "Hard X-ray photon correlation spectroscopy methods
   for materials studies." *Annual Review of Materials Research*, 48, 167-195.

### Nonequilibrium Soft Matter

4. **Bandyopadhyay, R., et al. (2004)**. "Evolution of particle-scale dynamics in an
   aging clay suspension." *Physical Review Letters*, 93(22), 228302.

5. **Rogers, S. A., et al. (2011)**. "A sequence of physical processes determined and
   quantified in LAOS: Application to a yield stress fluid." *Journal of Rheology*,
   55(2), 435-458.

### Robust Optimization

6. **Bertsimas, D., et al. (2011)**. "Theory and applications of robust optimization."
   *SIAM Review*, 53(3), 464-501.

7. **Esfahani, P. M., & Kuhn, D. (2018)**. "Data-driven distributionally robust
   optimization using the Wasserstein metric: Performance guarantees and tractable
   reformulations." *Mathematical Programming*, 171(1-2), 115-166.

## Appendix: Mathematical Derivations

### Diffusion Integral Derivation

For the power-law diffusion coefficient $D(t) = D_0 t^\\alpha + D\_{\\text{offset}}$:

$$I_D(t_1, t_2) = \\int\_{t_1}^{t_2} (D_0 t'^\\alpha + D\_{\\text{offset}}) dt'$$

**Case 1: $\\alpha \\neq -1$**

$$I_D = D_0 \\int\_{t_1}^{t_2} t'^\\alpha dt' + D\_{\\text{offset}} \\int\_{t_1}^{t_2}
dt'$$

$$= D_0 \\left[\\frac{t'^{\\alpha+1}}{\\alpha+1}\\right]_{t_1}^{t_2} +
D_{\\text{offset}}(t_2 - t_1)$$

$$= D_0 \\frac{t_2^{\\alpha+1} - t_1^{\\alpha+1}}{\\alpha+1} + D\_{\\text{offset}}(t_2 -
t_1)$$

**Case 2: $\\alpha = -1$**

$$I_D = D_0 \\int\_{t_1}^{t_2} \\frac{1}{t'} dt' + D\_{\\text{offset}}(t_2 - t_1)$$

$$= D_0 [\\ln t']_{t_1}^{t_2} + D_{\\text{offset}}(t_2 - t_1)$$

$$= D_0 \\ln\\left(\\frac{t_2}{t_1}\\right) + D\_{\\text{offset}}(t_2 - t_1)$$

### Scaling Parameter Solution

For each angle $i$, minimize:

$$\\min\_{A_i, B_i} S = \\sum\_{j,k} w\_{jk} \\left\[c\_{\\text{exp}}(j,k) - (A_i
c\_{\\text{theo}}(j,k) + B_i)\\right\]^2$$

where $w\_{jk} = 1/c\_{\\text{exp}}^2(j,k)$ are weights.

**Normal Equations:**

$$\\frac{\\partial S}{\\partial A_i} = -2 \\sum\_{j,k} w\_{jk} c\_{\\text{theo}}(j,k)
\\left[c\_{\\text{exp}}(j,k) - (A_i c\_{\\text{theo}}(j,k) + B_i)\\right] = 0$$

$$\\frac{\\partial S}{\\partial B_i} = -2 \\sum\_{j,k} w\_{jk}
\\left[c\_{\\text{exp}}(j,k) - (A_i c\_{\\text{theo}}(j,k) + B_i)\\right] = 0$$

**Matrix Form:**

$$\\begin{bmatrix} \\sum w c\_{\\text{theo}}^2 & \\sum w c\_{\\text{theo}} \\ \\sum w
c\_{\\text{theo}} & \\sum w \\end{bmatrix} \\begin{bmatrix} A_i \\ B_i \\end{bmatrix} =
\\begin{bmatrix} \\sum w c\_{\\text{theo}} c\_{\\text{exp}} \\ \\sum w c\_{\\text{exp}}
\\end{bmatrix}$$

This 2×2 linear system is solved using `np.linalg.solve()` for efficiency.

______________________________________________________________________

**Document**: Research Methodology and Theoretical Framework **Last Updated**:
2025-10-01 **Version**: 1.0.0 **Authors**: Wei Chen, Hongrui He (Argonne National
Laboratory)
