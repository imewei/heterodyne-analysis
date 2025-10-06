# Heterodyne Model: Documentation vs Implementation Analysis

**Date**: 2025-10-06
**Analysis**: Comparison of PNAS 2024 equations with actual code implementation

## Executive Summary

The code correctly implements **PNAS Equation S-95** (general time-dependent two-component heterodyne scattering), but the documentation incorrectly claims it implements **Equation S-98** (equilibrium form). Additionally, parameters are mislabeled as "diffusion coefficients" when they are actually "transport coefficients" in the PNAS framework.

## PNAS Paper Reference

He, H., Liang, H., Chu, M., et al. (2024). Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter. *PNAS*, 121(31), e2401162121. https://doi.org/10.1073/pnas.2401162121

## Detailed Comparison

### 1. Mathematical Formulation

#### PNAS Equation S-95 (General Time-Dependent Form)

```
c₂(q⃗, t₁, t₂) = 1 + β/f² [
  [xᵣ(t₁)xᵣ(t₂)]² exp(-q² ∫[t₁ to t₂] Jᵣ(t) dt) +
  [xₛ(t₁)xₛ(t₂)]² exp(-q² ∫[t₁ to t₂] Jₛ(t) dt) +
  2xᵣ(t₁)xᵣ(t₂)xₛ(t₁)xₛ(t₂) exp(-½q² ∫[t₁ to t₂] [Jₛ(t)+Jᵣ(t)] dt) cos(...)
]

f² = [xₛ(t₁)² + xᵣ(t₁)²][xₛ(t₂)² + xᵣ(t₂)²]
```

Where:
- **J_n(t)**: Transport coefficient for component n [Å²/s]
- **x_n(t)**: Time-dependent fraction of component n
- **β**: Contrast factor

#### PNAS Equation S-98 (Equilibrium Simplified Form)

```
g₂(q⃗,τ) = 1 + β[(1-x)²e^(-6q²Dᵣτ) + x²e^(-6q²Dₛτ) +
               2x(1-x)e^(-3q²(Dᵣ+Dₛ)τ)cos(q cos(φ)𝔼[v]τ)]
```

Where:
- **D_n**: Diffusion coefficient [Å²/s]
- **J_n = 6D_n**: For equilibrium Wiener process
- **x**: Composition fraction (time-independent)
- **τ = t₂ - t₁**: Delay time

#### Actual Code Implementation

**File**: `heterodyne/analysis/core.py`, method `calculate_heterodyne_correlation()`

```python
# Calculate diffusion integral (actually transport coefficient integral)
D_integral = self.create_time_integral_matrix_cached(f"D_{param_hash}", D_t)

# Compute g1 = exp(-q²/2 * dt * D_integral)
g1 = np.exp(-self.wavevector_q_squared_half_dt * D_integral)

# Both components use SAME g1
g1_ref = g1
g1_sample = g1

# Reference term: [f_r(t1) * f_r(t2) * g1]²
ref_term = (f1_ref * f2_ref * g1_ref) ** 2

# Sample term: [f_s(t1) * f_s(t2) * g1]²
sample_term = (f1_sample * f2_sample * g1_sample) ** 2

# Cross term: 2 * f_r * f_s * g1² * cos(velocity_term)
cross_term = 2 * f1_sample * f2_sample * f1_ref * f2_ref * cos_velocity_term * g1_sample * g1_ref

# Normalization
ftotal_squared = (f1_sample**2 + f1_ref**2) * (f2_sample**2 + f2_ref**2)

# Final correlation
g2_heterodyne = (ref_term + sample_term + cross_term) / ftotal_squared
```

### 2. Key Findings

#### Finding 1: Implements S-95, NOT S-98 ✓

**What the code does**:
- Uses time-dependent transport coefficient J(t) with integral ∫J(t)dt
- Exponential: exp(-q² ∫J(t)dt) matches S-95 form
- Time-dependent fractions f(t₁), f(t₂) match S-95

**What documentation claims**:
- Implements S-98 (equilibrium form)
- Uses diffusion coefficients D with J = 6D relationship

**Verdict**: Code correctly implements **S-95**, documentation is **incorrect**.

#### Finding 2: Parameter Mislabeling ⚠️

**Current labels** (in code/docs):
- `D0`: "Reference diffusion coefficient [Å²/s]"
- `alpha`: "Diffusion power-law exponent"
- `D_offset`: "Baseline diffusion offset [Å²/s]"

**Correct interpretation**:
- These are **transport coefficient** parameters J₀, α, J_offset
- NOT traditional diffusion coefficients (where J = 6D for Wiener process)
- In PNAS framework: J(t) parameterizes the transport coefficient directly

**Model equation**:
```
J(t) = J₀ · t^α + J_offset
```

**Physical interpretation**:
- For equilibrium Wiener process: J = 6D
- For nonequilibrium: J(t) is time-dependent transport coefficient
- The code implements the general transport coefficient approach

#### Finding 3: Single Transport Coefficient ⚠️

**PNAS S-95 allows**:
- Different J_r(t) for reference component
- Different J_s(t) for sample component
- Cross term: exp(-½q² ∫[J_r(t)+J_s(t)]dt)

**Code implementation**:
```python
g1_ref = g1   # Reference uses same g1
g1_sample = g1  # Sample uses same g1
```

This means: **J_r(t) = J_s(t) = J(t)** (single transport coefficient for both components)

**Impact**: This is a simplification of S-95, assuming both components have identical transport dynamics. Still valid, but less general than full S-95.

#### Finding 4: Contrast/Offset Fitting ✓

**PNAS form**: g₂ = 1 + β[...]

**Code implementation**:
```python
# Solves least squares: c2_fitted = offset + contrast * c2_theory
contrast_batch, offset_batch = self._solve_scaling_batch(theory_flat, exp_flat, ...)
```

**Verdict**: Correct! This properly implements the "1 + β[...]" form where:
- offset ≈ 1
- contrast ≈ β

### 3. Units and Dimensions

#### Transport Coefficient J(t)

For the exponential exp(-q² ∫J(t)dt) to be dimensionless:

```
[q²] × [∫J dt] = [1/Å²] × [Å² · s] = dimensionless ✓
```

Therefore: J has units **[Å²/s]** (same as diffusion coefficient)

#### Time-dependent parameterization

```
J(t) = J₀ · t^α + J_offset
```

For dimensional consistency:
- If α = 0: J₀ has units [Å²/s]
- If α ≠ 0: J₀ has units [Å²/s^(1-α)]
- J_offset always has units [Å²/s]

**Example**:
- α = -0.5 (aging): J₀ has units [Å²·s^0.5]
- α = 0.5 (accelerating): J₀ has units [Å²/s^0.5]

### 4. Velocity/Flow Term

**PNAS S-95 cross term**:
```
cos[q ∫[t₁ to t₂] 𝔼[v](t)cos(φ(t)) dt]
```

**Code implementation**:
```python
v_integral = self.create_time_integral_matrix_cached(f"v_{param_hash}", v_t)
angle_rad = np.deg2rad(phi0 - phi_angle)
cos_phi = np.cos(angle_rad)
velocity_argument = self.wavevector_q * v_integral * self.dt * cos_phi
cos_velocity_term = np.cos(velocity_argument)
```

**Verdict**: ✓ **Correct** - matches PNAS formulation exactly

Where:
```
v(t) = v₀ · t^β + v_offset
```

### 5. Fraction Dynamics

**PNAS notation**:
- x_s(t): Sample fraction (time-dependent)
- x_r(t) = 1 - x_s(t): Reference fraction

**Code implementation**:
```python
f_t = self.calculate_fraction_coefficient(fraction_params)
f1_sample, f2_sample = np.meshgrid(f_t, f_t)
f1_ref = 1 - f1_sample
f2_ref = 1 - f2_sample

# Normalization factor
ftotal_squared = (f1_sample**2 + f1_ref**2) * (f2_sample**2 + f2_ref**2)
```

**Verdict**: ✓ **Correct** - matches PNAS S-95 normalization exactly

**Fraction model**:
```
f(t) = f₀ · exp(f₁(t - f₂)) + f₃
```
with constraint: 0 ≤ f(t) ≤ 1

## Recommendations

### 1. Correct Documentation (HIGH PRIORITY)

**Current statement** (INCORRECT):
> "Implements the heterodyne scattering model from He et al. PNAS 2024 (Equation S-98)"

**Should be** (CORRECT):
> "Implements the two-component heterodyne scattering model from He et al. PNAS 2024 (Equation S-95), using time-dependent transport coefficients J(t). This is the general form that reduces to Equation S-98 under equilibrium conditions with J(t) = 6D."

### 2. Relabel Parameters (MEDIUM PRIORITY)

**Current naming**:
- `D0`, `alpha`, `D_offset` → labeled as "diffusion coefficient parameters"

**Options**:

**Option A: Minimal change (add clarification)**
```python
"""
Transport Coefficient Parameters (labeled as D for historical reasons):
- D0: Reference transport coefficient J₀ [Å²/s^(1-α)]
- alpha: Transport coefficient time-scaling exponent
- D_offset: Baseline transport coefficient J_offset [Å²/s]

Note: For equilibrium Wiener process, J = 6D where D is traditional diffusion.
For nonequilibrium conditions, J(t) is the generalized transport coefficient.
"""
```

**Option B: Full relabeling (breaking change)**
```python
# Rename parameters
J0, alpha, J_offset = parameters[0:3]  # Transport coefficient params
```

**Recommendation**: Use **Option A** (clarification) to avoid breaking existing configs.

### 3. Add Equation Hierarchy Documentation

Create a new section explaining the relationship:

```markdown
## Equation Hierarchy (PNAS 2024)

1. **Equation 14**: General N-component heterogeneous system
   - Most general form
   - Multiple components with different J_n(t), v_n(t), φ_n(t)

2. **Equation S-95**: Two-component specialization
   - Reference (r) and sample (s) components
   - Time-dependent: J_r(t), J_s(t), x_r(t), x_s(t)
   - **THIS PACKAGE IMPLEMENTS THIS FORM**

3. **Equation S-96**: Equilibrium Wiener process
   - J_n(t) = 6D_n (constant)
   - Time-independent fractions

4. **Equation S-98**: One-time correlation with composition
   - Simplified to single fraction x
   - τ = t₂ - t₁ (delay time)
   - "Commonly used heterodyne equation"

## Simplifications in This Implementation

Compared to full S-95, this package uses:
- **Single transport coefficient**: J_r(t) = J_s(t) = J(t)
- **Power-law time dependence**: J(t) = J₀·t^α + J_offset
- **Exponential fraction dynamics**: f(t) = f₀·exp(f₁(t-f₂)) + f₃
```

### 4. Update API Reference

**File**: `docs/api-reference/analysis-core.md`

Add clarification:
```markdown
### Parameter Interpretation

The 11-parameter model uses transport coefficient parameterization:

**Transport Coefficient (3 parameters)**:
- J(t) = J₀ · t^α + J_offset
- Labeled as "D" for historical compatibility
- For equilibrium: J = 6D (Wiener process)
- For nonequilibrium: J(t) is generalized transport coefficient

**Relationship to Diffusion**:
- Traditional diffusion coefficient D: [Å²/s]
- Transport coefficient J = 6D for Brownian motion
- This package uses J directly for nonequilibrium dynamics
```

### 5. Add Implementation Notes

Create new file: `docs/implementation-notes.md`

```markdown
# Implementation Notes

## Mathematical Foundation

This package implements PNAS 2024 Equation S-95, which is the general
time-dependent two-component heterodyne scattering equation using
transport coefficients J(t).

### Difference from Traditional Formulations

Many papers use diffusion coefficients D and write:
- exp(-6q²Dτ) for the decay

This package uses transport coefficients J and writes:
- exp(-q²∫J(t)dt) for the decay

The relationship is: **J = 6D** for equilibrium Wiener process.

For nonequilibrium dynamics, J(t) generalizes beyond this relationship.

### Why Transport Coefficients?

The transport coefficient approach (PNAS 2024) provides:
1. Unified framework for equilibrium and nonequilibrium
2. Direct parameterization of experimental observables
3. Power-law time dependence J(t) = J₀·t^α captures aging/rejuvenation
4. Avoids assumptions about underlying diffusion mechanism
```

## Conclusion

The code implementation is **mathematically correct** and implements PNAS Equation S-95 properly. However, documentation needs updating to:

1. ✅ Correctly state which equation is implemented (S-95, not S-98)
2. ⚠️ Clarify that parameters are transport coefficients, not traditional diffusion
3. 📝 Add equation hierarchy showing S-95 → S-96 → S-98 relationships
4. 📝 Explain simplification: single J(t) for both components

**Priority**:
- **HIGH**: Fix documentation claiming S-98 implementation
- **MEDIUM**: Add parameter interpretation notes
- **LOW**: Consider future enhancement to support J_r ≠ J_s

## References

1. He, H., Liang, H., Chu, M., et al. (2024). PNAS, 121(31), e2401162121
2. Code: `heterodyne/analysis/core.py`, line 1086-1199
3. Documentation: `docs/research/theoretical_framework.rst`
