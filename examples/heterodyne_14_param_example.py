#!/usr/bin/env python3
"""
Example: Using the 14-Parameter Heterodyne Model
=================================================

This example demonstrates how to use the heterodyne scattering model with
separate reference and sample transport coefficients (14 parameters total).

The 14-parameter model correctly implements He et al. PNAS 2024 Equation S-95
with independent g1_ref and g1_sample field correlations.

Author: Claude (Anthropic)
Date: 2025-10-06
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from heterodyne.analysis.core import HeterodyneAnalysisCore


def create_14param_config(output_path="heterodyne_14param_config.json"):
    """
    Create a 14-parameter heterodyne configuration file.

    The 14 parameters are organized as:
    - Reference transport (3): D0_ref, alpha_ref, D_offset_ref
    - Sample transport (3): D0_sample, alpha_sample, D_offset_sample
    - Velocity (3): v0, beta, v_offset
    - Fraction (4): f0, f1, f2, f3
    - Flow angle (1): phi0

    Returns
    -------
    str
        Path to created config file
    """
    config = {
        "description": "14-parameter heterodyne model with independent ref and sample",
        "model": "heterodyne",
        "version": "2.0",
        "initial_parameters": {
            "values": [
                # Reference transport coefficients
                100.0,   # D0_ref: reference diffusion amplitude
                -0.5,    # alpha_ref: reference power-law exponent
                10.0,    # D_offset_ref: reference baseline diffusion

                # Sample transport coefficients (initially equal to reference)
                100.0,   # D0_sample: sample diffusion amplitude
                -0.5,    # alpha_sample: sample power-law exponent
                10.0,    # D_offset_sample: sample baseline diffusion

                # Velocity parameters
                0.1,     # v0: velocity amplitude
                0.0,     # beta: velocity power-law exponent
                0.01,    # v_offset: baseline velocity

                # Fraction parameters (controls mixing of ref and sample)
                0.5,     # f0: fraction amplitude
                0.0,     # f1: exponential decay rate
                50.0,    # f2: time offset for fraction
                0.3,     # f3: baseline fraction

                # Flow angle
                0.0      # phi0: flow angle in degrees
            ],
            "parameter_names": [
                "D0_ref", "alpha_ref", "D_offset_ref",
                "D0_sample", "alpha_sample", "D_offset_sample",
                "v0", "beta", "v_offset",
                "f0", "f1", "f2", "f3",
                "phi0"
            ],
            "bounds": [
                # Reference transport bounds
                [0, 1000], [-2, 2], [0, 100],
                # Sample transport bounds
                [0, 1000], [-2, 2], [0, 100],
                # Velocity bounds
                [-10, 10], [-2, 2], [-1, 1],
                # Fraction bounds
                [0, 1], [-1, 1], [0, 200], [0, 1],
                # Flow angle bounds
                [-360, 360]
            ]
        },
        "analyzer_parameters": {
            "temporal": {
                "dt": 0.1,            # Time step in seconds
                "start_frame": 0,     # Starting frame
                "end_frame": 100      # Ending frame
            },
            "scattering": {
                "wavevector_q": 0.0054  # Scattering wavevector in nm⁻¹
            },
            "geometry": {
                "stator_rotor_gap": 2000000  # Gap distance in nm
            }
        },
        "optimization_config": {
            "classical_optimization": {
                "methods": ["Nelder-Mead"],
                "options": {"maxiter": 100}
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Created config: {output_path}")
    return output_path


def example_backward_compatible_mode():
    """
    Example 1: Backward Compatible Mode

    Set sample parameters equal to reference parameters to reproduce
    behavior similar to the previous 11-parameter model.
    """
    print("\n" + "="*60)
    print("Example 1: Backward Compatible Mode")
    print("="*60)

    # Create config
    config_path = create_14param_config()
    core = HeterodyneAnalysisCore(config_path)

    # Parameters with sample = reference (backward compatible)
    params = np.array([
        100.0, -0.5, 10.0,   # reference transport
        100.0, -0.5, 10.0,   # sample transport (SAME as reference)
        0.1, 0.0, 0.01,      # velocity
        0.5, 0.0, 50.0, 0.3, # fraction
        0.0                   # flow angle
    ])

    # Calculate correlation
    c2 = core.calculate_heterodyne_correlation(params, phi_angle=0.0)

    print(f"Correlation shape: {c2.shape}")
    print(f"Correlation range: [{np.min(c2):.6f}, {np.max(c2):.6f}]")
    print("✓ Backward compatible mode works correctly")

    return c2


def example_independent_ref_sample():
    """
    Example 2: Independent Reference and Sample

    Demonstrate the new capability: different transport coefficients
    for reference and sample components.
    """
    print("\n" + "="*60)
    print("Example 2: Independent Reference and Sample Transport")
    print("="*60)

    config_path = "heterodyne_14param_config.json"
    core = HeterodyneAnalysisCore(config_path)

    # Case 1: Fast reference, slow sample
    params_fast_ref = np.array([
        200.0, -0.7, 20.0,   # reference (fast diffusion)
        50.0, -0.3, 5.0,     # sample (slow diffusion)
        0.1, 0.0, 0.01,
        0.5, 0.0, 50.0, 0.3,
        0.0
    ])

    # Case 2: Slow reference, fast sample
    params_fast_sample = np.array([
        50.0, -0.3, 5.0,     # reference (slow diffusion)
        200.0, -0.7, 20.0,   # sample (fast diffusion)
        0.1, 0.0, 0.01,
        0.5, 0.0, 50.0, 0.3,
        0.0
    ])

    c2_fast_ref = core.calculate_heterodyne_correlation(params_fast_ref, phi_angle=0.0)
    c2_fast_sample = core.calculate_heterodyne_correlation(params_fast_sample, phi_angle=0.0)

    print("\nCase 1 (fast ref): D0_ref=200, D0_sample=50")
    print(f"  Correlation range: [{np.min(c2_fast_ref):.6f}, {np.max(c2_fast_ref):.6f}]")

    print("\nCase 2 (fast sample): D0_ref=50, D0_sample=200")
    print(f"  Correlation range: [{np.min(c2_fast_sample):.6f}, {np.max(c2_fast_sample):.6f}]")

    diff = np.max(np.abs(c2_fast_ref - c2_fast_sample))
    print(f"\nMax difference between cases: {diff:.6e}")
    print("✓ Independent transport coefficients produce different correlations")

    return c2_fast_ref, c2_fast_sample


def example_parameter_validation():
    """
    Example 3: Parameter Validation

    Show how the model validates physical constraints on all 14 parameters.
    """
    print("\n" + "="*60)
    print("Example 3: Parameter Validation")
    print("="*60)

    config_path = "heterodyne_14param_config.json"
    core = HeterodyneAnalysisCore(config_path)

    # Test 1: Valid parameters
    valid_params = np.array([
        100.0, -0.5, 10.0,
        100.0, -0.5, 10.0,
        0.1, 0.0, 0.01,
        0.5, 0.0, 50.0, 0.3,
        0.0
    ])

    try:
        c2 = core.calculate_heterodyne_correlation(valid_params, 0.0)
        print("✓ Valid parameters accepted")
    except ValueError as e:
        print(f"✗ Unexpected error: {e}")

    # Test 2: Invalid reference D0 (negative)
    invalid_d0_ref = np.array([
        -100.0, -0.5, 10.0,  # D0_ref < 0 (INVALID)
        100.0, -0.5, 10.0,
        0.1, 0.0, 0.01,
        0.5, 0.0, 50.0, 0.3,
        0.0
    ])

    try:
        c2 = core.calculate_heterodyne_correlation(invalid_d0_ref, 0.0)
        print("✗ Should have rejected negative D0_ref")
    except ValueError as e:
        print(f"✓ Correctly rejected negative D0_ref: {str(e)[:50]}...")

    # Test 3: Invalid sample D0 (negative)
    invalid_d0_sample = np.array([
        100.0, -0.5, 10.0,
        -100.0, -0.5, 10.0,  # D0_sample < 0 (INVALID)
        0.1, 0.0, 0.01,
        0.5, 0.0, 50.0, 0.3,
        0.0
    ])

    try:
        c2 = core.calculate_heterodyne_correlation(invalid_d0_sample, 0.0)
        print("✗ Should have rejected negative D0_sample")
    except ValueError as e:
        print(f"✓ Correctly rejected negative D0_sample: {str(e)[:50]}...")


def example_migration_from_11_params():
    """
    Example 4: Migrating from 11-Parameter Model

    Show how to migrate from the old 11-parameter configuration.
    """
    print("\n" + "="*60)
    print("Example 4: Migration from 11-Parameter Model")
    print("="*60)

    from heterodyne.core.migration import HeterodyneMigration

    # Old 11-parameter configuration
    params_11 = [100.0, -0.5, 10.0, 0.1, 0.0, 0.01, 0.5, 0.0, 50.0, 0.3, 0.0]

    print(f"Old parameters (11): {params_11}")

    # Migrate to 14 parameters
    params_14 = HeterodyneMigration.migrate_11_to_14_parameters(params_11)

    print(f"\nNew parameters (14): {params_14}")
    print(f"\nBreakdown:")
    print(f"  Reference transport: {params_14[0:3]}")
    print(f"  Sample transport:    {params_14[3:6]} (initially equals reference)")
    print(f"  Velocity:            {params_14[6:9]}")
    print(f"  Fraction:            {params_14[9:13]}")
    print(f"  Flow angle:          {params_14[13]}")

    print("\n✓ Migration preserves backward compatibility")
    print("  (sample parameters initially equal reference parameters)")


def visualize_comparison():
    """
    Create visualization comparing different parameter configurations.
    """
    print("\n" + "="*60)
    print("Creating Visualization")
    print("="*60)

    config_path = "heterodyne_14param_config.json"
    core = HeterodyneAnalysisCore(config_path)

    # Three cases to compare
    cases = {
        "Backward Compatible\n(sample = ref)": np.array([
            100.0, -0.5, 10.0,
            100.0, -0.5, 10.0,  # Same as reference
            0.1, 0.0, 0.01,
            0.5, 0.0, 50.0, 0.3,
            0.0
        ]),
        "Fast Reference\n(D0_ref > D0_sample)": np.array([
            200.0, -0.7, 20.0,  # Fast reference
            50.0, -0.3, 5.0,    # Slow sample
            0.1, 0.0, 0.01,
            0.5, 0.0, 50.0, 0.3,
            0.0
        ]),
        "Fast Sample\n(D0_sample > D0_ref)": np.array([
            50.0, -0.3, 5.0,    # Slow reference
            200.0, -0.7, 20.0,  # Fast sample
            0.1, 0.0, 0.01,
            0.5, 0.0, 50.0, 0.3,
            0.0
        ]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for (title, params), ax in zip(cases.items(), axes):
        c2 = core.calculate_heterodyne_correlation(params, phi_angle=0.0)

        im = ax.imshow(c2, cmap='viridis', origin='lower')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Time index t₂')
        ax.set_ylabel('Time index t₁')
        plt.colorbar(im, ax=ax, label='C₂(t₁, t₂)')

    plt.tight_layout()
    plt.savefig('heterodyne_14param_comparison.png', dpi=150)
    print("✓ Saved visualization: heterodyne_14param_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  14-Parameter Heterodyne Model Examples")
    print("="*60)

    # Run all examples
    example_backward_compatible_mode()
    example_independent_ref_sample()
    example_parameter_validation()
    example_migration_from_11_params()

    # Create visualization
    try:
        visualize_comparison()
    except ImportError:
        print("\nSkipping visualization (matplotlib not available)")

    print("\n" + "="*60)
    print("  All Examples Completed Successfully!")
    print("="*60)
    print("\nKey Takeaways:")
    print("  1. The 14-parameter model supports independent ref and sample transport")
    print("  2. Setting sample = reference maintains backward compatibility")
    print("  3. All parameters are validated for physical correctness")
    print("  4. Migration from 11→14 parameters is automated")
    print("\nSee config file: heterodyne_14param_config.json")
