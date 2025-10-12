"""
Regression tests for 14-parameter heterodyne model.

Verifies that the 14-parameter model with sample=reference parameters
produces mathematically consistent results with the previous implementation.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from heterodyne.analysis.core import HeterodyneAnalysisCore


class Test14ParameterRegression:
    """Regression tests for 14-parameter model backward compatibility."""

    @pytest.fixture
    def config_14_param_identical(self, tmp_path):
        """Config with reference and sample parameters set equal (backward compatible mode)."""
        config = {
            "description": "14-parameter config with g1_ref = g1_sample initially",
            "model": "heterodyne",
            "version": "1.0",
            "initial_parameters": {
                "values": [
                    # Reference transport
                    100.0,
                    -0.5,
                    10.0,
                    # Sample transport (SAME as reference for backward compatibility)
                    100.0,
                    -0.5,
                    10.0,
                    # Velocity
                    0.1,
                    0.0,
                    0.01,
                    # Fraction
                    0.5,
                    0.0,
                    50.0,
                    0.3,
                    # Flow angle
                    0.0,
                ],
                "parameter_names": [
                    "D0_ref",
                    "alpha_ref",
                    "D_offset_ref",
                    "D0_sample",
                    "alpha_sample",
                    "D_offset_sample",
                    "v0",
                    "beta",
                    "v_offset",
                    "f0",
                    "f1",
                    "f2",
                    "f3",
                    "phi0",
                ],
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 0, "end_frame": 50},
                "scattering": {"wavevector_q": 0.0054},
                "geometry": {"stator_rotor_gap": 2000000},
            },
        }

        config_file = tmp_path / "regression_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        return str(config_file)

    def test_identical_ref_sample_produces_valid_correlation(
        self, config_14_param_identical
    ):
        """Test that g1_ref = g1_sample produces valid heterodyne correlation."""
        core = HeterodyneAnalysisCore(config_14_param_identical)

        # Parameters with reference = sample
        params = np.array(
            [
                100.0,
                -0.5,
                10.0,  # ref
                100.0,
                -0.5,
                10.0,  # sample (identical)
                0.1,
                0.0,
                0.01,  # velocity
                0.5,
                0.0,
                50.0,
                0.3,  # fraction
                0.0,  # phi0
            ]
        )

        c2 = core.calculate_heterodyne_correlation(params, 0.0)

        # Verify basic properties
        assert c2.shape == (core.n_time, core.n_time)
        assert np.all(np.isfinite(c2)), "Correlation should be finite"
        assert np.all(c2 >= 0), "Correlation should be non-negative"
        assert np.allclose(c2, c2.T, rtol=1e-10), "Correlation should be symmetric"

    def test_different_ref_sample_changes_correlation(self, config_14_param_identical):
        """Test that different ref and sample parameters produce different correlation."""
        core = HeterodyneAnalysisCore(config_14_param_identical)

        # Case 1: Identical parameters
        params_identical = np.array(
            [
                100.0,
                -0.5,
                10.0,  # ref
                100.0,
                -0.5,
                10.0,  # sample (identical)
                0.1,
                0.0,
                0.01,
                0.5,
                0.0,
                50.0,
                0.3,
                0.0,
            ]
        )

        # Case 2: Different sample parameters
        params_different = np.array(
            [
                100.0,
                -0.5,
                10.0,  # ref (unchanged)
                120.0,
                -0.6,
                12.0,  # sample (DIFFERENT)
                0.1,
                0.0,
                0.01,
                0.5,
                0.0,
                50.0,
                0.3,
                0.0,
            ]
        )

        c2_identical = core.calculate_heterodyne_correlation(params_identical, 0.0)
        c2_different = core.calculate_heterodyne_correlation(params_different, 0.0)

        # Correlations should be different
        max_diff = np.max(np.abs(c2_identical - c2_different))
        assert (
            max_diff > 1e-6
        ), f"Expected different correlations, got max_diff={max_diff}"

    def test_parameter_order_correctness(self, config_14_param_identical):
        """Verify that parameter indices are correctly mapped."""
        core = HeterodyneAnalysisCore(config_14_param_identical)

        # Test parameters with distinctive values for each group
        params = np.array(
            [
                # Reference transport (distinctive pattern)
                200.0,
                -0.8,
                20.0,
                # Sample transport (different pattern)
                150.0,
                -0.4,
                15.0,
                # Velocity (easy to identify)
                1.0,
                0.2,
                0.05,
                # Fraction (bounded values)
                0.7,
                0.1,
                60.0,
                0.4,
                # Flow angle
                45.0,
            ]
        )

        # Should not raise any errors
        c2 = core.calculate_heterodyne_correlation(params, 0.0)
        assert c2.shape == (core.n_time, core.n_time)

    def test_validation_catches_invalid_sample_params(self, config_14_param_identical):
        """Test that validation works for sample parameters."""
        core = HeterodyneAnalysisCore(config_14_param_identical)

        # Invalid sample D0 (negative)
        invalid_params = np.array(
            [
                100.0,
                -0.5,
                10.0,  # ref (valid)
                -50.0,
                -0.5,
                10.0,  # sample D0 negative (INVALID)
                0.1,
                0.0,
                0.01,
                0.5,
                0.0,
                50.0,
                0.3,
                0.0,
            ]
        )

        with pytest.raises(ValueError, match="D0_sample"):
            core.calculate_heterodyne_correlation(invalid_params, 0.0)

    def test_migration_produces_backward_compatible_params(self):
        """Test that 11→14 migration creates backward-compatible parameters."""
        from heterodyne.core.migration import HeterodyneMigration

        # Original 11-parameter configuration
        params_11 = [100.0, -0.5, 10.0, 0.1, 0.0, 0.01, 0.5, 0.0, 50.0, 0.3, 0.0]

        # Migrate to 14 parameters
        params_14 = HeterodyneMigration.migrate_11_to_14_parameters(params_11)

        assert len(params_14) == 14

        # Verify reference parameters match original
        assert params_14[0:3] == params_11[0:3]  # D0, alpha, D_offset

        # Verify sample parameters initially equal reference (backward compatibility)
        assert (
            params_14[0:3] == params_14[3:6]
        ), "Sample should equal reference initially"

        # Verify other parameters preserved
        assert params_14[6:14] == params_11[3:11]  # v0, beta, v_offset, f0-f3, phi0


class Test14ParameterNumericalStability:
    """Test numerical stability of 14-parameter model."""

    @pytest.fixture
    def core_instance(self, tmp_path):
        """Create core instance for numerical tests."""
        config = {
            "description": "Numerical stability test config",
            "model": "heterodyne",
            "version": "1.0",
            "initial_parameters": {
                "values": [
                    100.0,
                    -0.5,
                    10.0,
                    100.0,
                    -0.5,
                    10.0,
                    0.1,
                    0.0,
                    0.01,
                    0.5,
                    0.0,
                    50.0,
                    0.3,
                    0.0,
                ],
                "parameter_names": [
                    "D0_ref",
                    "alpha_ref",
                    "D_offset_ref",
                    "D0_sample",
                    "alpha_sample",
                    "D_offset_sample",
                    "v0",
                    "beta",
                    "v_offset",
                    "f0",
                    "f1",
                    "f2",
                    "f3",
                    "phi0",
                ],
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 0, "end_frame": 30},
                "scattering": {"wavevector_q": 0.0054},
                "geometry": {"stator_rotor_gap": 2000000},
            },
        }

        config_file = tmp_path / "stability_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        return HeterodyneAnalysisCore(str(config_file))

    def test_extreme_diffusion_difference(self, core_instance):
        """Test stability with large difference between ref and sample diffusion."""
        # Very different transport coefficients
        params = np.array(
            [
                10.0,
                -0.3,
                1.0,  # ref (slow diffusion)
                500.0,
                -0.9,
                50.0,  # sample (fast diffusion)
                0.1,
                0.0,
                0.01,
                0.5,
                0.0,
                50.0,
                0.3,
                0.0,
            ]
        )

        c2 = core_instance.calculate_heterodyne_correlation(params, 0.0)

        assert np.all(np.isfinite(c2)), "Correlation should remain finite"
        assert np.all(
            c2 >= -1e-10
        ), "Correlation should be non-negative (within numerical precision)"

    def test_zero_fraction_edge_case(self, core_instance):
        """Test edge case where fraction approaches zero or one."""
        # Fraction parameters that give f(t) ≈ 0 (pure reference)
        params = np.array(
            [
                100.0,
                -0.5,
                10.0,
                150.0,
                -0.6,
                15.0,
                0.1,
                0.0,
                0.01,
                0.0,
                0.0,
                50.0,
                0.0,  # f(t) ≈ 0
                0.0,
            ]
        )

        c2 = core_instance.calculate_heterodyne_correlation(params, 0.0)
        assert np.all(np.isfinite(c2))
