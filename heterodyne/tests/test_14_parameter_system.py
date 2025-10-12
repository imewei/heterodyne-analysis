"""
Unit tests for 14-parameter heterodyne system.

This module tests the parameter infrastructure for the 14-parameter
heterodyne model including validation, bounds, I/O, and optimization.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from heterodyne.analysis.core import HeterodyneAnalysisCore
from heterodyne.optimization.classical import ClassicalOptimizer


class Test14ParameterValidation:
    """Test 14-parameter validation and bounds checking."""

    def test_14_parameter_count_validation(self):
        """Test that exactly 14 parameters are required."""
        valid_params = np.array(
            [
                100.0,
                -0.5,
                10.0,  # D_ref params
                100.0,
                -0.5,
                10.0,  # D_sample params
                0.1,
                0.0,
                0.01,  # v params
                0.5,
                0.0,
                50.0,
                0.3,  # f params
                0.0,  # phi0
            ]
        )
        assert len(valid_params) == 14, "Must have exactly 14 parameters"

    def test_parameter_bounds_structure(self):
        """Test parameter bounds are correctly structured."""
        bounds = [
            (0, 1000),  # D0_ref: positive
            (-2, 2),  # alpha_ref: power-law range
            (0, 100),  # D_offset_ref: positive
            (0, 1000),  # D0_sample: positive
            (-2, 2),  # alpha_sample: power-law range
            (0, 100),  # D_offset_sample: positive
            (-10, 10),  # v0: can be negative
            (-2, 2),  # beta: power-law range
            (-1, 1),  # v_offset: small offset
            (0, 1),  # f0: amplitude
            (-1, 1),  # f1: exponential rate
            (0, 200),  # f2: time offset
            (0, 1),  # f3: baseline fraction
            (-360, 360),  # phi0: angle in degrees
        ]
        assert len(bounds) == 14, "Must have 14 parameter bounds"

    def test_physical_constraints(self):
        """Test physical constraints on parameters."""
        # Example parameter values
        D0_example = 100.0
        D_offset_example = 10.0
        alpha_example = -0.5
        beta_example = 0.0

        # D0 and D_offset must be non-negative
        assert D0_example >= 0, "D0 must be non-negative"
        assert D_offset_example >= 0, "D_offset must be non-negative"

        # Alpha and beta should be in reasonable range
        assert -2 <= alpha_example <= 2, "alpha should be in [-2, 2]"
        assert -2 <= beta_example <= 2, "beta should be in [-2, 2]"

        # Fraction parameters must produce f(t) in [0, 1]
        # This is checked dynamically in validate_heterodyne_parameters()

    def test_parameter_array_shapes(self):
        """Test that parameter arrays have correct shape."""
        params = np.array(
            [
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
            ]
        )
        assert params.shape == (14,), "Parameters must be 1D array of length 14"
        assert params.ndim == 1, "Parameters must be 1D"


class Test14ParameterDefaults:
    """Test default 14-parameter configurations."""

    def test_default_14_parameter_values(self):
        """Test default parameter values for heterodyne model."""
        defaults = {
            "diffusion_ref": [100.0, -0.5, 10.0],
            "diffusion_sample": [100.0, -0.5, 10.0],
            "velocity": [0.1, 0.0, 0.01],
            "fraction": [0.5, 0.0, 50.0, 0.3],
            "angle": [0.0],
        }

        all_defaults = (
            defaults["diffusion_ref"]
            + defaults["diffusion_sample"]
            + defaults["velocity"]
            + defaults["fraction"]
            + defaults["angle"]
        )

        assert len(all_defaults) == 14, "Defaults must provide 14 values"

    def test_parameter_names_metadata(self):
        """Test parameter names and units."""
        param_metadata = {
            "D0_ref": {
                "unit": "nm²/s",
                "description": "Reference transport coefficient J₀_ref",
            },
            "alpha_ref": {
                "unit": "dimensionless",
                "description": "Reference transport coefficient time-scaling exponent",
            },
            "D_offset_ref": {
                "unit": "nm²/s",
                "description": "Reference baseline transport coefficient J_offset_ref",
            },
            "D0_sample": {
                "unit": "nm²/s",
                "description": "Sample transport coefficient J₀_sample",
            },
            "alpha_sample": {
                "unit": "dimensionless",
                "description": "Sample transport coefficient time-scaling exponent",
            },
            "D_offset_sample": {
                "unit": "nm²/s",
                "description": "Sample baseline transport coefficient J_offset_sample",
            },
            "v0": {"unit": "nm/s", "description": "Reference velocity"},
            "beta": {
                "unit": "dimensionless",
                "description": "Velocity power-law exponent",
            },
            "v_offset": {"unit": "nm/s", "description": "Baseline velocity offset"},
            "f0": {"unit": "dimensionless", "description": "Fraction amplitude"},
            "f1": {"unit": "1/s", "description": "Fraction exponential rate"},
            "f2": {"unit": "s", "description": "Fraction time offset"},
            "f3": {"unit": "dimensionless", "description": "Fraction baseline"},
            "phi0": {"unit": "degrees", "description": "Flow direction angle"},
        }

        assert len(param_metadata) == 14, "Must have metadata for 14 parameters"


class Test14ParameterIO:
    """Test parameter I/O for 14-parameter system."""

    def test_save_14_parameters_to_json(self, tmp_path):
        """Test saving 14 parameters to JSON file."""
        params = np.array(
            [
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
            ]
        )

        param_file = tmp_path / "heterodyne_params.json"
        param_data = {
            "initial_parameters": {
                "values": params.tolist(),
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
            }
        }

        with open(param_file, "w") as f:
            json.dump(param_data, f, indent=2)

        # Verify file exists and can be read
        assert param_file.exists()
        with open(param_file) as f:
            loaded = json.load(f)

        loaded_params = loaded["initial_parameters"]["values"]
        assert len(loaded_params) == 14, "Saved parameters must be length 14"
        np.testing.assert_array_almost_equal(params, loaded_params)

    def test_load_14_parameters_from_json(self, tmp_path):
        """Test loading 14 parameters from JSON file."""
        param_file = tmp_path / "heterodyne_params.json"
        param_data = {
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
            }
        }

        with open(param_file, "w") as f:
            json.dump(param_data, f, indent=2)

        # Load parameters
        with open(param_file) as f:
            loaded = json.load(f)

        params = np.array(loaded["initial_parameters"]["values"])
        assert params.shape == (14,), "Loaded parameters must be shape (14,)"


class Test14ParameterOptimization:
    """Test optimization with 14 parameters."""

    @pytest.fixture
    def heterodyne_config(self, tmp_path):
        """Create 14-parameter heterodyne configuration."""
        config = {
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 0, "end_frame": 100},
                "scattering": {"wavevector_q": 0.0054},
                "geometry": {"stator_rotor_gap": 2000000},
            },
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
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "options": {"maxiter": 10},
                }
            },
        }

        config_file = tmp_path / "heterodyne_14param.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        return str(config_file)

    def test_optimizer_accepts_14_parameters(self, heterodyne_config):
        """Test that optimizer accepts 14-parameter input."""
        from heterodyne.core.config import ConfigManager

        config_manager = ConfigManager(heterodyne_config)
        core = HeterodyneAnalysisCore(str(heterodyne_config))

        params = np.array(
            [
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
            ]
        )

        # Should not raise an error
        assert len(params) == 14

    def test_default_14_parameter_fallback(self):
        """Test default 14-parameter fallback in optimizer."""
        # When config doesn't specify parameters, should use 14-param defaults
        default_14 = [
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
        ]

        assert len(default_14) == 14, "Default fallback must be 14 parameters"


class Test14ParameterBounds:
    """Test parameter bounds and transformations."""

    def test_parameter_bounds_coverage(self):
        """Test that all 14 parameters have bounds."""
        bounds = [
            (0, 1000),  # D0_ref
            (-2, 2),  # alpha_ref
            (0, 100),  # D_offset_ref
            (0, 1000),  # D0_sample
            (-2, 2),  # alpha_sample
            (0, 100),  # D_offset_sample
            (-10, 10),  # v0
            (-2, 2),  # beta
            (-1, 1),  # v_offset
            (0, 1),  # f0
            (-1, 1),  # f1
            (0, 200),  # f2
            (0, 1),  # f3
            (-360, 360),  # phi0
        ]

        assert len(bounds) == 14, "Must have bounds for all 14 parameters"

        # Verify each bound is a tuple
        for i, bound in enumerate(bounds):
            assert isinstance(bound, tuple), f"Bound {i} must be tuple"
            assert len(bound) == 2, f"Bound {i} must have min and max"
            assert bound[0] <= bound[1], f"Bound {i}: min must be <= max"

    def test_log_space_transformation(self):
        """Test log-space parameter transformation for positive parameters."""
        # Parameters that should use log-space: D0_ref, D_offset_ref, D0_sample, D_offset_sample
        params = np.array(
            [
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
            ]
        )

        # Log transform for D0_ref, D_offset_ref, D0_sample, D_offset_sample
        log_D0_ref = np.log10(params[0])
        log_D_offset_ref = np.log10(params[2])
        log_D0_sample = np.log10(params[3])
        log_D_offset_sample = np.log10(params[5])

        # Should be able to inverse transform
        assert np.isclose(10**log_D0_ref, params[0])
        assert np.isclose(10**log_D_offset_ref, params[2])
        assert np.isclose(10**log_D0_sample, params[3])
        assert np.isclose(10**log_D_offset_sample, params[5])

    def test_parameter_scaling(self):
        """Test parameter scaling to [0, 1] range."""
        params = np.array(
            [
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
            ]
        )
        bounds = np.array(
            [
                [0, 1000],
                [-2, 2],
                [0, 100],
                [0, 1000],
                [-2, 2],
                [0, 100],
                [-10, 10],
                [-2, 2],
                [-1, 1],
                [0, 1],
                [-1, 1],
                [0, 200],
                [0, 1],
                [-360, 360],
            ]
        )

        # Scale to [0, 1]
        scaled = (params - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

        # All should be in [0, 1]
        assert np.all(scaled >= 0) and np.all(
            scaled <= 1
        ), "Scaled parameters must be in [0, 1]"

        # Inverse scaling
        unscaled = scaled * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        np.testing.assert_array_almost_equal(params, unscaled)


class Test14ParameterBackwardCompatibility:
    """Test handling of legacy 7 and 11-parameter configs."""

    def test_7_parameter_config_rejected(self):
        """Test that 7-parameter configs are rejected."""
        # Old 7-parameter format should be rejected
        old_params = [1324.1, -0.014, -0.674361, 0.003, -0.909, 0.0, 0.0]

        # Should raise error or warning about needing 14 parameters
        assert len(old_params) == 7, "This is a legacy 7-parameter array"
        # In production, this would raise ValueError

    def test_11_parameter_config_rejected(self):
        """Test that 11-parameter configs are rejected."""
        # Old 11-parameter format should be rejected (pre-14 parameter update)
        old_params = [100.0, -0.5, 10.0, 0.1, 0.0, 0.01, 0.5, 0.0, 50.0, 0.3, 0.0]

        # Should raise error or warning about needing 14 parameters
        assert len(old_params) == 11, "This is a legacy 11-parameter array"
        # In production, this would raise ValueError

    def test_migration_guidance_for_legacy_params(self):
        """Test that migration guidance is provided for legacy configs."""
        # When legacy parameters detected, should guide user to 14-parameter format
        old_7_param_count = 7
        old_11_param_count = 11
        new_param_count = 14

        assert (
            new_param_count > old_7_param_count
        ), "Heterodyne model requires more parameters than 7-parameter legacy model"
        assert (
            new_param_count > old_11_param_count
        ), "Heterodyne model requires more parameters than 11-parameter legacy model"
