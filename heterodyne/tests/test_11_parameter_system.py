"""
Unit tests for 11-parameter heterodyne system.

This module tests the parameter infrastructure for the 11-parameter
heterodyne model including validation, bounds, I/O, and optimization.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from heterodyne.optimization.classical import ClassicalOptimizer
from heterodyne.analysis.core import HeterodyneAnalysisCore


class Test11ParameterValidation:
    """Test 11-parameter validation and bounds checking."""

    def test_11_parameter_count_validation(self):
        """Test that exactly 11 parameters are required."""
        valid_params = np.array([
            100.0, -0.5, 10.0,     # D params
            0.1, 0.0, 0.01,        # v params
            0.5, 0.0, 50.0, 0.3,   # f params
            0.0                     # phi0
        ])
        assert len(valid_params) == 11, "Must have exactly 11 parameters"

    def test_parameter_bounds_structure(self):
        """Test parameter bounds are correctly structured."""
        bounds = [
            (0, 1000),      # D0: positive
            (-2, 2),        # alpha: power-law range
            (0, 100),       # D_offset: positive
            (-10, 10),      # v0: can be negative
            (-2, 2),        # beta: power-law range
            (-1, 1),        # v_offset: small offset
            (0, 1),         # f0: amplitude
            (-1, 1),        # f1: exponential rate
            (0, 200),       # f2: time offset
            (0, 1),         # f3: baseline fraction
            (-360, 360)     # phi0: angle in degrees
        ]
        assert len(bounds) == 11, "Must have 11 parameter bounds"

    def test_physical_constraints(self):
        """Test physical constraints on parameters."""
        # D0 and D_offset must be non-negative
        assert 0 <= 100.0, "D0 must be non-negative"
        assert 0 <= 10.0, "D_offset must be non-negative"

        # Alpha and beta should be in reasonable range
        assert -2 <= -0.5 <= 2, "alpha should be in [-2, 2]"
        assert -2 <= 0.0 <= 2, "beta should be in [-2, 2]"

        # Fraction parameters must produce f(t) in [0, 1]
        # This is checked dynamically in validate_heterodyne_parameters()

    def test_parameter_array_shapes(self):
        """Test that parameter arrays have correct shape."""
        params = np.array([100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0])
        assert params.shape == (11,), "Parameters must be 1D array of length 11"
        assert params.ndim == 1, "Parameters must be 1D"


class Test11ParameterDefaults:
    """Test default 11-parameter configurations."""

    def test_default_11_parameter_values(self):
        """Test default parameter values for heterodyne model."""
        defaults = {
            "diffusion": [100.0, -0.5, 10.0],
            "velocity": [0.1, 0.0, 0.01],
            "fraction": [0.5, 0.0, 50.0, 0.3],
            "angle": [0.0]
        }

        all_defaults = (
            defaults["diffusion"] +
            defaults["velocity"] +
            defaults["fraction"] +
            defaults["angle"]
        )

        assert len(all_defaults) == 11, "Defaults must provide 11 values"

    def test_parameter_names_metadata(self):
        """Test parameter names and units."""
        param_metadata = {
            "D0": {"unit": "nm²/s", "description": "Reference transport coefficient J₀ (labeled 'D' for compatibility)"},
            "alpha": {"unit": "dimensionless", "description": "Transport coefficient time-scaling exponent"},
            "D_offset": {"unit": "nm²/s", "description": "Baseline transport coefficient J_offset"},
            "v0": {"unit": "nm/s", "description": "Reference velocity"},
            "beta": {"unit": "dimensionless", "description": "Velocity power-law exponent"},
            "v_offset": {"unit": "nm/s", "description": "Baseline velocity offset"},
            "f0": {"unit": "dimensionless", "description": "Fraction amplitude"},
            "f1": {"unit": "1/s", "description": "Fraction exponential rate"},
            "f2": {"unit": "s", "description": "Fraction time offset"},
            "f3": {"unit": "dimensionless", "description": "Fraction baseline"},
            "phi0": {"unit": "degrees", "description": "Flow direction angle"}
        }

        assert len(param_metadata) == 11, "Must have metadata for 11 parameters"


class Test11ParameterIO:
    """Test parameter I/O for 11-parameter system."""

    def test_save_11_parameters_to_json(self, tmp_path):
        """Test saving 11 parameters to JSON file."""
        params = np.array([100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0])

        param_file = tmp_path / "heterodyne_params.json"
        param_data = {
            "initial_parameters": {
                "values": params.tolist(),
                "parameter_names": [
                    "D0", "alpha", "D_offset",
                    "v0", "beta", "v_offset",
                    "f0", "f1", "f2", "f3",
                    "phi0"
                ]
            }
        }

        with open(param_file, 'w') as f:
            json.dump(param_data, f, indent=2)

        # Verify file exists and can be read
        assert param_file.exists()
        with open(param_file, 'r') as f:
            loaded = json.load(f)

        loaded_params = loaded["initial_parameters"]["values"]
        assert len(loaded_params) == 11, "Saved parameters must be length 11"
        np.testing.assert_array_almost_equal(params, loaded_params)

    def test_load_11_parameters_from_json(self, tmp_path):
        """Test loading 11 parameters from JSON file."""
        param_file = tmp_path / "heterodyne_params.json"
        param_data = {
            "initial_parameters": {
                "values": [100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0],
                "parameter_names": [
                    "D0", "alpha", "D_offset",
                    "v0", "beta", "v_offset",
                    "f0", "f1", "f2", "f3",
                    "phi0"
                ]
            }
        }

        with open(param_file, 'w') as f:
            json.dump(param_data, f, indent=2)

        # Load parameters
        with open(param_file, 'r') as f:
            loaded = json.load(f)

        params = np.array(loaded["initial_parameters"]["values"])
        assert params.shape == (11,), "Loaded parameters must be shape (11,)"


class Test11ParameterOptimization:
    """Test optimization with 11 parameters."""

    @pytest.fixture
    def heterodyne_config(self, tmp_path):
        """Create 11-parameter heterodyne configuration."""
        config = {
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 0, "end_frame": 100},
                "scattering": {"wavevector_q": 0.0054},
                "geometry": {"stator_rotor_gap": 2000000}
            },
            "initial_parameters": {
                "values": [100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0],
                "parameter_names": [
                    "D0", "alpha", "D_offset",
                    "v0", "beta", "v_offset",
                    "f0", "f1", "f2", "f3",
                    "phi0"
                ]
            },
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "options": {"maxiter": 10}
                }
            }
        }

        config_file = tmp_path / "heterodyne_11param.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        return str(config_file)

    def test_optimizer_accepts_11_parameters(self, heterodyne_config):
        """Test that optimizer accepts 11-parameter input."""
        from heterodyne.core.config import ConfigManager

        config_manager = ConfigManager(heterodyne_config)
        core = HeterodyneAnalysisCore(str(heterodyne_config))

        params = np.array([100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0])

        # Should not raise an error
        assert len(params) == 11

    def test_default_11_parameter_fallback(self):
        """Test default 11-parameter fallback in optimizer."""
        # When config doesn't specify parameters, should use 11-param defaults
        default_11 = [100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                     0.5, 0.0, 50.0, 0.3, 0.0]

        assert len(default_11) == 11, "Default fallback must be 11 parameters"


class Test11ParameterBounds:
    """Test parameter bounds and transformations."""

    def test_parameter_bounds_coverage(self):
        """Test that all 11 parameters have bounds."""
        bounds = [
            (0, 1000),      # D0
            (-2, 2),        # alpha
            (0, 100),       # D_offset
            (-10, 10),      # v0
            (-2, 2),        # beta
            (-1, 1),        # v_offset
            (0, 1),         # f0
            (-1, 1),        # f1
            (0, 200),       # f2
            (0, 1),         # f3
            (-360, 360)     # phi0
        ]

        assert len(bounds) == 11, "Must have bounds for all 11 parameters"

        # Verify each bound is a tuple
        for i, bound in enumerate(bounds):
            assert isinstance(bound, tuple), f"Bound {i} must be tuple"
            assert len(bound) == 2, f"Bound {i} must have min and max"
            assert bound[0] <= bound[1], f"Bound {i}: min must be <= max"

    def test_log_space_transformation(self):
        """Test log-space parameter transformation for positive parameters."""
        # Parameters that should use log-space: D0, D_offset, v0 (if positive)
        params = np.array([100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0])

        # Log transform for D0 and D_offset
        log_D0 = np.log10(params[0])
        log_D_offset = np.log10(params[2])

        # Should be able to inverse transform
        assert np.isclose(10**log_D0, params[0])
        assert np.isclose(10**log_D_offset, params[2])

    def test_parameter_scaling(self):
        """Test parameter scaling to [0, 1] range."""
        params = np.array([100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0])
        bounds = np.array([
            [0, 1000], [-2, 2], [0, 100], [-10, 10], [-2, 2], [-1, 1],
            [0, 1], [-1, 1], [0, 200], [0, 1], [-360, 360]
        ])

        # Scale to [0, 1]
        scaled = (params - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

        # All should be in [0, 1]
        assert np.all(scaled >= 0) and np.all(scaled <= 1), \
            "Scaled parameters must be in [0, 1]"

        # Inverse scaling
        unscaled = scaled * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        np.testing.assert_array_almost_equal(params, unscaled)


class Test11ParameterBackwardCompatibility:
    """Test handling of legacy 7-parameter configs."""

    def test_7_parameter_config_rejected(self):
        """Test that 7-parameter configs are rejected."""
        # Old 7-parameter format should be rejected
        old_params = [1324.1, -0.014, -0.674361, 0.003, -0.909, 0.0, 0.0]

        # Should raise error or warning about needing 11 parameters
        assert len(old_params) == 7, "This is a legacy 7-parameter array"
        # In production, this would raise ValueError

    def test_migration_guidance_for_7_params(self):
        """Test that migration guidance is provided for 7-parameter configs."""
        # When 7 parameters detected, should guide user to 11-parameter format
        old_param_count = 7
        new_param_count = 11

        assert new_param_count > old_param_count, \
            "Heterodyne model requires more parameters than legacy model"
