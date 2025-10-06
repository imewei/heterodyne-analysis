"""
Integration tests for heterodyne model pipeline.

End-to-end tests verifying the complete heterodyne analysis workflow
from configuration to correlation calculation.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from heterodyne.analysis.core import HeterodyneAnalysisCore
from heterodyne.core.config import ConfigManager


class TestHeterodyneIntegration:
    """Integration tests for complete heterodyne pipeline."""

    @pytest.fixture
    def heterodyne_config_file(self, tmp_path):
        """Create a complete heterodyne configuration file."""
        config = {
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
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 0, "end_frame": 100},
                "scattering": {"wavevector_q": 0.0054},
                "geometry": {"stator_rotor_gap": 2000000}
            },
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "options": {"maxiter": 10}
                }
            }
        }

        config_file = tmp_path / "heterodyne_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        return str(config_file)

    def test_11_parameter_pipeline(self, heterodyne_config_file):
        """Test complete pipeline with 11 parameters."""
        # Initialize core analysis
        core = HeterodyneAnalysisCore(heterodyne_config_file)

        # Verify 11-parameter configuration
        assert hasattr(core, 'config_manager')
        param_count = core.config_manager.get_effective_parameter_count()
        assert param_count == 11, f"Expected 11 parameters, got {param_count}"

        # Get active parameters
        active_params = core.config_manager.get_active_parameters()
        expected_params = [
            "D0", "alpha", "D_offset",
            "v0", "beta", "v_offset",
            "f0", "f1", "f2", "f3",
            "phi0"
        ]
        assert active_params == expected_params

    def test_heterodyne_correlation_calculation(self, heterodyne_config_file):
        """Test heterodyne correlation function calculation."""
        core = HeterodyneAnalysisCore(heterodyne_config_file)

        params = np.array([100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0])
        phi_angle = 0.0

        # Calculate heterodyne correlation
        c2 = core.calculate_heterodyne_correlation(params, phi_angle)

        # Verify output shape and properties
        assert c2.shape == (core.n_time, core.n_time)
        assert np.all(np.isfinite(c2)), "Correlation should be finite"
        assert np.all(c2 >= 0), "Correlation should be non-negative"

        # Verify symmetry
        assert np.allclose(c2, c2.T), "Correlation should be symmetric"

    def test_parameter_validation_in_pipeline(self, heterodyne_config_file):
        """Test that parameter validation works in full pipeline."""
        core = HeterodyneAnalysisCore(heterodyne_config_file)

        # Valid parameters should work
        valid_params = np.array([100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                                0.5, 0.0, 50.0, 0.3, 0.0])
        c2 = core.calculate_heterodyne_correlation(valid_params, 0.0)
        assert c2 is not None

        # Invalid parameters should raise error
        invalid_params = np.array([100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                                  2.0, 0.0, 50.0, -0.5, 0.0])  # f(t) may go outside [0,1]

        with pytest.raises(ValueError, match="Fraction"):
            core.calculate_heterodyne_correlation(invalid_params, 0.0)

    def test_multi_angle_calculation(self, heterodyne_config_file):
        """Test correlation calculation for multiple angles."""
        core = HeterodyneAnalysisCore(heterodyne_config_file)

        params = np.array([100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0])
        phi_angles = np.array([0, 45, 90, 135])

        # Calculate for all angles
        c2_results = core.calculate_c2_nonequilibrium_laminar_parallel(
            params, phi_angles
        )

        # Verify output
        assert c2_results.shape == (len(phi_angles), core.n_time, core.n_time)
        assert np.all(np.isfinite(c2_results))

        # Different angles should give different results
        assert not np.allclose(c2_results[0], c2_results[2])

    def test_parameter_metadata_integration(self, heterodyne_config_file):
        """Test parameter metadata in pipeline."""
        core = HeterodyneAnalysisCore(heterodyne_config_file)
        config_manager = core.config_manager

        # Get metadata
        metadata = config_manager.get_parameter_metadata()
        assert len(metadata) == 11

        # Verify each parameter has metadata
        for param_name in ["D0", "alpha", "D_offset", "v0", "beta", "v_offset",
                           "f0", "f1", "f2", "f3", "phi0"]:
            assert param_name in metadata
            assert "unit" in metadata[param_name]
            assert "description" in metadata[param_name]
            assert "index" in metadata[param_name]

        # Get bounds
        bounds = config_manager.get_parameter_bounds()
        assert len(bounds) == 11
        assert all(isinstance(b, tuple) and len(b) == 2 for b in bounds)

    def test_backward_incompatibility_with_7_params(self, tmp_path):
        """Test that 7-parameter configs raise clear errors."""
        legacy_config = {
            "initial_parameters": {
                "values": [1324.1, -0.014, -0.674, 0.003, -0.909, 0.0, 0.0]
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 0, "end_frame": 100},
                "scattering": {"wavevector_q": 0.0054},
                "geometry": {"stator_rotor_gap": 2000000}
            }
        }

        config_file = tmp_path / "legacy_7param.json"
        with open(config_file, 'w') as f:
            json.dump(legacy_config, f)

        core = HeterodyneAnalysisCore(str(config_file))

        # 7 parameters should fail validation
        legacy_params = np.array([1324.1, -0.014, -0.674, 0.003, -0.909, 0.0, 0.0])

        with pytest.raises(ValueError, match="11 parameters"):
            core.calculate_heterodyne_correlation(legacy_params, 0.0)


class TestHeterodyneOptimizationIntegration:
    """Integration tests for heterodyne optimization pipeline."""

    @pytest.fixture
    def optimization_config_file(self, tmp_path):
        """Create config with optimization settings."""
        config = {
            "initial_parameters": {
                "values": [100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0]
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 0, "end_frame": 50},
                "scattering": {"wavevector_q": 0.0054},
                "geometry": {"stator_rotor_gap": 2000000}
            },
            "optimization_config": {
                "classical_optimization": {
                    "methods": ["Nelder-Mead"],
                    "options": {"maxiter": 5}  # Limited for testing
                }
            }
        }

        config_file = tmp_path / "opt_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        return str(config_file)

    def test_optimization_with_11_parameters(self, optimization_config_file):
        """Test that optimization works with 11 parameters."""
        from heterodyne.optimization.classical import ClassicalOptimizer

        core = HeterodyneAnalysisCore(optimization_config_file)

        # Create optimizer
        with open(optimization_config_file, 'r') as f:
            config = json.load(f)

        optimizer = ClassicalOptimizer(core, config)

        # Initial parameters
        initial_params = np.array([100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                                  0.5, 0.0, 50.0, 0.3, 0.0])

        # Generate synthetic data
        phi_angles = np.array([0, 90])
        c2_synthetic = core.calculate_c2_nonequilibrium_laminar_parallel(
            initial_params, phi_angles
        )

        # Run optimization (should accept 11 parameters)
        result = optimizer.run_optimization(
            initial_params=initial_params,
            phi_angles=phi_angles,
            c2_experimental=c2_synthetic
        )

        # Verify result
        assert "parameters" in result
        assert len(result["parameters"]) == 11

    def test_default_parameter_fallback(self):
        """Test that default 11-parameter fallback works."""
        from heterodyne.core.config import ConfigManager

        config_manager = ConfigManager()
        defaults = config_manager.get_default_11_parameters()

        assert len(defaults) == 11
        assert defaults == [100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                           0.5, 0.0, 50.0, 0.3, 0.0]


class TestHeterodyneMigrationIntegration:
    """Integration tests for migration workflow."""

    def test_end_to_end_migration_workflow(self, tmp_path):
        """Test complete migration from 7-param to 11-param and analysis."""
        from heterodyne.core.migration import HeterodyneMigration

        # Create legacy 7-parameter config
        legacy_config = {
            "initial_parameters": {
                "values": [1324.1, -0.014, -0.674, 0.003, -0.909, 0.0, 0.0],
                "parameter_names": [
                    "D0", "alpha", "D_offset",
                    "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"
                ]
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 0, "end_frame": 50},
                "scattering": {"wavevector_q": 0.0054},
                "geometry": {"stator_rotor_gap": 2000000}
            }
        }

        legacy_file = tmp_path / "legacy.json"
        with open(legacy_file, 'w') as f:
            json.dump(legacy_config, f)

        # Migrate
        migrated_file = tmp_path / "migrated.json"
        migrated = HeterodyneMigration.migrate_config_file(legacy_file, migrated_file)

        # Verify migrated config
        assert len(migrated["initial_parameters"]["values"]) == 11
        assert "migration_info" in migrated

        # Use migrated config in analysis
        core = HeterodyneAnalysisCore(str(migrated_file))
        params = np.array(migrated["initial_parameters"]["values"])

        # Should work with heterodyne correlation
        c2 = core.calculate_heterodyne_correlation(params, 0.0)
        assert c2.shape == (core.n_time, core.n_time)
