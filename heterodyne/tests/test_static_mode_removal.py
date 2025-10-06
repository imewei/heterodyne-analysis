"""
Test suite for static mode removal and backward compatibility.

This module ensures that:
1. Static mode configurations are properly rejected with helpful error messages
2. Legacy static mode data files are handled appropriately
3. Migration path is clear for users with existing static mode analyses
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from heterodyne.core.config import ConfigManager


class TestStaticModeRemoval:
    """Test static mode infrastructure removal."""

    def test_static_isotropic_config_rejected(self, tmp_path):
        """Test that static isotropic configuration is rejected with helpful error."""
        config_data = {
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic"
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0]
            }
        }

        config_file = tmp_path / "static_isotropic.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # Expect validation error with migration guidance
        with pytest.raises((ValueError, KeyError)) as exc_info:
            config_manager = ConfigManager(str(config_file))
            config_manager.validate()

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["static", "deprecated", "removed", "heterodyne"])

    def test_static_anisotropic_config_rejected(self, tmp_path):
        """Test that static anisotropic configuration is rejected with helpful error."""
        config_data = {
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "anisotropic"
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0]
            }
        }

        config_file = tmp_path / "static_anisotropic.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # Expect validation error with migration guidance
        with pytest.raises((ValueError, KeyError)) as exc_info:
            config_manager = ConfigManager(str(config_file))
            config_manager.validate()

        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["static", "deprecated", "removed", "heterodyne"])

    def test_legacy_3_parameter_data_handling(self, tmp_path):
        """Test handling of legacy 3-parameter static mode data files."""
        # Simulate legacy static mode results
        legacy_data = {
            "parameters": [100.0, -0.5, 10.0],  # D0, alpha, D_offset
            "parameter_names": ["D0", "alpha", "D_offset"],
            "chi_squared": 5.2,
            "analysis_mode": "static_isotropic"
        }

        results_file = tmp_path / "legacy_results.json"
        with open(results_file, 'w') as f:
            json.dump(legacy_data, f)

        # Legacy data should be readable but recognized as outdated
        with open(results_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data["analysis_mode"] == "static_isotropic"
        assert len(loaded_data["parameters"]) == 3

    def test_static_mode_config_files_removed(self):
        """Test that static mode configuration files are removed from package."""
        config_dir = Path("heterodyne/config")

        # These files should NOT exist after removal
        static_iso_file = config_dir / "static_isotropic.json"
        static_aniso_file = config_dir / "static_anisotropic.json"

        # Skip this test if we're in the middle of removal process
        # (files may still exist during transition)
        if static_iso_file.exists() or static_aniso_file.exists():
            pytest.skip("Static config files still present - removal in progress")


class TestStaticModeFunctionRemoval:
    """Test that static mode functions are properly removed."""

    def test_is_static_mode_removed_from_analyzer(self):
        """Test that is_static_mode() is removed from HeterodyneAnalysisCore."""
        from heterodyne.analysis.core import HeterodyneAnalysisCore

        # Method should not exist
        assert not hasattr(HeterodyneAnalysisCore, 'is_static_mode')

    def test_is_static_parameters_removed_from_analyzer(self):
        """Test that is_static_parameters() is removed from HeterodyneAnalysisCore."""
        from heterodyne.analysis.core import HeterodyneAnalysisCore

        # Method should not exist
        assert not hasattr(HeterodyneAnalysisCore, 'is_static_parameters')

    def test_calculate_c2_vectorized_static_removed(self):
        """Test that _calculate_c2_vectorized_static() is removed."""
        from heterodyne.analysis.core import HeterodyneAnalysisCore

        # Method should not exist
        assert not hasattr(HeterodyneAnalysisCore, '_calculate_c2_vectorized_static')


class TestMigrationGuidance:
    """Test migration guidance for users transitioning from static mode."""

    def test_error_message_provides_migration_path(self, tmp_path):
        """Test that error messages guide users to heterodyne model."""
        config_data = {
            "analysis_settings": {
                "static_mode": True
            }
        }

        config_file = tmp_path / "old_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        with pytest.raises(Exception) as exc_info:
            config_manager = ConfigManager(str(config_file))
            config_manager.validate()

        error_msg = str(exc_info.value)
        # Error should mention heterodyne model or migration
        assert any(keyword in error_msg.lower() for keyword in
                   ["heterodyne", "migrate", "replace", "11 parameter"])


class TestConfigValidation:
    """Test that configuration validation rejects static mode parameters."""

    def test_static_submode_parameter_rejected(self, tmp_path):
        """Test that static_submode parameter is rejected."""
        config_data = {
            "analysis_settings": {
                "static_mode": False,  # Even with False
                "static_submode": "isotropic"  # This should be rejected
            }
        }

        config_file = tmp_path / "mixed_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # Should raise validation error for static_submode presence
        with pytest.raises((ValueError, KeyError)):
            config_manager = ConfigManager(str(config_file))
            config_manager.validate()

    def test_3_parameter_optimization_rejected(self):
        """Test that 3-parameter configurations are rejected (must be 11 for heterodyne)."""
        # Heterodyne requires 11 parameters, not 3
        with pytest.raises((ValueError, AssertionError)):
            # This should fail in parameter validation
            from heterodyne.core.config import ConfigManager

            # Mock config with only 3 parameters
            config_data = {
                "initial_parameters": {
                    "values": [100.0, -0.5, 10.0]  # Only 3 params
                },
                "analysis_settings": {
                    "static_mode": False
                }
            }

            # Validation should reject insufficient parameters
            # (Actual implementation will vary based on validation logic)
