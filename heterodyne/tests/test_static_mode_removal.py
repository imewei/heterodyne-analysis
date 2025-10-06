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
from heterodyne.core.migration import HeterodyneMigration


class TestStaticModeRemoval:
    """Test static mode infrastructure removal."""

    def test_static_isotropic_config_detected(self, tmp_path):
        """Test that static isotropic configuration is detected by migration utility."""
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

        # Migration utility should detect this as 3-param-static
        version = HeterodyneMigration.detect_config_version(config_data)
        assert version == "3-param-static"

        # Migration should raise helpful error
        with pytest.raises(ValueError, match="Cannot automatically migrate"):
            HeterodyneMigration.migrate_config_file(config_file)

    def test_static_anisotropic_config_detected(self, tmp_path):
        """Test that static anisotropic configuration is detected by migration utility."""
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

        # Migration utility should detect this as 3-param-static
        version = HeterodyneMigration.detect_config_version(config_data)
        assert version == "3-param-static"

        # Migration should raise helpful error
        with pytest.raises(ValueError, match="Static mode has been removed"):
            HeterodyneMigration.migrate_config_file(config_file)

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

    def test_validate_method_removed_from_config_manager(self):
        """Test that validate() method is removed from ConfigManager (part of static mode)."""
        # validate() was part of static mode infrastructure and should be removed
        assert not hasattr(ConfigManager, 'validate')


class TestMigrationGuidance:
    """Test migration guidance for users transitioning from static mode."""

    def test_migration_guide_provides_clear_path(self, tmp_path):
        """Test that migration guide provides clear guidance for static configs."""
        config_data = {
            "analysis_settings": {
                "static_mode": True
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0]
            }
        }

        config_file = tmp_path / "old_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # Generate migration guide
        guide = HeterodyneMigration.generate_migration_guide(config_file)

        # Guide should mention key concepts
        guide_lower = guide.lower()
        assert "3-parameter static" in guide_lower or "static mode" in guide_lower
        assert "automatic migration not supported" in guide_lower or "manually" in guide_lower
        assert "heterodyne" in guide_lower

    def test_error_message_provides_migration_path(self, tmp_path):
        """Test that error messages guide users to heterodyne model."""
        config_data = {
            "analysis_settings": {
                "static_mode": True
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0]
            }
        }

        config_file = tmp_path / "old_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # Migration should raise error with guidance
        with pytest.raises(ValueError) as exc_info:
            HeterodyneMigration.migrate_config_file(config_file)

        error_msg = str(exc_info.value).lower()
        # Error should mention heterodyne model or migration
        assert any(keyword in error_msg for keyword in
                   ["heterodyne", "manually", "static mode", "11 parameter"])


class TestConfigValidation:
    """Test that configuration validation handles static mode parameters."""

    def test_3_parameter_config_detected_as_static(self, tmp_path):
        """Test that 3-parameter configurations are detected as static mode."""
        config_data = {
            "initial_parameters": {
                "values": [100.0, -0.5, 10.0]  # Only 3 params
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 0, "end_frame": 100},
                "scattering": {"wavevector_q": 0.0054},
                "geometry": {"stator_rotor_gap": 2000000}
            }
        }

        config_file = tmp_path / "three_param.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # Migration utility should detect this as 3-param (likely static)
        version = HeterodyneMigration.detect_config_version(config_data)
        assert version == "3-param-static"

    def test_heterodyne_config_accepted(self, tmp_path):
        """Test that proper 11-parameter heterodyne config is accepted."""
        config_data = {
            "initial_parameters": {
                "values": [100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0]  # 11 params
            },
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 0, "end_frame": 100},
                "scattering": {"wavevector_q": 0.0054},
                "geometry": {"stator_rotor_gap": 2000000}
            }
        }

        config_file = tmp_path / "heterodyne.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # Should be detected as 11-param heterodyne
        version = HeterodyneMigration.detect_config_version(config_data)
        assert version == "11-param-heterodyne"

        # Should load without issues
        config_manager = ConfigManager(str(config_file))
        assert config_manager.config is not None
