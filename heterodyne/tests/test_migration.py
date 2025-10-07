"""
Unit tests for heterodyne migration utilities.

Tests migration from legacy 7-parameter laminar flow model to
14-parameter heterodyne model.
"""

import json
import pytest
from pathlib import Path
from heterodyne.core.migration import HeterodyneMigration


class TestConfigVersionDetection:
    """Test configuration version detection."""

    def test_detect_11_param_heterodyne(self):
        """Test detection of 11-parameter heterodyne config."""
        config = {
            "initial_parameters": {
                "values": [100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0]
            }
        }
        version = HeterodyneMigration.detect_config_version(config)
        assert version == "11-param-heterodyne"

    def test_detect_7_param_laminar(self):
        """Test detection of 7-parameter laminar flow config."""
        config = {
            "initial_parameters": {
                "values": [1324.1, -0.014, -0.674, 0.003, -0.909, 0.0, 0.0]
            }
        }
        version = HeterodyneMigration.detect_config_version(config)
        assert version == "7-param-laminar"

    def test_detect_3_param_static(self):
        """Test detection of 3-parameter static config."""
        config = {
            "analysis_settings": {"static_mode": True},
            "initial_parameters": {"values": [100.0, 0.0, 10.0]}
        }
        version = HeterodyneMigration.detect_config_version(config)
        assert version == "3-param-static"

    def test_detect_unknown_config(self):
        """Test detection of unknown config format."""
        config = {"initial_parameters": {"values": [1, 2, 3, 4, 5]}}
        version = HeterodyneMigration.detect_config_version(config)
        assert version == "unknown"


class TestParameterMigration:
    """Test parameter migration from 7 to 11."""

    def test_migrate_7_to_11_parameters(self):
        """Test migration of 7 parameters to 11."""
        legacy_params = [1324.1, -0.014, -0.674, 0.003, -0.909, 0.0, 0.0]
        new_params = HeterodyneMigration.migrate_7_to_11_parameters(legacy_params)

        # Should have 11 parameters
        assert len(new_params) == 11

        # Diffusion parameters unchanged
        assert new_params[0] == legacy_params[0]  # D0
        assert new_params[1] == legacy_params[1]  # alpha
        assert new_params[2] == legacy_params[2]  # D_offset

        # Velocity parameters derived
        assert new_params[3] == legacy_params[3] * 10  # v0 from gamma_dot_t0
        assert new_params[4] == legacy_params[4]        # beta unchanged
        assert new_params[5] == legacy_params[5] * 10  # v_offset from gamma_dot_t_offset

        # Fraction parameters use defaults
        assert new_params[6] == 0.5   # f0
        assert new_params[7] == 0.0   # f1
        assert new_params[8] == 50.0  # f2
        assert new_params[9] == 0.3   # f3

        # Flow angle unchanged
        assert new_params[10] == legacy_params[6]  # phi0

    def test_migrate_invalid_parameter_count(self):
        """Test that migration rejects invalid parameter counts."""
        with pytest.raises(ValueError, match="Expected 7 parameters"):
            HeterodyneMigration.migrate_7_to_11_parameters([1, 2, 3])


class TestConfigFileMigration:
    """Test full configuration file migration."""

    def test_migrate_7_param_config_file(self, tmp_path):
        """Test migration of 7-parameter config file."""
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
                "temporal": {"dt": 0.1},
                "scattering": {"wavevector_q": 0.0054}
            }
        }

        input_file = tmp_path / "legacy_config.json"
        with open(input_file, 'w') as f:
            json.dump(legacy_config, f)

        # Migrate
        output_file = tmp_path / "migrated_config.json"
        migrated = HeterodyneMigration.migrate_config_file(input_file, output_file)

        # Verify migration to 14 parameters
        assert migrated["initial_parameters"]["values"] == [
            1324.1, -0.014, -0.674,  # D_ref params
            1324.1, -0.014, -0.674,  # D_sample params (initially equal to ref)
            0.03, -0.909, 0.0,       # v params (0.003*10, beta, 0.0*10)
            0.5, 0.0, 50.0, 0.3,     # f params (defaults)
            0.0                       # phi0
        ]

        assert migrated["initial_parameters"]["parameter_names"] == [
            "D0_ref", "alpha_ref", "D_offset_ref",
            "D0_sample", "alpha_sample", "D_offset_sample",
            "v0", "beta", "v_offset",
            "f0", "f1", "f2", "f3",
            "phi0"
        ]

        # Migration metadata should be added
        assert "migration_info" in migrated
        assert migrated["migration_info"]["source_version"] == "7-param-laminar"
        assert migrated["migration_info"]["target_version"] == "14-param-heterodyne"

        # Output file should exist
        assert output_file.exists()

        # Verify saved file matches
        with open(output_file, 'r') as f:
            saved = json.load(f)
        assert saved == migrated

    def test_migrate_removes_static_mode_settings(self, tmp_path):
        """Test that migration removes static mode settings."""
        legacy_config = {
            "analysis_settings": {
                "static_mode": True,
                "static_submode": "isotropic",
                "other_setting": "value"
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0]
            }
        }

        input_file = tmp_path / "static_config.json"
        with open(input_file, 'w') as f:
            json.dump(legacy_config, f)

        # Should raise error for 3-param static
        with pytest.raises(ValueError, match="Cannot automatically migrate"):
            HeterodyneMigration.migrate_config_file(input_file)

    def test_migrate_11_param_config_to_14(self, tmp_path):
        """Test that 11-param configs are migrated to 14-parameter."""
        heterodyne_config = {
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

        input_file = tmp_path / "heterodyne_config.json"
        with open(input_file, 'w') as f:
            json.dump(heterodyne_config, f)

        # Migrate to 14 parameters
        migrated = HeterodyneMigration.migrate_config_file(input_file)

        # Should expand to 14 parameters with sample=reference
        assert migrated["initial_parameters"]["values"] == [
            100.0, -0.5, 10.0,  # D_ref
            100.0, -0.5, 10.0,  # D_sample (initially = ref)
            0.1, 0.0, 0.01,     # velocity
            0.5, 0.0, 50.0, 0.3,  # fraction
            0.0                  # phi0
        ]

        assert migrated["initial_parameters"]["parameter_names"] == [
            "D0_ref", "alpha_ref", "D_offset_ref",
            "D0_sample", "alpha_sample", "D_offset_sample",
            "v0", "beta", "v_offset",
            "f0", "f1", "f2", "f3",
            "phi0"
        ]


class TestMigrationGuide:
    """Test migration guide generation."""

    def test_generate_guide_for_7_param_config(self, tmp_path):
        """Test migration guide for 7-parameter config."""
        config = {
            "initial_parameters": {
                "values": [1324.1, -0.014, -0.674, 0.003, -0.909, 0.0, 0.0]
            }
        }

        config_file = tmp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        guide = HeterodyneMigration.generate_migration_guide(config_file)

        # Guide should mention migration
        assert "Migration Required" in guide
        assert "7-parameter → 14-parameter" in guide
        assert "OLD PARAMETERS (7)" in guide
        assert "NEW PARAMETERS (14)" in guide

        # Should show parameter mapping
        assert "D0" in guide
        assert "v0" in guide
        assert "f0" in guide
        assert "phi0" in guide

    def test_generate_guide_for_11_param_config(self, tmp_path):
        """Test migration guide for 11-param config (needs migration to 14)."""
        config = {
            "initial_parameters": {
                "values": [100.0, -0.5, 10.0, 0.1, 0.0, 0.01,
                          0.5, 0.0, 50.0, 0.3, 0.0]
            }
        }

        config_file = tmp_path / "heterodyne_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        guide = HeterodyneMigration.generate_migration_guide(config_file)

        # Should indicate migration to 14 params
        assert "Migration Required" in guide or "11-parameter → 14-parameter" in guide

    def test_generate_guide_for_static_config(self, tmp_path):
        """Test migration guide for static config."""
        config = {
            "analysis_settings": {"static_mode": True},
            "initial_parameters": {"values": [100.0, 0.0, 10.0]}
        }

        config_file = tmp_path / "static_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        guide = HeterodyneMigration.generate_migration_guide(config_file)

        # Should indicate manual migration needed
        assert "3-parameter static" in guide
        assert "Automatic migration not supported" in guide or "Static mode has been removed" in guide
