"""
Migration utilities for heterodyne model upgrade.

Provides tools for migrating from legacy configurations:
- 7-parameter homodyne/laminar flow → 14-parameter heterodyne
- 11-parameter heterodyne (old) → 14-parameter heterodyne (new)
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HeterodyneMigration:
    """Migration utility for upgrading to heterodyne model."""

    @staticmethod
    def detect_config_version(config: dict[str, Any]) -> str:
        """
        Detect the version/type of configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary

        Returns
        -------
        str
            One of: '3-param-static', '7-param-laminar', '11-param-heterodyne',
            '14-param-heterodyne', 'unknown'
        """
        # Check for static mode markers
        analysis_settings = config.get("analysis_settings", {})
        if analysis_settings.get("static_mode"):
            return "3-param-static"

        # Check parameter count
        initial_params = config.get("initial_parameters", {})
        param_values = initial_params.get("values", [])

        if len(param_values) == 3:
            return "3-param-static"
        elif len(param_values) == 7:
            return "7-param-laminar"
        elif len(param_values) == 11:
            return "11-param-heterodyne"
        elif len(param_values) == 14:
            return "14-param-heterodyne"

        return "unknown"

    @staticmethod
    def migrate_7_to_11_parameters(params_7: list[float]) -> list[float]:
        """
        Migrate 7-parameter laminar flow config to 11-parameter heterodyne.

        Old 7-parameter model (laminar flow):
        [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

        New 11-parameter model (heterodyne):
        [D0, alpha, D_offset, v0, beta, v_offset, f0, f1, f2, f3, phi0]

        Parameters
        ----------
        params_7 : list[float]
            7 legacy parameters

        Returns
        -------
        list[float]
            11 heterodyne parameters
        """
        if len(params_7) != 7:
            raise ValueError(f"Expected 7 parameters, got {len(params_7)}")

        # Extract legacy parameters
        D0 = params_7[0]
        alpha = params_7[1]
        D_offset = params_7[2]
        gamma_dot_t0 = params_7[3]  # Legacy velocity-like parameter
        beta = params_7[4]
        gamma_dot_t_offset = params_7[5]
        phi0 = params_7[6]

        # Map to new heterodyne parameters
        # Velocity parameters: use gamma_dot values as starting point
        v0 = gamma_dot_t0 * 10  # Scale factor (adjust based on physics)
        v_offset = gamma_dot_t_offset * 10

        # Fraction parameters: initialize with reasonable defaults
        # f(t) = f0 * exp(f1 * (t - f2)) + f3
        f0 = 0.5    # 50% amplitude
        f1 = 0.0    # No exponential decay initially
        f2 = 50.0   # Mid-point time offset
        f3 = 0.3    # 30% baseline fraction

        return [D0, alpha, D_offset, v0, beta, v_offset, f0, f1, f2, f3, phi0]

    @staticmethod
    def migrate_11_to_14_parameters(params_11: list[float]) -> list[float]:
        """
        Migrate 11-parameter heterodyne config to 14-parameter heterodyne.

        Old 11-parameter model (single g1):
        [D0, alpha, D_offset, v0, beta, v_offset, f0, f1, f2, f3, phi0]

        New 14-parameter model (separate g1_ref and g1_sample):
        [D0_ref, alpha_ref, D_offset_ref, D0_sample, alpha_sample, D_offset_sample,
         v0, beta, v_offset, f0, f1, f2, f3, phi0]

        Parameters
        ----------
        params_11 : list[float]
            11 legacy heterodyne parameters

        Returns
        -------
        list[float]
            14 heterodyne parameters
        """
        if len(params_11) != 11:
            raise ValueError(f"Expected 11 parameters, got {len(params_11)}")

        # Extract legacy parameters
        D0, alpha, D_offset = params_11[0:3]
        v0, beta, v_offset = params_11[3:6]
        f0, f1, f2, f3 = params_11[6:10]
        phi0 = params_11[10]

        # For backward compatibility, initialize sample parameters equal to reference
        # This ensures g1_ref = g1_sample initially (same behavior as old model)
        # During optimization, they will diverge as needed
        D0_ref = D0
        alpha_ref = alpha
        D_offset_ref = D_offset
        D0_sample = D0
        alpha_sample = alpha
        D_offset_sample = D_offset

        return [
            D0_ref, alpha_ref, D_offset_ref,
            D0_sample, alpha_sample, D_offset_sample,
            v0, beta, v_offset,
            f0, f1, f2, f3,
            phi0
        ]

    @staticmethod
    def migrate_config_file(
        input_path: str | Path, output_path: str | Path | None = None
    ) -> dict[str, Any]:
        """
        Migrate a configuration file to heterodyne format.

        Parameters
        ----------
        input_path : str or Path
            Path to legacy configuration file
        output_path : str or Path, optional
            Path to save migrated config. If None, returns dict only.

        Returns
        -------
        dict
            Migrated configuration
        """
        input_path = Path(input_path)

        # Load legacy config
        with open(input_path, 'r') as f:
            legacy_config = json.load(f)

        # Detect version
        version = HeterodyneMigration.detect_config_version(legacy_config)
        logger.info(f"Detected config version: {version}")

        # Create migrated config
        migrated_config = legacy_config.copy()

        # Remove static mode settings if present
        if "analysis_settings" in migrated_config:
            analysis_settings = migrated_config["analysis_settings"]
            if "static_mode" in analysis_settings:
                del analysis_settings["static_mode"]
                logger.info("Removed static_mode setting")
            if "static_submode" in analysis_settings:
                del analysis_settings["static_submode"]
                logger.info("Removed static_submode setting")

        # Migrate parameters based on version
        if version == "7-param-laminar":
            old_params = legacy_config["initial_parameters"]["values"]
            # Migrate 7→11 first
            params_11 = HeterodyneMigration.migrate_7_to_11_parameters(old_params)
            # Then migrate 11→14
            new_params = HeterodyneMigration.migrate_11_to_14_parameters(params_11)

            migrated_config["initial_parameters"]["values"] = new_params
            migrated_config["initial_parameters"]["parameter_names"] = [
                "D0_ref", "alpha_ref", "D_offset_ref",
                "D0_sample", "alpha_sample", "D_offset_sample",
                "v0", "beta", "v_offset",
                "f0", "f1", "f2", "f3",
                "phi0"
            ]

            logger.info(f"Migrated parameters from 7 to 14: {old_params} -> {new_params}")

        elif version == "11-param-heterodyne":
            old_params = legacy_config["initial_parameters"]["values"]
            new_params = HeterodyneMigration.migrate_11_to_14_parameters(old_params)

            migrated_config["initial_parameters"]["values"] = new_params
            migrated_config["initial_parameters"]["parameter_names"] = [
                "D0_ref", "alpha_ref", "D_offset_ref",
                "D0_sample", "alpha_sample", "D_offset_sample",
                "v0", "beta", "v_offset",
                "f0", "f1", "f2", "f3",
                "phi0"
            ]

            logger.info(f"Migrated parameters from 11 to 14: {old_params} -> {new_params}")

        elif version == "14-param-heterodyne":
            logger.info("Config is already 14-parameter heterodyne, no migration needed")

        elif version == "3-param-static":
            raise ValueError(
                "Cannot automatically migrate 3-parameter static configs. "
                "Static mode has been removed. Please configure for heterodyne model "
                "with 14 parameters manually."
            )

        # Add migration metadata
        migrated_config["migration_info"] = {
            "source_version": version,
            "target_version": "14-param-heterodyne",
            "source_file": str(input_path),
            "migration_note": (
                "Migrated from legacy model to heterodyne. "
                "Fraction parameters (f0-f3) use default values and may need tuning."
            )
        }

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                json.dump(migrated_config, f, indent=2)
            logger.info(f"Saved migrated config to {output_path}")

        return migrated_config

    @staticmethod
    def generate_migration_guide(config_path: str | Path) -> str:
        """
        Generate human-readable migration guide for a config file.

        Parameters
        ----------
        config_path : str or Path
            Path to configuration file

        Returns
        -------
        str
            Migration guide text
        """
        config_path = Path(config_path)

        with open(config_path, 'r') as f:
            config = json.load(f)

        version = HeterodyneMigration.detect_config_version(config)

        guide = f"""
HETERODYNE MODEL MIGRATION GUIDE
=================================

Configuration File: {config_path}
Detected Version: {version}

"""

        if version == "11-param-heterodyne":
            guide += "✅ Already using 11-parameter heterodyne model. No migration needed.\n"

        elif version == "7-param-laminar":
            params = config["initial_parameters"]["values"]
            new_params = HeterodyneMigration.migrate_7_to_11_parameters(params)

            guide += f"""
Migration Required: 7-parameter → 11-parameter heterodyne

OLD PARAMETERS (7):
  [0] D0 = {params[0]}
  [1] alpha = {params[1]}
  [2] D_offset = {params[2]}
  [3] gamma_dot_t0 = {params[3]}
  [4] beta = {params[4]}
  [5] gamma_dot_t_offset = {params[5]}
  [6] phi0 = {params[6]}

NEW PARAMETERS (11):
  Diffusion (3):
    [0] D0 = {new_params[0]} (unchanged)
    [1] alpha = {new_params[1]} (unchanged)
    [2] D_offset = {new_params[2]} (unchanged)

  Velocity (3):
    [3] v0 = {new_params[3]} (derived from gamma_dot_t0)
    [4] beta = {new_params[4]} (unchanged)
    [5] v_offset = {new_params[5]} (derived from gamma_dot_t_offset)

  Fraction (4):
    [6] f0 = {new_params[6]} (default: 0.5)
    [7] f1 = {new_params[7]} (default: 0.0)
    [8] f2 = {new_params[8]} (default: 50.0)
    [9] f3 = {new_params[9]} (default: 0.3)

  Flow angle (1):
    [10] phi0 = {new_params[10]} (unchanged)

⚠️  IMPORTANT: Fraction parameters (f0-f3) are initialized with defaults.
   You should tune these based on your experimental data.

To migrate automatically:
  python -m heterodyne.core.migration {config_path} output.json
"""

        elif version == "3-param-static":
            guide += """
Migration Required: 3-parameter static → 11-parameter heterodyne

❌ Automatic migration not supported for static mode configs.

Static mode has been removed from the package. You must create a new
11-parameter heterodyne configuration manually.

NEW 11-PARAMETER MODEL:
  Diffusion (3): D0, alpha, D_offset
  Velocity (3): v0, beta, v_offset
  Fraction (4): f0, f1, f2, f3
  Flow angle (1): phi0

See example configs in heterodyne/config/templates/
"""

        else:
            guide += f"""
Unknown configuration format.

Cannot determine migration path. Please create a new 11-parameter
heterodyne configuration manually.
"""

        return guide


def main():
    """CLI entry point for migration utility."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate legacy configs to heterodyne model"
    )
    parser.add_argument("input", help="Input configuration file")
    parser.add_argument("output", nargs="?", help="Output configuration file")
    parser.add_argument(
        "--guide", action="store_true", help="Show migration guide only"
    )

    args = parser.parse_args()

    if args.guide:
        guide = HeterodyneMigration.generate_migration_guide(args.input)
        print(guide)
    else:
        if not args.output:
            print("Error: output file required for migration")
            print("Use --guide to see migration guide without migrating")
            return 1

        try:
            migrated = HeterodyneMigration.migrate_config_file(
                args.input, args.output
            )
            print(f"✅ Successfully migrated {args.input} -> {args.output}")
            print("\nMigration summary:")
            print(json.dumps(migrated.get("migration_info", {}), indent=2))
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
