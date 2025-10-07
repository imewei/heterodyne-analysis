"""
Configuration Creator for Heterodyne Analysis
==========================================

Interactive configuration file generator for XPCS heterodyne analysis workflows.
Creates customized JSON configuration files from the 14-parameter heterodyne template,
enabling quick setup of analysis parameters for experimental scenarios.

Key Features:
- 14-parameter 2-component heterodyne model (2 shear bands)
- Customizable sample and experiment metadata
- Automatic path structure generation
- Validation and guidance for next steps
- Support for optimization methods

Analysis Model:
- 2-component heterodyne scattering analysis
- 14 parameters: separate reference/sample transport (6), velocity (3), fraction (4), flow angle (1)
- Supports two-shear-band systems

Usage Scenarios:
- New experiment setup for heterodyne analysis
- Batch analysis preparation with consistent naming
- Quick configuration generation
- Template customization for specific experimental conditions

Generated Configuration Includes:
- 14-parameter heterodyne physics model
- Data loading paths and file specifications
- Optimization method settings and hyperparameters
- Output formatting and result organization
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from ..config import get_template_path

# Import advanced shell completion functionality
try:
    from ..ui.completion.adapter import setup_shell_completion

    COMPLETION_AVAILABLE = True
    COMPLETION_SYSTEM = "advanced"
except ImportError:
    COMPLETION_AVAILABLE = False
    COMPLETION_SYSTEM = "none"

    def setup_shell_completion(parser: "argparse.ArgumentParser") -> None:
        """Fallback when completion is not available."""


def _remove_mcmc_sections(config):
    """
    Remove MCMC sections from configuration for clean generation.

    This function removes deprecated MCMC sections from configuration
    dictionaries to ensure that newly generated configurations are clean
    and don't contain deprecated sections.
    """
    if not isinstance(config, dict):
        return config

    # Remove top-level MCMC sections
    mcmc_sections_to_remove = []
    for key in config.keys():
        if key.startswith("mcmc_"):
            mcmc_sections_to_remove.append(key)

    # Create clean configuration
    clean_config = {}
    for key, value in config.items():
        if key not in mcmc_sections_to_remove:
            if key == "optimization_config" and isinstance(value, dict):
                # Clean optimization_config of MCMC subsections
                clean_opt_config = {}
                for opt_key, opt_value in value.items():
                    if not opt_key.startswith("mcmc_"):
                        clean_opt_config[opt_key] = opt_value
                clean_config[key] = clean_opt_config
            elif key == "workflow_integration" and isinstance(value, dict):
                # Clean workflow_integration of MCMC subsections
                clean_workflow_config = {}
                for workflow_key, workflow_value in value.items():
                    if not workflow_key.startswith("mcmc_"):
                        clean_workflow_config[workflow_key] = workflow_value
                clean_config[key] = clean_workflow_config
            elif key == "validation_rules" and isinstance(value, dict):
                # Clean validation_rules of MCMC subsections
                clean_validation_config = {}
                for val_key, val_value in value.items():
                    if not val_key.startswith("mcmc_"):
                        clean_validation_config[val_key] = val_value
                clean_config[key] = clean_validation_config
            elif key == "output_settings" and isinstance(value, dict):
                # Clean output_settings of MCMC plotting references
                clean_output_config = {}
                for out_key, out_value in value.items():
                    if out_key == "plotting" and isinstance(out_value, dict):
                        clean_plotting_config = {}
                        for plot_key, plot_value in out_value.items():
                            if not plot_key.startswith("mcmc_"):
                                clean_plotting_config[plot_key] = plot_value
                        clean_output_config[out_key] = clean_plotting_config
                    else:
                        clean_output_config[out_key] = out_value
                clean_config[key] = clean_output_config
            else:
                clean_config[key] = value

    return clean_config


def create_config_from_template(
    output_file="my_config.json",
    sample_name=None,
    experiment_name=None,
    author=None,
    mode="heterodyne",
):
    """
    Generate customized configuration file from heterodyne template.

    Creates a complete configuration file by loading the 14-parameter heterodyne template,
    applying user customizations, and generating appropriate file paths
    and metadata. Removes template-specific fields to create a clean
    production configuration.

    Customization Process:
    - Load 14-parameter heterodyne template
    - Apply user-specified metadata (author, experiment, sample)
    - Generate appropriate data paths based on sample name
    - Set creation/update timestamps
    - Remove template metadata for clean output
    - Provide usage guidance

    Parameters
    ----------
    output_file : str
        Output configuration filename (default: "my_config.json")
    sample_name : str, optional
        Sample identifier for automatic path generation
    experiment_name : str, optional
        Descriptive experiment name for metadata
    author : str, optional
        Author name for configuration attribution
    mode : str
        Analysis mode (always "heterodyne" for 14-parameter 2-component model)

    Raises
    ------
    FileNotFoundError
        Template file not found in expected location
    JSONDecodeError
        Template file contains invalid JSON
    OSError
        File system errors during creation
    ValueError
        Invalid analysis mode specified
    """

    # Validate mode and get template path using config module
    valid_modes = ["heterodyne"]

    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Only 'heterodyne' mode is supported.")

    # Get template path using the config module
    try:
        template_file = get_template_path(mode)
    except ValueError:
        # Fallback to template if mode not found
        print(f"Warning: Mode-specific template not found for '{mode}'")
        print("Falling back to master template...")
        template_file = get_template_path("template")

    if not template_file.exists():
        raise FileNotFoundError(f"Template not found: {template_file}")

    # Load template
    with open(template_file, encoding="utf-8") as f:
        config = json.load(f)

    # Remove template-specific fields from final config
    if "_template_info" in config:
        del config["_template_info"]

    # Remove deprecated MCMC sections from generated configuration
    # This ensures new configurations don't contain deprecated sections
    config = _remove_mcmc_sections(config)

    # Apply customizations
    current_date = datetime.now().strftime("%Y-%m-%d")

    if "metadata" in config:
        config["metadata"]["created_date"] = current_date
        config["metadata"]["updated_date"] = current_date

        # Update analysis mode in metadata
        config["metadata"]["analysis_mode"] = mode

        if experiment_name:
            config["metadata"]["description"] = experiment_name
        elif "description" in config["metadata"]:
            # Set heterodyne description
            config["metadata"]["description"] = "2-Component Heterodyne Scattering Analysis - 14-parameter model with two shear bands"

        if author:
            config["metadata"]["authors"] = [author]

    # Apply sample-specific customizations
    if sample_name and "experimental_data" in config:
        config["experimental_data"]["data_folder_path"] = f"./data/{sample_name}/"
        if "cache_file_path" in config["experimental_data"]:
            config["experimental_data"]["cache_file_path"] = f"./data/{sample_name}/"

        # Update cache filename template for heterodyne
        config["experimental_data"]["cache_filename_template"] = f"cached_c2_heterodyne_{sample_name}_{{start_frame}}_{{end_frame}}.npz"

    # Save configuration
    output_path = Path(output_file)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✓ Configuration created: {output_path.absolute()}")
    print(f"✓ Analysis mode: 2-component heterodyne (14 parameters)")

    # Print heterodyne model information
    print("  • 14-parameter 2-component heterodyne scattering model")
    print("  • Parameters: Reference transport (3), Sample transport (3), Velocity (3), Fraction (4), Flow angle (1)")
    print("  • Supports two-shear-band systems with separate g₁_ref and g₁_sample")

    # Provide next steps
    print("\nNext steps:")
    print(f"1. Edit {output_path} and customize the parameters for your experiment")
    print("2. Replace placeholder values (YOUR_*) with actual values")
    print("3. Adjust initial_parameters.values for all 14 parameters based on your system")
    print("4. Ensure phi_angles_file exists and contains your scattering angles")
    print(f"5. Run analysis with: heterodyne --config {output_path}")

    print("\nAvailable methods:")
    print("  --method classical  # Nelder-Mead and Gurobi optimization")
    print(
        "  --method robust     # Wasserstein, scenario, and ellipsoidal robust methods"
    )
    print("  --method all        # All available methods (classical + robust)")
    print("\nDocumentation: See CLAUDE.md for heterodyne model details")


def main():
    """Command-line interface for config creation."""
    # Check Python version requirement
    parser = argparse.ArgumentParser(
        description="Create heterodyne analysis configuration from 14-parameter template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Model:
  heterodyne - 14-parameter 2-component heterodyne scattering model
               Supports two-shear-band systems with separate reference/sample transport

Parameters (14 total):
  - Reference transport: D0_ref, alpha_ref, D_offset_ref
  - Sample transport: D0_sample, alpha_sample, D_offset_sample
  - Velocity: v0, beta, v_offset
  - Fraction: f0, f1, f2, f3
  - Flow angle: phi0

Examples:
  # Create heterodyne configuration
  heterodyne-config --output my_config.json

  # Create configuration with sample name
  heterodyne-config --sample protein_01

  # Create configuration with full metadata
  heterodyne-config --sample collagen \
                    --author "Your Name" \
                    --experiment "Collagen two-component analysis"
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        default="my_config.json",
        help="Output configuration file name (default: my_config.json)",
    )

    parser.add_argument("--sample", "-s", help="Sample name (used in data paths)")

    parser.add_argument("--experiment", "-e", help="Experiment description")

    parser.add_argument("--author", "-a", help="Author name")

    # Setup shell completion if available
    if COMPLETION_AVAILABLE:
        setup_shell_completion(parser)

    args = parser.parse_args()

    try:
        create_config_from_template(
            output_file=args.output,
            sample_name=args.sample,
            experiment_name=args.experiment,
            author=args.author,
            # mode defaults to "heterodyne" (only supported mode)
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
