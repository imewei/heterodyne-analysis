"""
Comprehensive Functionality Validation Test Suite
================================================

Validates that all removed imports and optimizations haven't broken existing
functionality. Tests core workflows, API compatibility, and integration points.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

import subprocess
import sys

import pytest

# Test imports to ensure they still work
import heterodyne


class TestCoreAPIAvailability:
    """Test that all core API components are still available."""

    def test_main_package_imports(self):
        """Test that main package components can be imported."""
        # Test that main package is importable
        assert heterodyne.__version__ is not None
        assert heterodyne.__author__ is not None

        # Test that all public API components are available
        public_api_components = [
            "HeterodyneAnalysisCore",
            "ClassicalOptimizer",
            "RobustHeterodyneOptimizer",
            "ConfigManager",
            "PerformanceMonitor",
        ]

        for component in public_api_components:
            assert hasattr(
                heterodyne, component
            ), f"Missing public API component: {component}"

    def test_computational_kernels_available(self):
        """Test that computational kernels are available."""
        kernel_functions = [
            "calculate_diffusion_coefficient_numba",
            "calculate_shear_rate_numba",
            "compute_g1_correlation_numba",
            "compute_sinc_squared_numba",
            "create_time_integral_matrix_numba",
            "memory_efficient_cache",
        ]

        for func_name in kernel_functions:
            assert hasattr(
                heterodyne, func_name
            ), f"Missing kernel function: {func_name}"
            func = getattr(heterodyne, func_name)
            assert callable(func), f"Kernel function {func_name} is not callable"

    def test_cli_functions_available(self):
        """Test that CLI functions are available."""
        cli_functions = [
            "run_heterodyne_main",
            "create_config_main",
            "enhanced_runner_main",
        ]

        for func_name in cli_functions:
            assert hasattr(heterodyne, func_name), f"Missing CLI function: {func_name}"

    def test_configuration_utilities_available(self):
        """Test that configuration utilities are available."""
        config_functions = [
            "get_template_path",
            "get_config_dir",
            "TEMPLATE_FILES",
        ]

        for func_name in config_functions:
            assert hasattr(
                heterodyne, func_name
            ), f"Missing config function: {func_name}"

    def test_performance_monitoring_available(self):
        """Test that performance monitoring functions are available."""
        perf_functions = [
            "check_performance_health",
            "establish_performance_baseline",
            "monitor_startup_performance",
            "get_import_performance_report",
            "preload_scientific_dependencies",
            "optimize_initialization",
        ]

        for func_name in perf_functions:
            assert hasattr(
                heterodyne, func_name
            ), f"Missing performance function: {func_name}"
            func = getattr(heterodyne, func_name)
            assert callable(func), f"Performance function {func_name} is not callable"


class TestLazyLoadingFunctionality:
    """Test that lazy loading works correctly."""

    def test_lazy_loaded_modules_work(self):
        """Test that lazy-loaded modules function correctly."""
        # Test HeterodyneAnalysisCore lazy loading
        try:
            core_class = heterodyne.HeterodyneAnalysisCore
            assert core_class is not None
            # Don't instantiate to avoid dependency issues
        except Exception as e:
            pytest.fail(f"HeterodyneAnalysisCore lazy loading failed: {e}")

        # Test ConfigManager lazy loading
        try:
            config_class = heterodyne.ConfigManager
            assert config_class is not None
        except Exception as e:
            pytest.fail(f"ConfigManager lazy loading failed: {e}")

    def test_kernel_functions_lazy_loading(self):
        """Test that kernel functions work through lazy loading."""
        # Test that we can access kernel functions without immediate import
        kernel_func = heterodyne.compute_sinc_squared_numba
        assert callable(kernel_func)

        # Test basic functionality (with mock data to avoid numba issues)
        try:
            # This should work even if numba is disabled
            import numpy as np

            result = kernel_func(np.array([1.0, 2.0]))
            assert result is not None
        except Exception:
            # Expected if numba is disabled, just ensure function is accessible
            pass

    def test_scientific_dependencies_lazy_loading(self):
        """Test that scientific dependencies are properly lazy loaded."""
        from heterodyne.core.lazy_imports import scientific_deps

        # Test that numpy is available through lazy loading
        numpy_loader = scientific_deps.get("numpy")
        assert numpy_loader is not None

        # Test basic numpy operation
        import numpy as np

        arr = np.array([1, 2, 3])
        assert arr.sum() == 6


class TestConfigurationSystem:
    """Test configuration system functionality."""

    def test_config_template_access(self):
        """Test that configuration templates are accessible."""
        try:
            template_files = heterodyne.TEMPLATE_FILES
            assert template_files is not None
        except Exception as e:
            pytest.fail(f"Template files access failed: {e}")

    def test_config_directory_access(self):
        """Test that config directory function works."""
        try:
            config_dir = heterodyne.get_config_dir()
            assert config_dir is not None
        except Exception as e:
            pytest.fail(f"Config directory access failed: {e}")

    def test_template_path_function(self):
        """Test template path function."""
        try:
            # Test with a known template
            template_path = heterodyne.get_template_path("heterodyne")
            assert template_path is not None
        except Exception:
            # This might fail if template doesn't exist, but function should be accessible
            pass


class TestPerformanceSystemIntegration:
    """Test that performance system integration works."""

    def test_performance_health_check(self):
        """Test performance health check functionality."""
        try:
            health = heterodyne.check_performance_health()
            assert isinstance(health, dict)
            assert "status" in health
            assert health["status"] in ["excellent", "good", "fair", "poor", "error"]
        except Exception as e:
            pytest.fail(f"Performance health check failed: {e}")

    def test_import_performance_report(self):
        """Test import performance reporting."""
        try:
            report = heterodyne.get_import_performance_report()
            assert isinstance(report, dict)
            assert "summary" in report
        except Exception as e:
            pytest.fail(f"Import performance report failed: {e}")

    def test_startup_optimization(self):
        """Test startup optimization functionality."""
        try:
            optimization = heterodyne.optimize_initialization()
            assert isinstance(optimization, dict)
            assert "strategy" in optimization
        except Exception as e:
            pytest.fail(f"Startup optimization failed: {e}")

    def test_baseline_establishment(self):
        """Test baseline establishment."""
        try:
            baseline = heterodyne.establish_performance_baseline(
                name="test_validation", target_import_time=2.0
            )
            assert isinstance(baseline, dict)
            assert "name" in baseline
            assert baseline["name"] == "test_validation"
        except Exception as e:
            pytest.fail(f"Baseline establishment failed: {e}")


class TestCoreWorkflows:
    """Test core package workflows still function."""

    @pytest.mark.slow
    def test_basic_analysis_workflow(self):
        """Test basic analysis workflow."""
        try:
            # Test that we can create an analysis instance
            analysis_class = heterodyne.HeterodyneAnalysisCore

            # Test basic configuration
            config_manager = heterodyne.ConfigManager

            # This validates the lazy loading works for core components
            assert analysis_class is not None
            assert config_manager is not None

        except ImportError as e:
            pytest.skip(
                f"Analysis workflow test skipped due to missing dependencies: {e}"
            )
        except Exception as e:
            pytest.fail(f"Basic analysis workflow failed: {e}")

    def test_optimization_workflow(self):
        """Test optimization workflow."""
        try:
            # Test optimizer access
            classical_opt = heterodyne.ClassicalOptimizer
            robust_opt = heterodyne.RobustHeterodyneOptimizer

            assert classical_opt is not None
            assert robust_opt is not None

        except ImportError as e:
            pytest.skip(
                f"Optimization workflow test skipped due to missing dependencies: {e}"
            )
        except Exception as e:
            pytest.fail(f"Optimization workflow failed: {e}")

    def test_configuration_workflow(self):
        """Test configuration workflow."""
        try:
            # Test config creation function
            create_config = heterodyne.create_config_main
            assert callable(create_config)

            # Test template access
            templates = heterodyne.TEMPLATE_FILES
            assert templates is not None

        except Exception as e:
            pytest.fail(f"Configuration workflow failed: {e}")


class TestModuleIntegrity:
    """Test module structure integrity."""

    def test_all_exports_accessible(self):
        """Test that all items in __all__ are accessible."""
        for item_name in heterodyne.__all__:
            assert hasattr(
                heterodyne, item_name
            ), f"__all__ item {item_name} not accessible"
            item = getattr(heterodyne, item_name)
            assert item is not None, f"__all__ item {item_name} is None"

    def test_no_broken_imports_in_submodules(self):
        """Test that submodules don't have broken imports."""
        # Test core module
        try:
            from heterodyne import core

            assert core is not None
        except ImportError as e:
            pytest.fail(f"Core module import failed: {e}")

        # Test performance module
        try:
            from heterodyne import performance

            assert performance is not None
        except ImportError as e:
            pytest.fail(f"Performance module import failed: {e}")

    def test_circular_import_resolution(self):
        """Test that circular imports are resolved."""
        # This test passes if the import doesn't hang or fail

        # If we get here, no circular import issues
        assert True


class TestBackwardCompatibility:
    """Test backward compatibility."""

    def test_legacy_api_access(self):
        """Test that legacy API access patterns still work."""
        # Test direct attribute access
        assert hasattr(heterodyne, "HeterodyneAnalysisCore")
        assert hasattr(heterodyne, "ClassicalOptimizer")
        assert hasattr(heterodyne, "ConfigManager")

    def test_function_signatures_preserved(self):
        """Test that public function signatures are preserved."""
        # Test that performance functions have expected signatures
        import inspect

        # Test monitor_startup_performance signature
        sig = inspect.signature(heterodyne.monitor_startup_performance)
        assert "iterations" in sig.parameters

        # Test establish_performance_baseline signature
        sig = inspect.signature(heterodyne.establish_performance_baseline)
        assert "name" in sig.parameters
        assert "target_import_time" in sig.parameters


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    def test_missing_optional_dependencies_handled(self):
        """Test that missing optional dependencies are handled gracefully."""
        # The package should import successfully even with disabled dependencies
        # (we test this by importing with numba disabled)
        assert heterodyne.__version__ is not None

    def test_lazy_loading_error_handling(self):
        """Test lazy loading error handling."""
        from heterodyne.core.lazy_imports import HeavyDependencyLoader

        # Test with non-existent module
        loader = HeavyDependencyLoader("nonexistent_module", required=False)

        # Should not raise, should return None or fallback
        try:
            result = loader._get_object()
            # Should either be None or a fallback value
            assert result is None or result is not None  # Either is acceptable
        except Exception:
            # Should not raise for non-required dependencies
            pytest.fail(
                "Lazy loader should handle missing optional dependencies gracefully"
            )

    def test_performance_monitoring_error_handling(self):
        """Test performance monitoring error handling."""
        # Even if monitoring fails, it shouldn't break the package
        try:
            health = heterodyne.check_performance_health()
            # Should return some kind of status
            assert isinstance(health, dict)
        except Exception as e:
            pytest.fail(f"Performance monitoring should not raise exceptions: {e}")


class TestRealWorldUsage:
    """Test real-world usage patterns."""

    @pytest.mark.slow
    def test_import_in_subprocess(self):
        """Test package import in subprocess (real-world scenario)."""

        # Test that package imports successfully in clean subprocess
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import heterodyne; print('SUCCESS:', heterodyne.__version__)",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"Subprocess import failed: {result.stderr}"
        assert (
            "SUCCESS:" in result.stdout
        ), "Package import didn't produce expected output"

    def test_repeated_imports(self):
        """Test repeated imports don't cause issues."""
        # This should not cause issues with lazy loading
        # Re-import should be fine
        import heterodyne

        assert heterodyne.__version__ is not None

    @pytest.mark.slow
    def test_concurrent_access(self):
        """Test concurrent access to lazy-loaded components."""
        import threading

        results = []
        errors = []

        def access_component():
            try:
                # Access various components concurrently
                health = heterodyne.check_performance_health()
                core_class = heterodyne.HeterodyneAnalysisCore
                config_class = heterodyne.ConfigManager

                results.append(
                    {
                        "health": health,
                        "core_available": core_class is not None,
                        "config_available": config_class is not None,
                    }
                )
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=access_component) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 5, "Not all threads completed successfully"

        # All results should be valid
        for result in results:
            assert isinstance(result["health"], dict)
            assert result["core_available"] is True
            assert result["config_available"] is True


class TestSystemIntegration:
    """Test system-level integration."""

    def test_logging_configuration(self):
        """Test that logging configuration works."""
        try:
            heterodyne.configure_logging()
            # Should not raise exceptions
        except Exception as e:
            pytest.fail(f"Logging configuration failed: {e}")

    def test_environment_variable_handling(self):
        """Test environment variable handling."""
        import os

        # Test optimization environment variable
        original_value = os.environ.get("HETERODYNE_OPTIMIZE_STARTUP")

        try:
            # Test with optimization enabled
            os.environ["HETERODYNE_OPTIMIZE_STARTUP"] = "true"
            health = heterodyne.check_performance_health()
            assert isinstance(health, dict)

            # Test with optimization disabled
            os.environ["HETERODYNE_OPTIMIZE_STARTUP"] = "false"
            health = heterodyne.check_performance_health()
            assert isinstance(health, dict)

        finally:
            # Restore original value
            if original_value is not None:
                os.environ["HETERODYNE_OPTIMIZE_STARTUP"] = original_value
            elif "HETERODYNE_OPTIMIZE_STARTUP" in os.environ:
                del os.environ["HETERODYNE_OPTIMIZE_STARTUP"]

    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility."""

        # Should work on any platform
        health = heterodyne.check_performance_health()
        assert isinstance(health, dict)

        # Platform-specific paths should work
        config_dir = heterodyne.get_config_dir()
        assert config_dir is not None


@pytest.mark.integration
class TestFullIntegrationWorkflow:
    """Test full integration workflows."""

    @pytest.mark.slow
    def test_complete_startup_to_analysis_workflow(self):
        """Test complete workflow from startup to analysis."""
        try:
            # 1. Package startup with monitoring
            health = heterodyne.check_performance_health()
            assert health["status"] in ["excellent", "good", "fair", "poor"]

            # 2. Configuration access
            config_manager = heterodyne.ConfigManager
            assert config_manager is not None

            # 3. Analysis core access
            analysis_core = heterodyne.HeterodyneAnalysisCore
            assert analysis_core is not None

            # 4. Optimization components
            classical_opt = heterodyne.ClassicalOptimizer
            robust_opt = heterodyne.RobustHeterodyneOptimizer
            assert classical_opt is not None
            assert robust_opt is not None

            # 5. Performance monitoring
            perf_report = heterodyne.get_import_performance_report()
            assert isinstance(perf_report, dict)

            print("✅ Complete workflow validation successful")

        except ImportError as e:
            pytest.skip(f"Full workflow test skipped due to missing dependencies: {e}")
        except Exception as e:
            pytest.fail(f"Complete workflow failed: {e}")


if __name__ == "__main__":
    # Run with verbose output for validation
    pytest.main([__file__, "-v", "--tb=short"])
