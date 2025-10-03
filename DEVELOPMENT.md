# Development Guide

This guide covers all aspects of developing the heterodyne analysis package: setup,
workflow, testing, API reference, and security practices.

## Development Setup

### Prerequisites

- Python 3.12+
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/imewei/heterodyne.git
cd heterodyne

# Install with development dependencies
pip install -e ".[all,dev,docs]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest -v
```

## Code Quality Standards

### Formatting and Linting

```bash
# Format code
black heterodyne --line-length 88
isort heterodyne --profile black

# Lint code
flake8 heterodyne --max-line-length 88
ruff heterodyne

# Type checking
mypy heterodyne --ignore-missing-imports

# Security scanning
bandit -r heterodyne/
pip-audit
```

### Code Quality Status

- ✅ **Black**: 100% compliant (88-character line length)
- ✅ **isort**: 100% compliant (import sorting)
- ✅ **Bandit**: 0 medium/high severity security issues
- ⚠️ **flake8**: ~400 remaining issues (mostly line length in data scripts)
- ⚠️ **mypy**: ~250 type annotation issues (missing library stubs)

### Pre-commit Hooks

Pre-commit hooks run automatically on every commit. Manual usage:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run ruff --all-files

# Skip hooks (emergency only)
SKIP=mypy,bandit git commit -m "Emergency commit"
```

**Configured hooks:**

- **Black**: Code formatting (88 characters)
- **isort**: Import sorting (black profile)
- **Ruff**: Fast linting with auto-fixes
- **Flake8**: Style guide enforcement
- **MyPy**: Type checking (excludes tests)
- **Bandit**: Security scanning
- **Pre-commit**: File quality (whitespace, EOF, YAML/JSON validation)
- **mdformat**: Markdown formatting
- **nbqa**: Jupyter notebook formatting

## Testing

### Running Tests

```bash
# Basic test run
pytest -v

# With coverage
pytest --cov=heterodyne --cov-report=html

# Specific test categories
pytest -v -m "not slow"           # Skip slow tests
pytest -v -m "performance"       # Performance tests only
pytest -v -m "integration"       # Integration tests only
```

### Test Categories

- `slow`: Time-intensive tests
- `integration`: Integration tests
- `performance`: Performance benchmarks
- `regression`: Regression tests

### Writing Tests

- Place tests in `heterodyne/tests/`
- Use descriptive test names: `test_[module]_[feature]_[condition]`
- Add appropriate markers: `@pytest.mark.slow`
- Maintain test coverage

## API Reference

### Core Analysis Classes

#### `HeterodyneAnalysisCore`

Main analysis engine for XPCS data processing.

```python
from heterodyne.analysis.core import HeterodyneAnalysisCore

class HeterodyneAnalysisCore:
    def __init__(self, config: Dict[str, Any] | str)

    # Key Methods
    def load_experimental_data(
        self, phi_angles: np.ndarray, num_angles: int,
        file_path: str = None
    ) -> np.ndarray

    def calculate_chi_squared_optimized(
        self, parameters: np.ndarray, phi_angles: np.ndarray,
        c2_experimental: np.ndarray, method_name: str = ""
    ) -> float

    def is_static_mode(self) -> bool
    def get_effective_parameter_count(self) -> int
    def calculate_c2_nonequilibrium_laminar_parallel(
        self, parameters: np.ndarray, phi_angles: np.ndarray
    ) -> np.ndarray
```

#### `ClassicalOptimizer`

Classical optimization methods (Nelder-Mead, Gurobi).

```python
from heterodyne.optimization.classical import ClassicalOptimizer

class ClassicalOptimizer:
    def __init__(self, analysis_core: HeterodyneAnalysisCore, config: Dict[str, Any])

    def run_classical_optimization_optimized(
        self, initial_parameters: Optional[np.ndarray] = None,
        methods: Optional[List[str]] = None,
        phi_angles: Optional[np.ndarray] = None,
        c2_experimental: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Any]

    def run_single_method(
        self, method: str, objective_func: Callable, initial_parameters: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Tuple[bool, Union[scipy.optimize.OptimizeResult, Exception]]

    def get_available_methods(self) -> List[str]
    def create_objective_function(
        self, phi_angles: np.ndarray, c2_experimental: np.ndarray, method_name: str = "Classical"
    ) -> Callable
```

**Gurobi Trust Region Features:**

- Iterative trust region SQP approach
- Adaptive trust region: radius 0.1 → 1e-8 to 1.0 range
- Parameter-scaled finite differences
- Expected convergence: 10-30 iterations

#### `RobustHeterodyneOptimizer`

Robust optimization with uncertainty quantification.

```python
from heterodyne.optimization.robust import RobustHeterodyneOptimizer

class RobustHeterodyneOptimizer:
    def __init__(self, analysis_core: HeterodyneAnalysisCore, config: Dict[str, Any])

    def _solve_distributionally_robust(
        self, theta_init: np.ndarray, phi_angles: np.ndarray,
        c2_experimental: np.ndarray, uncertainty_radius: float = 0.05
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]

    def _solve_scenario_based_robust(...) -> Tuple[Optional[np.ndarray], Dict[str, Any]]
    def _solve_ellipsoidal_robust(...) -> Tuple[Optional[np.ndarray], Dict[str, Any]]
```

**Available Methods:**

- `wasserstein`: Wasserstein Distributionally Robust Optimization
- `scenario`: Scenario-based robust optimization
- `ellipsoidal`: Ellipsoidal uncertainty sets

#### `ConfigManager`

Configuration management and validation.

```python
from heterodyne.core.config import ConfigManager

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None)

    def load_config(self, config_path: str) -> Dict[str, Any]
    def validate_config(self, config: Dict[str, Any]) -> bool
    def get_default_config(self) -> Dict[str, Any]
    def is_static_mode_enabled(self) -> bool
    def get_effective_parameter_count(self) -> int
    def get_parameter_bounds(self) -> List[Tuple[float, float]]
```

### Utility Functions

#### High-Performance Kernels

```python
from heterodyne.core.kernels import (
    compute_g1_correlation_numba,
    create_time_integral_matrix_numba,
    solve_least_squares_batch_numba,
    compute_chi_squared_batch_numba
)

# JIT-compiled functions with fallbacks when Numba unavailable
def compute_g1_correlation_numba(diffusion_coeff, shear_rate, time_points, angles) -> np.ndarray
def solve_least_squares_batch_numba(theory_batch, exp_batch) -> np.ndarray
```

#### Data Utilities

```python
from heterodyne.utils.data import (
    load_experimental_data,
    validate_data_format,
    preprocess_correlation_data,
    filter_angles_by_ranges
)

def load_experimental_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
def validate_data_format(data: np.ndarray) -> bool
```

#### Performance Monitoring

```python
from heterodyne.utils.performance import (
    benchmark_method,
    monitor_memory_usage,
    optimize_numerical_environment
)

def benchmark_method(method_func: Callable, *args, **kwargs) -> Dict[str, Any]
def monitor_memory_usage() -> Dict[str, float]
```

### Error Handling

```python
from heterodyne.exceptions import (
    HeterodyneError,
    ConfigurationError,
    DataFormatError,
    OptimizationError
)

try:
    core = HeterodyneAnalysisCore(config)
    optimizer = ClassicalOptimizer(core)
    result = optimizer.run_optimization()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except OptimizationError as e:
    print(f"Optimization failed: {e}")
```

### Advanced Usage

#### Custom Objective Functions

```python
from heterodyne.optimization.classical import ClassicalOptimizer

class CustomOptimizer(ClassicalOptimizer):
    def create_objective_function(self) -> Callable:
        def custom_objective(params: np.ndarray) -> float:
            D0, alpha, D_offset = params[:3]
            model_c2 = self.compute_model_correlation(params)
            chi2 = np.sum((self.analysis_core.c2_experimental - model_c2)**2)
            regularization = 0.01 * np.sum(params**2)
            return chi2 + regularization
        return custom_objective
```

#### Batch Processing

```python
def batch_analyze(data_files: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = []
    for data_file in data_files:
        core = HeterodyneAnalysisCore(config)
        core.load_data(data_file)
        optimizer = ClassicalOptimizer(core)
        result = optimizer.run_optimization()
        results.append({
            "file": data_file,
            "parameters": result.x,
            "chi_squared": result.fun,
            "success": result.success
        })
    return results
```

## Repository Management

### Cleaning Development Artifacts

```bash
# Quick clean (recommended)
make clean

# Manual clean (advanced)
git clean -xfd --exclude=data --exclude=heterodyne_results

# Clean specific types
make clean-pyc     # Python bytecode
make clean-test    # Test artifacts
make clean-build   # Build artifacts
```

**Files automatically ignored:**

- Bytecode: `__pycache__/`, `*.py[cod]`
- Build: `build/`, `dist/`, `*.egg-info/`
- Tests: `.pytest_cache/`, `.coverage*`, `htmlcov/`
- IDE: `.mypy_cache/`, `.idea/`, `.vscode/`, `.DS_Store`
- Data: `data/`, `heterodyne_results*/`, `my_config*.json`

### Before Committing

```bash
make clean
git status  # Verify working tree is clean
```

## Submitting Changes

### Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes following guidelines
# ...

# Clean and test
make clean
pytest -v
make lint

# Commit changes
git add .
git commit -m "Descriptive commit message"

# Push and create pull request
git push origin feature/your-feature-name
```

### Pull Request Guidelines

- Include tests for new features
- Update documentation as needed
- Follow code style guidelines
- Ensure all CI checks pass
- Write clear commit messages
- Reference related issues

## Release Process

For maintainers:

```bash
# Update version in heterodyne/__init__.py
# Clean and test
make clean
pytest -v
make lint

# Build and check distribution
make build
make check

# Tag and release
git tag v0.x.x
git push origin v0.x.x
make upload  # Upload to PyPI
```

## Security

### Security Features

- **Bandit**: Continuous security scanning (0 medium/high severity issues)
- **pip-audit**: Dependency vulnerability scanning
- **Pre-commit hooks**: Automatic security checks
- **Safe coding practices**: No hardcoded secrets, secure file operations

### Reporting Vulnerabilities

**Critical security issues:**

- Email: wchen@anl.gov with subject "SECURITY: Heterodyne Vulnerability"
- Include: Description, reproduction steps, impact assessment
- Response: Within 48 hours
- Timeline: Fix within 7-14 days

**Non-critical issues:**

- Create GitHub issue with "security" label
- Use standard issue template

### Security Response Process

1. **Acknowledgment**: Receipt within 48 hours
2. **Assessment**: Impact review (1-3 days)
3. **Development**: Fix development and testing (3-7 days)
4. **Release**: Security patch with advisory
5. **Disclosure**: Coordinated disclosure after fix

### Security Tools

```bash
# Manual security scanning
bandit -r heterodyne/ -f json -o bandit_report.json
pip-audit --desc --format=json --output=audit_report.json
safety check --json --output safety_report.json
```

## Support

- **Issues**: [GitHub Issues](https://github.com/imewei/heterodyne/issues)
- **Documentation**: [heterodyne.readthedocs.io](https://heterodyne.readthedocs.io/)
- **Email**: wchen@anl.gov

## License

By contributing, you agree that your contributions will be licensed under the MIT
License.
