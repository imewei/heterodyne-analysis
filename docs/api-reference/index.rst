API Reference
=============

Complete API documentation for the heterodyne analysis package.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   overview
   analysis-core
   core
   robust
   utilities

Core Classes
------------

* :class:`~heterodyne.analysis.core.HeterodyneAnalysisCore` - Main analysis orchestrator
* :class:`~heterodyne.core.config.ConfigManager` - Configuration management
* :class:`~heterodyne.optimization.classical.ClassicalOptimizer` - Classical optimization
* :class:`~heterodyne.optimization.robust.RobustHeterodyneOptimizer` - Robust optimization

Quick Reference
---------------

**Essential Imports**:

.. code-block:: python

   from heterodyne import HeterodyneAnalysisCore, ConfigManager
   from heterodyne.optimization.classical import ClassicalOptimizer
   from heterodyne.optimization.robust import RobustHeterodyneOptimizer

**Basic Workflow**:

.. code-block:: python

   import json
   from heterodyne.analysis.core import HeterodyneAnalysisCore
   from heterodyne.optimization.classical import ClassicalOptimizer
   from heterodyne.optimization.robust import RobustHeterodyneOptimizer

   # 1. Load configuration
   with open("config.json") as f:
       config = json.load(f)

   # 2. Initialize analysis core
   core = HeterodyneAnalysisCore(config)
   core.load_experimental_data()

   # 3. Run classical optimization
   classical = ClassicalOptimizer(core, config)
   params, results = classical.run_classical_optimization_optimized(
       phi_angles=phi_angles,
       c2_experimental=c2_data
   )

   # 4. Or run robust optimization for noisy data
   robust = RobustHeterodyneOptimizer(core, config)
   robust_results = robust.optimize(
       phi_angles=phi_angles,
       c2_experimental=c2_data,
       method="wasserstein"
   )

Module Index
------------

The package includes the following key modules:

* **heterodyne.core** - Core functionality and configuration
* **heterodyne.analysis.core** - Main analysis engine
* **heterodyne.optimization.classical** - Classical optimization (Nelder-Mead, Gurobi)
* **heterodyne.optimization.robust** - Robust optimization (Wasserstein DRO, Scenario-based, Ellipsoidal)

.. note::
   For detailed API documentation, see the individual module pages in the navigation.

..
   Temporarily disabled autosummary due to import issues

   .. autosummary::
      :toctree: _autosummary
      :template: module.rst

      heterodyne.core
      heterodyne.core.config
      heterodyne.core.kernels
      heterodyne.core.io_utils
      heterodyne.analysis.core
      heterodyne.optimization.classical
