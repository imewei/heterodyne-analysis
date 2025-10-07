Research Documentation
======================

This section provides comprehensive research-grade documentation for the heterodyne-analysis package,
including theoretical foundations, computational methods, and experimental validation.

.. toctree::
   :maxdepth: 2
   :caption: Research Components

   theoretical_framework
   computational_methods
   publications

Overview
--------

The heterodyne-analysis package implements the theoretical framework for analyzing X-ray Photon
Correlation Spectroscopy (XPCS) data under nonequilibrium conditions, as described in:

**He, H., Liang, H., Chu, M., Jiang, Z., de Pablo, J.J., Tirrell, M.V., Narayanan, S., & Chen, W.**
*Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter.*
Proceedings of the National Academy of Sciences, 121(31), e2401162121 (2024).
DOI: `10.1073/pnas.2401162121 <https://doi.org/10.1073/pnas.2401162121>`_

Research Scope
--------------

Scientific Applications
~~~~~~~~~~~~~~~~~~~~~~~

* **Soft Matter Physics**: Colloidal dynamics, active matter, biological systems
* **Flow Rheology**: Shear thinning/thickening, microfluidics, complex fluids
* **Materials Science**: Phase transitions, glass transition, crystallization

Computational Innovation
~~~~~~~~~~~~~~~~~~~~~~~~

* **High-Performance Computing**: Numba JIT compilation with 3-5x speedup
* **Robust Optimization**: Distributionally robust optimization with uncertainty quantification
* **Memory Efficiency**: Vectorized operations with smart caching strategies

Experimental Integration
~~~~~~~~~~~~~~~~~~~~~~~~

* **Synchrotron Facilities**: Advanced Photon Source (APS) integration
* **Data Standards**: HDF5 and standardized XPCS data formats
* **Validation Protocols**: Statistical validation and uncertainty analysis

Mathematical Framework
----------------------

The package implements the **two-component heterodyne scattering model** (Equation S-95):

.. math::

   c_2(\vec{q}, t_1, t_2, \phi) = 1 + \frac{\beta}{f^2} \Bigg[
   [x_r(t_1)x_r(t_2)]^2 e^{-q^2 \int_{t_1}^{t_2} J_r(t) dt} + \\
   [x_s(t_1)x_s(t_2)]^2 e^{-q^2 \int_{t_1}^{t_2} J_s(t) dt} + \\
   2x_r(t_1)x_r(t_2)x_s(t_1)x_s(t_2)e^{-\frac{1}{2}q^2 \int_{t_1}^{t_2} [J_s(t)+J_r(t)] dt}
   \cos\left[q \cos(\phi) \int_{t_1}^{t_2} v(t) dt\right]
   \Bigg]

where :math:`f^2 = [x_s(t_1)^2 + x_r(t_1)^2][x_s(t_2)^2 + x_r(t_2)^2]`.

**Key Parameters:**

* :math:`\vec{q}`: scattering wavevector [Å⁻¹]
* :math:`J_r(t), J_s(t)`: transport coefficients for reference and sample [Å²/s]
* :math:`x_s(t_1), x_s(t_2)`: sample fraction at times t₁ and t₂ (two-time correlation)
* :math:`v(t)`: time-dependent mean velocity [nm/s]
* :math:`\phi`: relative angle = :math:`\phi_0 - \phi_{\text{scattering}}` (flow minus scattering)
* :math:`\beta`: instrumental contrast parameter

Analysis Modes
--------------

.. list-table:: Analysis Capabilities
   :header-rows: 1
   :widths: 20 10 40 15 15

   * - Mode
     - Parameters
     - Physical Description
     - Complexity
     - Applications
   * - Static Isotropic
     - 3
     - Brownian motion only
     - O(N)
     - Equilibrium systems
   * - Static Anisotropic
     - 3
     - Angular dependence
     - O(N log N)
     - Structured materials
   * - Laminar Flow
     - 7
     - Full nonequilibrium
     - O(N²)
     - Flowing systems

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Performance Comparison (Intel Xeon, 8 cores)
   :header-rows: 1
   :widths: 20 20 20 20

   * - Data Size
     - Pure Python
     - Numba JIT
     - Speedup
   * - 100 points
     - 2.3 s
     - 0.7 s
     - 3.3×
   * - 500 points
     - 12.1 s
     - 3.2 s
     - 3.8×
   * - 1000 points
     - 45.2 s
     - 8.9 s
     - 5.1×
   * - 5000 points
     - 892 s
     - 178 s
     - 5.0×

Citation Guidelines
-------------------

Primary Research Citation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @article{he2024transport,
     title={Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter},
     author={He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh and Chen, Wei},
     journal={Proceedings of the National Academy of Sciences},
     volume={121},
     number={31},
     pages={e2401162121},
     year={2024},
     publisher={National Academy of Sciences},
     doi={10.1073/pnas.2401162121}
   }

Software Package Citation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @software{heterodyne_analysis,
     title={heterodyne-analysis: High-performance XPCS analysis with robust optimization},
     author={Chen, Wei and He, Hongrui},
     year={2024},
     url={https://github.com/imewei/heterodyne},
     version={0.7.1},
     institution={Argonne National Laboratory}
   }

Research Collaboration
----------------------

The development of this research software is supported by:

**Funding Agencies**
  * U.S. Department of Energy, Office of Science, Basic Energy Sciences
  * Advanced Photon Source User Facility

**Collaborating Institutions**
  * Argonne National Laboratory - X-ray Science Division
  * University of Chicago - Pritzker School of Molecular Engineering

**Contact Information**
  * **Principal Investigator**: Wei Chen (wchen@anl.gov)
  * **Lead Developer**: Hongrui He
  * **Technical Support**: GitHub Issues
  * **Research Collaboration**: Argonne National Laboratory, X-ray Science Division
