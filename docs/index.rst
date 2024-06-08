Documentation
=============

.. admonition:: tensortrax

   Differentiable Tensors based on NumPy Arrays.

Highlights
----------
- Write differentiable code with Tensors based on NumPy arrays
- Designed to operate on input arrays with (elementwise-operating) trailing axes
- Essential vector/tensor Hyper-Dual number math, including limited support for ``einsum`` (restricted to max. three operands)
- Math is limited but similar to NumPy, try to use ``import tensortrax.math as tm`` instead of ``import numpy as np`` inside functions to be differentiated
- Forward Mode Automatic Differentiation (AD) using Hyper-Dual Tensors, up to second order derivatives
- Create functions in terms of Hyper-Dual Tensors
- Evaluate the function, the gradient (jacobian) and the hessian of scalar-valued functions or functionals on given input arrays
- Straight-forward definition of custom functions in variational-calculus notation
- Stable gradient and hessian of eigenvalues obtained from ``eigvalsh`` in case of repeated equal eigenvalues

.. note::
   Please keep in mind that ``tensortrax`` is not imitating a 100% full-featured NumPy, e.g. like https://github.com/HIPS/autograd [1]_. No arbitrary-order gradients or gradients-of-gradients are supported. The capability is limited to first- and second order gradients of a given function. Also, ``tensortrax`` provides no support for ``dtype=complex`` and ``out``-keywords are not supported.

Motivation
----------
Gradient and hessian evaluations of functions or functionals based on tensor-valued input arguments are a fundamental repetitive and (error-prone) task in constitutive hyperelastic material formulations used in continuum mechanics of solid bodies. In the worst case, conceptual ideas are impossible to pursue because the required tensorial derivatives are not readily achievable. The Hyper-Dual number approach enables a generalized and systematic way to overcome this deficiency [2]_. Compared to existing Hyper-Dual Number libaries ([3]_, [4]_) which introduce a new (hyper-) dual ``dtype`` (treated as ``dtype=object`` in NumPy), ``tensortrax`` relies on its own ``Tensor`` class. This approach involves a re-definition of all essential math operations (and NumPy-functions), whereas the ``dtype``-approach supports most basic math operations out of the box. However, in ``tensortrax``, NumPy and all its underlying linear algebra functions operate on default data types (e.g. ``dtype=float``). This allows to support functions like ``np.einsum()``. Beside the differences concerning the underlying ``dtype``, ``tensortrax`` is formulated on easy-to-understand (tensorial) calculus of variation. Hence, gradient- and hessian-vector products are evaluated with very little overhead compared to analytic formulations.

Installation
------------
Install ``tensortrax`` from `PyPI <https://pypi.org/project/tensortrax/>`_, the Python Package Index.

.. code-block:: shell

   pip install tensortrax[all]

``tensortrax`` has minimal requirements, all available at PyPI.

..  list-table:: Dependencies
    :widths: 20 80
    :header-rows: 1

    * - Package
      - Usage
    * - `numpy <https://github.com/numpy/numpy>`_
      - for array operations
    * - `joblib <https://github.com/joblib/joblib>`_
      - for threaded function, gradient, hessian and jacobian evaluations

To install optional dependencies as well, add ``[all]`` to the install command: ``pip install tensortrax[all]``.

..  list-table:: Optional Dependencies
    :widths: 20 80
    :header-rows: 1
    
    * - Package
      - Usage
    * - `scipy <https://github.com/scipy/scipy>`_
      - for extended math-support

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   examples/index
   knowledge
   tensortrax

License
-------

tensortrax - Math on (Hyper-Dual) Tensors with Trailing Axes (C) 2022-2024 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see `<https://www.gnu.org/licenses/>`_.

References
----------
.. [1] D. Maclaurin, D. Duvenaud, M. Johnson and J. Townsend, *Autograd*. Online. Available: https://github.com/HIPS/autograd.
.. [2] J. Fike and J. Alonso, *The Development of Hyper-Dual Numbers for Exact Second-Derivative Calculations*, 49th AIAA Aerospace Sciences Meeting including the New Horizons Forum and Aerospace Exposition. American Institute of Aeronautics and Astronautics, Jan. 04, 2011, doi: `10.2514/6.2011-886 <https://doi.org/10.2514/6.2011-886>`_.
.. [3] P. Rehner and G. Bauer, *Application of Generalized (Hyper-) Dual Numbers in Equation of State Modeling*, Frontiers in Chemical Engineering, vol. 3, 2021. Available: https://github.com/itt-ustutt/num-dual.
.. [4] T. Oberbichler, *HyperJet*. Online. Available: http://github.com/oberbichler/HyperJet.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
