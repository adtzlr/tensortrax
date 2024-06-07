tensortrax documentation
========================

Highlights
----------
- Designed to operate on input arrays with (elementwise-operating) trailing axes
- Essential vector/tensor Hyper-Dual number math, including limited support for ``einsum`` (restricted to max. three operands)
- Math is limited but similar to NumPy, try to use ``import tensortrax.math as tm`` instead of ``import numpy as np`` inside functions to be differentiated
- Forward Mode Automatic Differentiation (AD) using Hyper-Dual Tensors, up to second order derivatives
- Create functions in terms of Hyper-Dual Tensors
- Evaluate the function, the gradient (jacobian) and the hessian of scalar-valued functions or functionals on given input arrays
- Straight-forward definition of custom functions in variational-calculus notation
- Stable gradient and hessian of eigenvalues obtained from `eigvalsh` in case of repeated equal eigenvalues

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
