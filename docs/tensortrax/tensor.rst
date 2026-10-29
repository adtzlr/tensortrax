.. _api-tensor:

Tensor
~~~~~~
This module provides the base class for a tensor.

**Core**

.. currentmodule:: tensortrax

.. autosummary::

   Tensor


**Detailed API Reference**

.. autoclass:: tensortrax.Tensor
   :members:
   :undoc-members:
   :inherited-members:

.. data:: ZERO

   A structural zero for dual data which is not tracked. It absorbs products and
   vanishes in sums, without allocating (and computing with) arrays of zeros, e.g.
   ``Tensor(x, δx=ZERO, Δx=Δ(A), Δδx=ZERO)``, see
   :func:`~tensortrax.math.no_variation`.