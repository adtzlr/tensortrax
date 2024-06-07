.. _knowledge-derivative:

Variational Calculus
====================
The calculus of variation deals with variations, i.e. small changes in functions and functionals. A small-change in a function is evaluated by applying small changes on the input arguments, see Eq. :eq:`variation`. While the original function only depends on the input argument(s), its **variation** also depends on the **variation of the input argument(s)**.

.. math::
   :label: variation

   f &= f(x)

   \delta f &= \delta f(x, \delta x)

   \delta f &= \frac{\partial f(x)}{\partial x}\ \delta x

The partial derivative of a function :math:`f(x)` w.r.t. its input argument :math:`x` is obtained from Eq. :eq:`variation` by setting the variation of the input argument to :math:`\delta x=1`. The same holds also for nested functions by the application of the chain rule, see Eq. :eq:`variation-chainrule-1`.

.. math::
   :label: variation-chainrule-1

   y &= y(x)

   f &= f(y(x))

   \delta f &= \delta f(y(x), \delta y(x, \delta x))

The partial derivative of a nested function :math:`f(y(x))` w.r.t. its base input argument :math:`x` is obtained from Eq. :eq:`variation-chainrule-2` by setting the variation of the base input argument as before to :math:`\delta x=1`.

.. math::
   :label: variation-chainrule-2

   \delta f &= \frac{\partial f(y)}{\partial y}\ \delta y

   \delta f &= \frac{\partial f(y)}{\partial y}\ \frac{\partial y(x)}{\partial x}\ \delta x

By inserting given values for :math:`\delta x` one obtains the so-called **gradient-vector-product** for vector-valued input arguments.

Example
~~~~~~~
Given a differentiable (nested) function along with its derivative. The variation of the nested function w.r.t. its base input argument is carried out by the chain rule, see Eq. :eq:`example-variation-1` and Eq. :eq:`example-variation-2`.

.. math::
   :label: example-variation-1

   y(x) &= x^2

   \delta y(x, \delta x) &= 2 x\ \delta x

.. math::
   :label: example-variation-2

   f(y) &= \sin(y)

   \delta f(y, \delta y) &= \cos(y)\ \delta y

The default evaluation graph of the nested function is shown in Eq.
:eq:`evaluation-graph`.

.. math::
   :label: evaluation-graph

   \begin{matrix}
      x & & \rightarrow & & y(x) = x^2 & & \rightarrow & & f(y) = \sin(y)
   \end{matrix}

By augmenting the evaluation graph with its dual counterpart, variations are computed
side-by-side with the default graph, see Eq. :eq:`evaluation-graph-augmented`.

.. math::
   :label: evaluation-graph-augmented

   \begin{matrix}
      x & & \rightarrow & & y(x) = x^2 & & \rightarrow & & f(y) = \sin(y) \\
      \delta x & & \rightarrow & & \delta y(x, \delta x) = 2x\ \delta x & & \rightarrow & & \delta f(y, \delta y) = \cos(y)\ \delta y
   \end{matrix}
