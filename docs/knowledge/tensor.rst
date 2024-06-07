.. _knowledge-tensor:

Variations of Tensors
=====================
As an extension to the scalar-valued explanations given in :ref:`knowledge-derivative`, let us consider a scalar-valued function of as second-order tensor-based input argument, see Eq. :eq:`variation-tensor`.

.. math::
   :label: variation-tensor

   \psi &= \psi(\boldsymbol{F})

   \delta \psi &= \delta \psi(\boldsymbol{F}, \delta \boldsymbol{F})


Let's take the trace of a tensor product as an example. The variation is carried out in Eq. :eq:`variation-tensor-example-1`.

.. math::
   :label: variation-tensor-example-1

   \psi &= tr(\boldsymbol{F}^T \boldsymbol{F}) = \boldsymbol{F} : \boldsymbol{F}

   \delta \psi &= \delta \boldsymbol{F} : \boldsymbol{F} + \boldsymbol{F} : \delta \boldsymbol{F} = 2 \ \boldsymbol{F} : \delta \boldsymbol{F}

The :math:`P_{ij}` - component of the jacobian :math:`\boldsymbol{P}` is now numerically evaluated by setting the respective variational component :math:`\delta F_{ij}` of the tensor to one and all other components to zero. In total, :math:`i \cdot j` function calls are necessary to assemble the full jacobian. For example, the :math:`12` - component is evaluated as given in Eq. :eq:`variation-tensor-component`.

.. math::
   :label: variation-tensor-component

   \delta \boldsymbol{F}_{(12)} &= \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}

   \delta_{(12)} \psi &= \frac{\partial \psi}{\partial F_{12}} &= 2 \ \boldsymbol{F} : \delta \boldsymbol{F}_{(12)} = 2 \ \boldsymbol{F} : \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}

The second order variation, i.e. a variation applied on another variation of a function is evaluated in the same way as a first order variation, see Eq. :eq:`variation-tensor-second`.

.. math::
   :label: variation-tensor-second

   \Delta \delta \psi = 2 \ \delta \boldsymbol{F} : \Delta \boldsymbol{F} + 2 \ \boldsymbol{F} : \Delta \delta \boldsymbol{F}

Once again, each component :math:`A_{ijkl}` of the fourth-order hessian is numerically evaluated. In total, :math:`i \cdot j \cdot k \cdot l` function calls are necessary to assemble the full hessian (without considering symmetry). For example, the :math:`1223` - component is evaluated by setting :math:`\Delta \delta \boldsymbol{F} = \boldsymbol{0}` and :math:`\delta \boldsymbol{F}` as well as :math:`\Delta \boldsymbol{F}` as given in Eq. :eq:`variation-tensor-second-block` and Eq. :eq:`variation-tensor-second-components`.

.. math::
   :label: variation-tensor-second-block

   \delta \boldsymbol{F}_{(12)} &= \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}

   \Delta \boldsymbol{F}_{(23)} &= \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}

   \Delta \delta \boldsymbol{F} &= \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}

.. math::
   :label: variation-tensor-second-components

   \Delta_{(23)} \delta_{(12)} \psi &= \Delta_{(12)} \delta_{(23)} \psi = \frac{\partial^2 \psi}{\partial F_{12}\ \partial F_{23}}

   \Delta_{(23)} \delta_{(12)} \psi &= 2 \ \delta \boldsymbol{F}_{(12)} : \Delta \boldsymbol{F}_{(23)} + 2 \ \boldsymbol{F} : \Delta \delta \boldsymbol{F}

