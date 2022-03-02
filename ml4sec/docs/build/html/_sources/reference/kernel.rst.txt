Kernel functions
================

For a given kernel and support vectors we compute the ``soft_score`` that computes the overall ``score`` of a point ``X`` by summing the ``kernel_scores`` of the point ``X`` with each support vector ``Y``.

.. automethod:: ml4sec.Assignment.soft_score

Plotting
^^^^^^^^
To show how this works, we plot the kernel score of randomly generated ``support_vectors`` for all combinations of parameters ``sigmas`` and ``thresholds`` using the following methods:

.. automethod:: ml4sec.Assignment.plot_kernels

.. automethod:: ml4sec.Assignment._plot_kernel_
