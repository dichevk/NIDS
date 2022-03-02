Scale features
==============
Students need to implement the ``scale()`` method to scale features in the ``feature_matrix``.
The method takes the ``matrix`` and, optionally ``minimum`` and ``maximum`` values as input.
Students should compute the ``minimum`` and ``maximum`` values of each feature in this matrix if these values are `not` given as input, i.e., when they are ``None``.
Next, the (computed or given) ``minimum`` and ``maximum`` should be used to apply min-max scaling for the feature matrix.

.. automethod:: ml4sec.Assignment.scale

Visualisation
^^^^^^^^^^^^^
We visualise the scaled ``feature_matrix`` as a table using the ``show_scaled()`` method.
This produces a pandas.DataFrame of the scaled ``feature_matrix`` that is automatically displayed nicely in Jupyter notebook.

.. automethod:: ml4sec.Assignment.show_scaled

Additionally, we want to compare the unscaled and scaled data for both the ``benign.csv`` and ``unknown.csv`` data.
We plot this data using the ``plot_scaled`` method (which in turn uses the ``plot_data`` helper method).

.. note::
   As we may have an arbitrary number of features that we want to plot in a 2D space, we use PCA to `compress` the data into two dimensions.
   We basically only show the two principal components with the highest eigenvalues.
   See the `PCA wikipedia article <https://en.wikipedia.org/wiki/Principal_component_analysis>`_ for a full explanation on this technique.

.. automethod:: ml4sec.Assignment.plot_scaled

.. automethod:: ml4sec.Assignment.plot_data
