Extract features
================

Students will need to implement their own ``extract()`` method.
This method receives the ``protocol, src, sport, dst, dport`` values giving information for the entire flow.
Additionally, the method receives ``timestamps`` and ``sizes``, two ``np.array`` objects giving the numerical value of the timestamps and packet sizes, respectively.
In this assignment, the students should extract features from these values and output them as an ``array-like`` object, e.g., ``list()``, ``np.array()``, ``tuple``.

.. automethod:: ml4sec.Assignment.extract

We create a feature matrix using the student implementation of ``Assignment.extract()`` with the ``feature_matrix()`` method.
This method calls the ``extract()`` method for each flow in the dataset and transforms the output to a numpy array (matrix) of shape=``(n_flows, n_features)``.
Because of the transformation to a numpy array, the output of ``extract()`` should be an ``array-like`` object.

.. automethod:: ml4sec.Assignment.feature_matrix

We show the feature matrix by transforming the output of ``Assignment.feature_matrix()`` to a pandas DataFrame, which is automatically displayed nicely in Jupyter Notebook.

.. automethod:: ml4sec.Assignment.show_matrix

The ``test_extract()`` method loops over a given number of flows, calls the ``extract()`` method for each flow, and shows the output.
This method is available for students to test their implementation of ``extract()``.

.. automethod:: ml4sec.Assignment.test_extract
