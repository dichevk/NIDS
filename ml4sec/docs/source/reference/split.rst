Split data
==========
Students need to split the data by implementing the ``split()`` method.
This takes as input the ``data`` that needs to be split, together with its corresponding ``labels``.
The split is based on three parameters:

 * ``apps_train`` specifies the apps used for both training and testing. The given ``ratio`` of flows for these apps should be in ``data_train`` and ``1-ratio`` of flows for these apps should be in ``data_test``.
 * ``apps_test``  specifies the apps used `only` for testing, all flows of these apps should be in the resulting ``data_test`` and no flows flows of these apps should be in the resulting ``data_train``.
 * ``ratio`` specifies the ratio of flows from ``apps_train`` that should be in ``data_train``. The remaining flows should be in ``data_test``.

.. note::
   Students may expect that an app ``A`` is specified in either the ``apps_train`` or ``apps_test`` but not both. I.e., the union of ``apps_train`` and ``apps_test`` is the complete set of apps and the intersection of ``apps_train`` and ``apps_test`` is empty.

.. automethod:: ml4sec.Assignment.split

We automatically apply the split on the data from ``benign.csv`` for a given ``apps_train`` (which automatically computes the ``apps_test``) and a given ``ratio``.

.. automethod:: ml4sec.Assignment.get_split

Test
^^^^
We provide some test cases for students to check their implementation of ``split()``. The ``test_split`` method runs each test case and ``assert_split`` checks whether each test case is correct.

.. automethod:: ml4sec.Assignment.test_split

.. automethod:: ml4sec.Assignment.assert_split
