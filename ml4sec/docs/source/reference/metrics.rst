Metrics
=======

True/False Positive/Negatives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The methods to compute True/False Positive/Negative values should be implemented by the students.
To this end, they receive the actual values ``y_true`` and predicted values ``y_pred``, both as a ``numpy.array`` that are ``+1`` for known values and ``-1`` for anomalous values.

.. automethod:: ml4sec.Assignment.TP

.. automethod:: ml4sec.Assignment.TN

.. automethod:: ml4sec.Assignment.FP

.. automethod:: ml4sec.Assignment.FN

Test
^^^^
To test the implementations of True/False Positive/Negative values, we provide test cases using the function ``test_metrics()``.

.. automethod:: ml4sec.Assignment.test_metrics

Metric implementations
^^^^^^^^^^^^^^^^^^^^^^
Based on the student impelmentations, we compute the following metrics:

.. automethod:: ml4sec.Assignment.TPR

.. automethod:: ml4sec.Assignment.TNR

.. automethod:: ml4sec.Assignment.FPR

.. automethod:: ml4sec.Assignment.FNR

.. automethod:: ml4sec.Assignment.accuracy

.. automethod:: ml4sec.Assignment.precision

.. automethod:: ml4sec.Assignment.recall

.. automethod:: ml4sec.Assignment.f1_score

Results
^^^^^^^
We provide functions to compute all metrics, and show them using ``show_report()``.

.. automethod:: ml4sec.Assignment.prediction_report

.. automethod:: ml4sec.Assignment.show_report
