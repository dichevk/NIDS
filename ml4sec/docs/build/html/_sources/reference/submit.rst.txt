Submit
======
Students can submit their assignment using the ``Assignment.submit()`` function.
Submission requires a network connection and the python ``requests`` library.
Once called, the ``submit()`` method automatically calls ``Assignment.NIDS.fit()`` with the data from ``benign.csv``.
Next, it calls ``Assignment.NIDS.predict()`` with the data from ``unknown.csv`` to produce a prediction.
This prediction is then sent to the server which checks the prediction for correctness.
The performance is displayed back to the user, together with a message stating whether they passed the assignment or not.

.. note::
   If the server is unable to handle the request, it will display a message back to the user with the given error.
   Errors are also logged server side.
   If the student is unable to solve the error themselves, please contact `t.s.vanede@utwente.nl <mailto:t.s.vanede@utwente.nl>`_.

.. automethod:: ml4sec.Assignment.submit
