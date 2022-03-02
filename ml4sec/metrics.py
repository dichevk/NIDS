import numpy as np

################################################################################
#                  Metrics to be implemented by the students                   #
################################################################################

def TP(self, y_true, y_pred):
    """Computes the number of True Positives for given y_true and y_pred.

        Parameters
        ----------
        y_true : array-like of shape=(n_samples,)
            True values corresponding to each prediction.

        y_pred : array-like of shape=(n_samples,)
            Predicted values corresponding to each true value.

        Returns
        -------
        result : int
            Number of True Positives.
        """
    raise NotImplementedError(
        "You will need to implement the 'TP()' function yourself."
    )

def TN(self, y_true, y_pred):
    """Computes the number of True Negatives for given y_true and y_pred.

        Parameters
        ----------
        y_true : array-like of shape=(n_samples,)
            True values corresponding to each prediction.

        y_pred : array-like of shape=(n_samples,)
            Predicted values corresponding to each true value.

        Returns
        -------
        result : int
            Number of True Negatives.
        """
    raise NotImplementedError(
        "You will need to implement the 'TN()' function yourself."
    )

def FP(self, y_true, y_pred):
    """Computes the number of False Positives for given y_true and y_pred.

        Parameters
        ----------
        y_true : array-like of shape=(n_samples,)
            True values corresponding to each prediction.

        y_pred : array-like of shape=(n_samples,)
            Predicted values corresponding to each true value.

        Returns
        -------
        result : int
            Number of False Positives.
        """
    raise NotImplementedError(
        "You will need to implement the 'FP()' function yourself."
    )

def FN(self, y_true, y_pred):
    """Computes the number of False Negatives for given y_true and y_pred.

        Parameters
        ----------
        y_true : array-like of shape=(n_samples,)
            True values corresponding to each prediction.

        y_pred : array-like of shape=(n_samples,)
            Predicted values corresponding to each true value.

        Returns
        -------
        result : int
            Number of False Negatives.
        """
    raise NotImplementedError(
        "You will need to implement the 'FN()' function yourself."
    )

################################################################################
#             Metrics based on True/False Positive/Negative values             #
################################################################################

def TPR(self, y_true, y_pred):
    """True positive rate."""
    tp = self.TP(y_true, y_pred)
    fn = self.FN(y_true, y_pred)
    return tp / (tp + fn)

def TNR(self, y_true, y_pred):
    """True negative rate."""
    tn = self.TN(y_true, y_pred)
    fp = self.FP(y_true, y_pred)
    return tn / (tn + fp)

def FPR(self, y_true, y_pred):
    """False positive rate."""
    fp = self.FP(y_true, y_pred)
    tn = self.TN(y_true, y_pred)
    return fp / (tn + fp)

def FNR(self, y_true, y_pred):
    """False negative rate."""
    fn = self.FN(y_true, y_pred)
    tp = self.TP(y_true, y_pred)
    return fn / (tp + fn)

def accuracy(self, y_true, y_pred):
    """Accuracy."""
    tp = self.TP(y_true, y_pred)
    tn = self.TN(y_true, y_pred)
    fp = self.FP(y_true, y_pred)
    fn = self.FN(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)

def precision(self, y_true, y_pred):
    """Precision."""
    tp = self.TP(y_true, y_pred)
    fp = self.FP(y_true, y_pred)
    return tp / (tp + fp)

def recall(self, y_true, y_pred):
    """Recall."""
    tp = self.TP(y_true, y_pred)
    fn = self.FN(y_true, y_pred)
    return tp / (tp + fn)

def f1_score(self, y_true, y_pred):
    """F1-score."""
    tp = self.TP(y_true, y_pred)
    fp = self.FP(y_true, y_pred)
    fn = self.FN(y_true, y_pred)
    return 2*tp / (2*tp + fp + fn)

################################################################################
#                                Visualisation                                 #
################################################################################
def prediction_report(self, y_true, y_pred):
    """Computes prediction metrics and prints them.

        Parameters
        ----------
        y_true : array-like of shape=(n_samples,)
            True values corresponding to each prediction.

        y_pred : array-like of shape=(n_samples,)
            Predicted values corresponding to each true value.

        """
    self.show_report(
        self.TPR     (y_true, y_pred),
        self.TNR     (y_true, y_pred),
        self.FPR     (y_true, y_pred),
        self.FNR     (y_true, y_pred),
        self.accuracy(y_true, y_pred),

        self.precision(y_true, y_pred),
        self.recall   (y_true, y_pred),
        self.f1_score (y_true, y_pred),
    )

def show_report(self, tpr, tnr, fpr, fnr, accuracy, precision, recall, f1_score):
    """Prints prediction report.

        Parameters
        ----------
        tpr : float
            True Positive Rate to show.

        tnr : float
            True Negative Rate to show.

        fpr : float
            False Positive Rate to show.

        fnr : float
            False Negative Rate to show.

        accuracy : float
            Accuracy to show.

        precision : float
            Precision to show.

        recall : float
            Recall to show.

        f1_score : float
            F1-score to show.
        """
    # Print report
    print("""
Prediction report
-----------------
True  Positive Rate (TPR) = {:.6f}
True  Negative Rate (TNR) = {:.6f}
False Positive Rate (FPR) = {:.6f}
False Negative Rate (FNR) = {:.6f}
Accuracy            (ACC) = {:.6f}

Precision                 = {:.6f}
Recall                    = {:.6f}
F1-score                  = {:.6f}
""".format(tpr, tnr, fpr, fnr, accuracy, precision, recall, f1_score))

################################################################################
#                                 Test methods                                 #
################################################################################

def test_metrics(self, verbose=False):
    """Test whether the computed metrics are correct."""
    # Test case 1
    y_pred = np.array([-1, -1, 1, 1, -1, 1, 1, -1, 1, -1], dtype=int)
    y_true = np.array([-1, -1, -1, 1, 1, 1, -1, 1, -1, 1], dtype=int)
    if verbose:
        print("Test case 1:")
        print("y_pred: {}".format(y_pred))
        print("y_true: {}".format(y_true))
    assert self.TP(y_true, y_pred) == 2
    assert self.TN(y_true, y_pred) == 2
    assert self.FP(y_true, y_pred) == 3
    assert self.FN(y_true, y_pred) == 3
    assert self.TP(y_true, y_pred) + self.TN(y_pred, y_true) + self.FP(y_pred, y_true) + self.FN(y_pred, y_true) == y_pred.shape[0]

    # Test case 2
    y_pred = np.array([-1, 1, -1, 1, -1, 1, -1, 1, 1, -1], dtype=int)
    y_true = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=int)
    if verbose:
        print("Test case 2:")
        print("y_pred: {}".format(y_pred))
        print("y_true: {}".format(y_true))
    assert self.TP(y_true, y_pred) == 4
    assert self.TN(y_true, y_pred) == 4
    assert self.FP(y_true, y_pred) == 1
    assert self.FN(y_true, y_pred) == 1
    assert self.TP(y_true, y_pred) + self.TN(y_pred, y_true) + self.FP(y_pred, y_true) + self.FN(y_pred, y_true) == y_pred.shape[0]

    # Test case 3
    y_pred = np.array([-1, 1, -1, 1, -1, 1, -1, -1, -1, -1], dtype=int)
    y_true = np.array([-1, 1, -1, 1, -1, 1, -1, 1, 1, -1], dtype=int)
    if verbose:
        print("Test case 3:")
        print("y_pred: {}".format(y_pred))
        print("y_true: {}".format(y_true))
    assert self.TP(y_true, y_pred) == 5
    assert self.TN(y_true, y_pred) == 3
    assert self.FP(y_true, y_pred) == 2
    assert self.FN(y_true, y_pred) == 0
    assert self.TP(y_true, y_pred) + self.TN(y_pred, y_true) + self.FP(y_pred, y_true) + self.FN(y_pred, y_true) == y_pred.shape[0]

    # Test case 4
    y_pred = np.array([-1, 1, -1, 1, -1, 1, -1, 1, 1, -1], dtype=int)
    y_true = np.array([-1, 1, -1, 1, -1, 1, -1, -1, -1, -1], dtype=int)
    if verbose:
        print("Test case 4:")
        print("y_pred: {}".format(y_pred))
        print("y_true: {}".format(y_true))
    assert self.TP(y_true, y_pred) == 5
    assert self.TN(y_true, y_pred) == 3
    assert self.FP(y_true, y_pred) == 0
    assert self.FN(y_true, y_pred) == 2
    assert self.TP(y_true, y_pred) + self.TN(y_pred, y_true) + self.FP(y_pred, y_true) + self.FN(y_pred, y_true) == y_pred.shape[0]

    # Test case 5
    y_pred = np.array([-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], dtype=int)
    y_true = np.array([-1, -1, -1, 1, 1, 1, 1, 1, -1, -1], dtype=int)
    if verbose:
        print("Test case 5:")
        print("y_pred: {}".format(y_pred))
        print("y_true: {}".format(y_true))
    assert self.TP(y_true, y_pred) == 3
    assert self.TN(y_true, y_pred) == 5
    assert self.FP(y_true, y_pred) == 0
    assert self.FN(y_true, y_pred) == 2
    assert self.TP(y_true, y_pred) + self.TN(y_pred, y_true) + self.FP(y_pred, y_true) + self.FN(y_pred, y_true) == y_pred.shape[0]

    print("All test cases passed.")
