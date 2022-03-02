import numpy as np

################################################################################
#                                    Split                                     #
################################################################################

def split(self, data, labels, apps_train, apps_test, ratio):
    """Extract features from each flow. Must be implemented by the students.

        Parameters
        ----------
        data : np.array of shape=(n_flows, n_features)
            Data for which to produce a split.

        labels : np.array of shape=(n_flows,)
            Labels corresponding to the data.

        apps_train : list()
            List of apps to use for training.

        apps_test : list()
            List of apps to use for testing.

        ratio : float
            Ratio of data for training apps to use for training.
            The rest should be used for testing.

        Returns
        -------
        X_train : array-like of shape=(n_flows_train, n_features)
            Datapoints selected for training.

        y_train : array-like of shape=(n_flows_train,)
            Corresponding labels selected for training.

        X_test : array-like of shape=(n_flows_test, n_features)
            Datapoints selected for testing.

        y_test : array-like of shape=(n_flows_test,)
            Corresponding labels selected for testing.
        """
    


def get_split(self, apps_train, ratio):
    """Returns the split of 'benign.csv' file for a given input.

        Parameters
        ----------
        apps_train : list()
            Apps to use in the training set, all apps not in apps_train will be
            used only for the testing set.

        ratio : float
            Ratio of datapoints for apps in app_train to use for training.

        Returns
        -------
        X_train : array-like of shape=(n_samples_train, n_features)
            Datapoints used for training.

        y_train : array-like of shape=(n_samples_train,)
            Labels corresponding to training data.
            +1 if the corresponding app is     in the training dataset,
            -1 if the corresponding app is not in the training dataset,
            I.e., all values are +1.

        X_test : array-like of shape=(n_samples_test, n_features)
            Datapoints used for testing.

        y_test : array-like of shape=(n_samples_test,)
            Labels corresponding to testing data.
            +1 if the corresponding app is     in the training dataset,
            -1 if the corresponding app is not in the training dataset.
        """
    # Get relevant data and labels
    data, _, _ = self.scale(self.feature_matrix(self.flows_benign))
    labels     = self.labels_benign

    # Compute apps in test set
    apps_test = list(set(list(np.unique(labels))) - set(apps_train))

    # Execute split
    X_train, y_train, X_test, y_test = map(np.asarray, self.split(
        data       = data,
        labels     = labels,
        apps_train = apps_train,
        apps_test  = apps_test,
        ratio      = ratio,
    ))

    # Transform labels to -1/+1 depending on whether they are inside (+1)
    # or outside (-1) of the training set
    y_train = np.ones(y_train.shape[0], dtype=int)
    y_test  = np.isin(y_test, apps_train).astype(int) * 2 - 1

    # Return result
    return X_train, y_train, X_test, y_test


################################################################################
#                                 Test methods                                 #
################################################################################

def test_split(self, verbose=False):
    """Executes several test cases for splitting the data.

        Parameters
        ----------
        verbose : boolean, default=False
            If True, print tests that are executed.

        """
    # Select data
    data, _, _ = self.scale(self.feature_matrix(self.flows_benign))
    labels     = self.labels_benign

    # Test cases
    all_apps_train = [
        ['Anti Virus', 'Chrome', 'DNS Server', 'Excel', 'Firefox', 'Git', 'IntelliJ'],
        ['Anti Virus', 'Outlook', 'DNS Server', 'Excel', 'Firefox', 'Git', 'IntelliJ', 'PowerPoint', 'Thunderbird', 'Web Server', 'Word'],
        ['IntelliJ', 'PowerPoint', 'Thunderbird', 'Web Server', 'Word'],
        ['IntelliJ', 'PowerPoint', 'Thunderbird', 'Web Server', 'Word', 'Mail Server', 'Chrome', 'Anti Virus', 'Outlook', 'DNS Server', 'Excel', 'Firefox', 'Git'],
        [],
    ]

    all_apps_test = [
        ['Mail Server', 'Outlook', 'PowerPoint', 'Thunderbird', 'Web Server', 'Word'],
        ['Mail Server', 'Chrome'],
        ['Mail Server', 'Chrome', 'Anti Virus', 'Outlook', 'DNS Server', 'Excel', 'Firefox', 'Git'],
        [],
        ['IntelliJ', 'PowerPoint', 'Thunderbird', 'Web Server', 'Word', 'Mail Server', 'Chrome', 'Anti Virus', 'Outlook', 'DNS Server', 'Excel', 'Firefox', 'Git'],
    ]

    all_ratios   = [0.5, 0.8, 0.25, 0.9, 0.0]
    all_expected = [4210, 8786, 784, 11330, 0]

    for case, (apps_train, apps_test, ratio, expected) in enumerate(
            zip(all_apps_train, all_apps_test, all_ratios, all_expected)
        ):
        # Print test case
        if verbose:
            print("Test case {} - benign.csv:".format(case + 1))
            print("apps_train = {}".format(apps_train))
            print("apps_test  = {}".format(apps_test ))
            print("ratio      = {}".format(ratio))
            print()

        # Execute split
        data_train, labels_train, data_test, labels_test = self.split(
            data       = data,
            labels     = labels,
            apps_train = apps_train,
            apps_test  = apps_test,
            ratio      = ratio,
        )

        # Test results
        self.assert_split(
            original_data          = data,
            original_labels        = labels,
            data_train             = data_train,
            labels_train           = labels_train,
            data_test              = data_test,
            labels_test            = labels_test,
            expected_samples_train = expected,
            apps_test              = apps_test,
        )

    # Confirm that all cases passed
    print("All test cases passed.")



def assert_split(self, original_data, original_labels,
                 data_train, labels_train, data_test, labels_test,
                 expected_samples_train, apps_test
    ):
    """Tests whether the split is correct.
        If an incorrect split is found, it raises an error with a suggestion for
        the students to debug their code.

        Parameters
        ----------
        original_data : np.array of shape=(n_samples, n_features)
            Original data before split.

        original_labels : np.array of shape=(n_samples, n_features)
            Original labels before split.

        data_train : np.array of shape=(n_samples_train, n_features)
            Data selected for training.

        labels_train : np.array of shape=(n_samples_train,)
            Corresponding labels selected for training.

        data_test : np.array of shape=(n_samples_test, n_features)
            Data selected for testing.

        labels_test : np.array of shape=(n_samples_test,)
            Corresponding labels selected for testing.

        expected_samples_train : int
            Number of samples expected to be in training.

        apps_test : list
            List of apps that should only appear in testing dataset.
        """
    # Try casting
    try:
        original_data = np.asarray(original_data)
        data_train    = np.asarray(data_train, dtype=float)
        labels_train  = np.asarray(labels_train)
        data_test     = np.asarray(data_test , dtype=float)
        labels_test   = np.asarray(labels_test )
    except Exception as e:
        raise ValueError("Output could not be cast to array: '{}'".format(e))

    # Check if apps in original are same as in train+test
    if not (np.unique(original_labels) ==
            np.unique(np.concatenate((labels_train, labels_test)))
           ).all():
        raise ValueError(
            "Labels in train and test were different from original."
        )

    # Check if apps in original are same as in train+test
    if np.isin(labels_train, apps_test).any():
        raise ValueError("Training data contains test labels.")

    # Check if number of train datapoints and labels correspond
    if data_train.shape[0] != labels_train.shape[0]:
        raise ValueError(
            "Number of train datapoints is not equal to number of train labels."
        )

    # Check if number of test datapoints and labels correspond
    if data_test .shape[0] != labels_test .shape[0]:
        raise ValueError(
            "Number of test datapoints is not equal to number of test labels."
        )

    # Check if number of train+test datapoints equals original
    if data_train.shape[0] + data_test.shape[0] != original_data.shape[0]:
        raise ValueError("Train data + test data != original data.")

    # Check if number of features is equal to original
    if  data_train.shape[0]  != 0 and\
        (data_train.shape[1] != data_test.shape[1] or\
         data_train.shape[1] != original_data.shape[1]):
        raise ValueError("A different number of features was returned.")

    # Check if labels only has single dimension
    if labels_train.ndim != 1 or labels_test.ndim != 1:
        raise ValueError("Labels should have one dimension.")

    # Check if number of train datapoints is expected
    if data_train.shape[0] != expected_samples_train:
        raise ValueError(
            "Number of training datapoints should be '{}', but was '{}'. Hint: "
            "if you are off by 1, try to round the ratio instead of taking the "
            "floor or ceiling value."
            .format(expected_samples_train, data_train.shape[0])
        )

    expected_test = original_data.shape[0] - expected_samples_train
    # Check if number of test datapoints is expected
    if data_test.shape[0] != expected_test:
        raise ValueError(
            "Number of testing datapoints should be '{}', but was '{}'. Hint: "
            "if you are off by 1, try to round the ratio instead of taking the "
            "floor or ceiling value."
            .format(data_test.shape[0], expected_test)
        )
