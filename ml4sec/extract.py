import numpy  as np
import pandas as pd

################################################################################
#               Methods used for extracting features from flows.               #
################################################################################

def extract(self, protocol, src, sport, dst, dport, timestamps, sizes):
    """Extract features from each flow. Must be implemented by the students.

        Parameters
        ----------
        protocol : string
            Either 'TCP' or 'UDP'.

        src : string
            Source IP address as string.

        sport : int
            Source port.

        dst : string
            Destination IP address as string.

        dport : int
            Destination port.

        timestamps : np.array of shape=(n_packets,)
            Timestamp for each packet in the flow.

        sizes : np.array of shape=(n_packets,)
            Size of each packet in the flow.

        Returns
        -------
        features : array-like of shape=(n_features,)
            Extracted features, determined by the student.
        """
    raise NotImplementedError(
        "You will need to implement the 'extract()' function yourself."
    )



def feature_matrix(self, data):
    """Computes the feature matrix for the implemented 'extract' method.

        Parameters
        ----------
        data : iterator
            Iterator over flow-tuple, flow-dataframe.

        Returns
        -------
        feature_matrix : np.array of shape=(n_flows, n_features)
            Feature matrix for given data.
        """
    # Initialise matrix
    matrix = list()

    # Loop over all flows
    for (protocol, src, sport, dst, dport), frame in data:
        # Call extract function
        features = self.extract(
            protocol   = protocol,
            src        = src,
            sport      = sport,
            dst        = dst,
            dport      = dport,
            timestamps = frame['timestamp'].values,
            sizes      = frame['size'     ].values,
        )
        # Append to matrix
        matrix.append(features)

    # Return matrix as pandas dataframe
    matrix = np.asarray(matrix, dtype=float)

    # Assertions over matrix
    assert matrix.shape[0] > 0
    assert not np.isnan(matrix).any()

    # Return result
    return matrix



################################################################################
#                                Visualisation                                 #
################################################################################

def show_matrix(self):
    """Shows the feature matrix for benign flows.

        1. Extracts the feature matrix
        2. Turns it into a pd.DataFrame.

        """
    return pd.DataFrame(self.feature_matrix(self.flows_benign))



################################################################################
#                                 Test methods                                 #
################################################################################

def test_extract(self, maximum=float('inf')):
    """Tests the extract function implemented by students.

        Parameters
        ----------
        maximum : float, default='inf'
            Maximum number of flows to test.
        """

    # Test everything if a non-positive value is given
    if maximum < 0: maximum = float('inf')

    # Loop over all flows
    for index, ((protocol, src, sport, dst, dport), frame) in enumerate(self.flows_benign):

        # Break if necessary
        if index >= maximum: break

        # Call extract function
        features = self.extract(
            protocol   = protocol,
            src        = src,
            sport      = sport,
            dst        = dst,
            dport      = dport,
            timestamps = frame['timestamp'].values,
            sizes      = frame['size'     ].values,
        )

        # Check if anything is returned
        if features is None:
            raise ValueError(
                "You did not return any value"
            )

        # Cast to numpy array
        try:
            features = np.asarray(features, dtype=float)
        except Exception as e:
            raise ValueError(
                "{}\nHint: Only return numerical features.".format(e)
            )

        # Check for NaN values
        if np.isnan(features).any():
            raise ValueError(
                "You have NaN values in your features. This error is "
                "thrown when values could not be parsed as numbers.\n Hint:"
                " Check for non-numerical values and divide by 0 errors."
            )

        # Check if features are computed:
        if features.shape[0] == 0:
            raise ValueError(
                "You returned an empty array."
            )

        # Print test overview
        print("Flow ({}, {}, {}, {}, {}):".format(
            protocol,
            src,
            sport,
            dst,
            dport,
        ))
        print("\ttimestamps        : {}".format(frame['timestamp'].values))
        print("\tsizes             : {}".format(frame['size'     ].values))
        print("\tExtracted features: {}".format(np.asarray(features)))
        print()
