from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

################################################################################
#                                   Scaling                                    #
################################################################################

def scale(self, matrix, minimum=None, maximum=None):
    """Scale features in matrix.

        Parameters
        ----------
        matrix : np.array of shape=(n_samples, n_features)
            Matrix to scale.

        minimum : array-like of shape=(n_features,), optional
            If given, should contain the minimum values for each feature to
            scale with.

        maximum : array-like of shape=(n_features,), optional
            If given, should contain the maximum values for each feature to
            scale with.

        Returns
        -------
        scaled_matrix : array-like of shape=(n_samples, n_features)
            Scaled matrix.

        minimum : array-like of shape=(n_features,)
            The minimum values for each feature used for scaling.

        maximum : array-like of shape=(n_features,)
            The maximum values for each feature used for scaling.
        """
    raise NotImplementedError(
        "You will need to implement the 'scale()' function yourself."
    )

################################################################################
#                                Visualisation                                 #
################################################################################

def show_scaled(self):
    """Shows the scaled feature matrix for benign flows.

        1. Extracts the feature matrix.
        2. Scales the data.
        3. Turns it into a pd.DataFrame.

        """
    return pd.DataFrame(self.scale(self.feature_matrix(self.flows_benign))[0])

def plot_scaled(self):
    """Plot scaled vs unscaled.

        1. Computes feature matrix for both benign.csv and unknown.csv.
        2. Applies scaling to both benign and unknown feature matrix.
        3. Plots the differences.

        """
    # Get unscaled matrices
    X_unscaled_benign  = self.feature_matrix(self.flows_benign )
    X_unscaled_unknown = self.feature_matrix(self.flows_unknown)

    # Get scaled matrices
    X_scaled_benign , min_, max_ = self.scale(X_unscaled_benign)
    X_scaled_unknown, _, _ = self.scale(X_unscaled_unknown, min_, max_)

    # Create 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Create plots
    axs[0, 0].set_title("Unscaled data", fontsize=20)
    axs[0, 0].set_ylabel("Benign data", fontsize=20)
    self.plot_data(axs[0, 0], X_unscaled_benign, self.labels_benign)
    axs[0, 1].set_title("Scaled data", fontsize=20)
    self.plot_data(axs[0, 1], X_scaled_benign  , self.labels_benign)
    axs[0, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1, 0].set_ylabel("Unknown data", fontsize=20)
    self.plot_data(axs[1, 0], X_unscaled_unknown, self.labels_unknown)
    self.plot_data(axs[1, 1], X_scaled_unknown  , self.labels_unknown)
    axs[1, 1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Show plot
    plt.show()

def plot_data(self, ax, X, y):
    """Plots the feature vectors given by X and labelled by y.

        Parameters
        ----------
        ax : matplotlib.axis
            Axis on which to plot data.

        X : array-like of shape=(n_samples, n_features)
            Feature vectors to plot.

        y : array-like of shape=(n_features,)
            Labels corresponding to the feature vectors.
        """
    # Transform to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)

    # The PCA(n) takes all features from X and compresses them
    # into n dimensions, in our case to 2 dimensions for plotting.
    X = PCA(2).fit_transform(X)

    # Next for each label, we scatter the corresponding points
    for label in np.unique(y):
        ax.scatter(X[:, 0][y == label], X[:, 1][y == label], label=label)
