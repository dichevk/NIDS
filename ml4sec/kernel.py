from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
import numpy             as np
import warnings

################################################################################
#                                  Soft score                                  #
################################################################################

def soft_score(self, X, Y, K, **kwargs):
    """Compute the soft score by computing the kernel function
        between each point and each support_vector.
        The scores are then added to create a soft score.
        Note that this function is not weighted, in reality
        weights are added to adjust the importance of each
        support vector.

        Parameters
        ----------
        X : np.array of shape=(n_samples, n_features)
            Points for which to compute the soft score.

        Y : np.array of shape=(n_samples, n_features)
            Support vectors used to compute the score.

        K : func
            Kernel function to compute each individual score.

        **kwargs : optional
            Optional arguments for the kernel.

        Returns
        -------
        result : np.array of shape=(n_samples,)
            Soft score for each point in X.
        """
    # Get vectors as numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Compute scores between each point of X and each support_vector
    scores = pairwise_distances(X, Y, metric=K, **kwargs)

    # Compute the added scores
    return scores.sum(axis=1)

################################################################################
#                                Visualisation                                 #
################################################################################

def plot_kernels(self, K, sigmas=[None], thresholds=[None]):
    """Plot an example of randomly generated kernels.

        Parameters
        ----------
        K : func
            Kernel function to compute each individual score.

        sigmas : list, default=[None]
            List of different values of sigma to plot.

        thresholds : list, default=[None]
            List of different thresholds to plot.
        """
    # Create seeded randomness generator
    random = np.random.RandomState(1)
    # Create support vectors to plot
    support_vectors = np.concatenate((random.rand(3, 2) + [-3,  3],
                                      random.rand(3, 2) + [ 2, -3],
                                      random.rand(3, 2) + [ 4,  5]))

    # Create plots for different threshold and sigma values
    # Create n_thresholds x n_sigmas grid
    fig, axs = plt.subplots(len(thresholds), len(sigmas), figsize=(15, 10))
    # Reshape
    axs = np.asarray(axs).reshape(len(thresholds), len(sigmas))

    # Loop over given thresholds
    for j, threshold in enumerate(thresholds):
        # Loop over given sigmas
        for i, sigma in enumerate(sigmas):

            # Add sigma title if new column
            if j == 0 and sigma:
                axs[j, i].set_title(
                    "Sigma = {}".format(sigma),
                    fontsize = 20,
                )

            # Add threshold row if new row
            if i == 0 and threshold:
                axs[j, i].set_ylabel(
                    "Threshold = {}".format(threshold),
                    fontsize = 20,
                )

            # Plot axis
            self._plot_kernel_(
                ax              = axs[j, i],
                support_vectors = support_vectors,
                K               = K,
                threshold       = threshold or 1,
                sigma           = sigma,
            )

    # Show plot
    plt.show()



def _plot_kernel_(self, ax, support_vectors, K, threshold=2, **kwargs):
    """Function to plot scores and decision boundary for given support vectors.

        Parameters
        ----------
        support_vectors : np.array of shape=(n_vectors, n_features)
            Support vectors for which to plot the soft score.

        K : func
            Kernel function to compute the soft scores.

        threshold : float, default=2
            Decision threshold.
            All scores higher than this threshold will be inside the model.
            All scores lower than this threshold will are anomalous.

        **kwargs: optional
            Optional arguments to give to the kernel function K.
        """
    # Create meshgrid
    xx, yy = np.meshgrid(
        np.linspace(-10, 10, 30),
        np.linspace(-10, 10, 30),
    )

    # Compute decision function
    Z = self.soft_score(
        X = np.c_[xx.ravel(), yy.ravel()],
        Y = support_vectors,
        K = K,
        **kwargs
    ).reshape(xx.shape)

    # Ignore warnings
    warnings.simplefilter("ignore")

    # Plot smooth contour
    try:
        ax.contourf(xx, yy, Z,
            levels = np.linspace(Z.min()+.001, threshold, 7),
            cmap   = plt.cm.PuBu,
        )
    except: pass
    try:
        ax.contour(xx, yy, Z,
            levels     = [threshold],
            linewidths = 2,
            colors     = 'darkred'
        )
    except: pass
    try:
        ax.contourf(xx, yy, Z,
            levels = [threshold, Z.max()],
            colors = 'palevioletred',
        )
    except: pass

    # Restore warnings
    warnings.simplefilter("default")

    # Plot support vectors
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], c='black')
