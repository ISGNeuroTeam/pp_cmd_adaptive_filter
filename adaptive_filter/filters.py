import numpy as np
from typing import Union, Tuple
from .utils import get_mean_error


class Filter:
    """
    Base class for adaptive filter algorithm
    """

    def __init__(self, filter_size, mu=0.99, eps=0.1, weights="random"):
        self.weights = weights
        self.filter_size = filter_size
        self.mu = mu
        self.eps = eps

    def init_weights(self, weights: Union[np.ndarray, str], filter_size: int = -1):
        """
        Initializes the adaptive weights of the filter
        Args:
            weights: initial weights of filter. Possible values are:
                    * array with initial weights (1-dimensional array) of filter size
                    * "random": create random weights
                    * "zeros": create zero value weights
            filter_size: number of filter coefficients
        """
        if filter_size == -1:
            filter_size = self.filter_size

        if type(weights) == str:
            if weights == "random":
                weights = np.random.normal(0, 0.5, filter_size)
            elif weights == "zeros":
                weights = np.zeros(filter_size)
            else:
                raise ValueError("Impossible to understand the weights")
        elif len(weights) == filter_size:
            weights = np.array(weights, dtype=np.float64)
        else:
            raise ValueError("Impossible to understand the weights")
        self.weights = weights

    def run(self, desired_signal: np.ndarray, input_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Base function run
        :param desired_signal:
        :param input_matrix:
        :return:
        """
        output_signal = errors = weight_history = np.zeros_like(desired_signal)
        return output_signal, errors, weight_history

    def pretrained_run(self, desired_signal: np.ndarray, input_signal: np.ndarray, ntrain: float = 0.5,
                       epochs: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function sacrifices part of the data for few epochs of learning.
        :param desired_signal: desired signal
        :param input_signal: input_signal
        :param ntrain: train to test ratio, default value is 0.5
                       (that means 50% of data is used for training)
        :param epochs: number of training epochs (int), default value is 1.
                       This number describes how many times the training will be repeated
                       on dedicated part of data.
        :return: output_signal: output value (1 dimensional array).
                           The size corresponds with the desired signal.
                 errors: filter error for every sample
                 history_weights: history of all weights

        """
        n_train = int(len(desired_signal) * ntrain)
        # train
        for epoch in range(epochs):
            self.run(desired_signal[:n_train], input_signal[:n_train])

        # test
        output_signal, errors, history_weights = self.run(desired_signal[n_train:], input_signal[n_train:])
        return output_signal, errors, history_weights

    def explore_learning(self, desired_signal: np.ndarray, input_signal: np.ndarray, mu_start: float = 0.,
                         mu_end: float = 1., steps: int = 100, ntrain: float = 0.5, epochs: int = 1,
                         criteria: str = "MSE") -> Tuple[np.ndarray, np.ndarray]:
        """
        Test what learning rate is the best.
        :param desired_signal: desired signal
        :param input_signal: input_signal
        :param mu_start: starting learning rate
        :param mu_end: final learning rate
        :param steps: how many learning rates should be tested between `mu_start`
                      and `mu_end`
        :param ntrain: train to test ratio, default value is 0.5
                       (that means 50% of data is used for training)
        :param epochs: number of training epochs (int), default value is 1.
                       This number describes how many times the training will be repeated
                       on dedicated part of data.
        :param criteria: how should be measured the mean error (str),
                         default value is "MSE".
        :return: errors: mean error for tested learning rates
        :return: mu_range: range of used learning rates
        """
        mu_range = np.linspace(mu_start, mu_end, steps)
        print(mu_range.shape)
        errors = np.zeros(len(mu_range))
        for i, mu in enumerate(mu_range):
            # init
            self.init_weights("zeros")
            # if mu == 0:
            #     self.mu = 0.01
            # else:
            self.mu = mu
            # run
            output_signal, err, history_weights = self.pretrained_run(desired_signal, input_signal, ntrain=ntrain,
                                                                         epochs=epochs)
            errors[i] = get_mean_error(err, function=criteria)
        return errors, mu_range

class FilterLMS(Filter):
    """
    Adaptive LMS filter
    """
    def __init__(
            self,
            filter_size: int,
            mu: float = 0.01,
            weights: Union[np.ndarray, str] = "random",
    ):
        """
        Args:
            filter_size: length of filter
            mu: learning rate.
            weights: initial weights of filter
        """
        super().__init__(filter_size=filter_size, mu=mu, weights=weights)
        if type(filter_size) == int:
            self.filter_size = filter_size
        else:
            raise ValueError("The size of filter must be an integer")

        self.mu = mu
        self.init_weights(weights, self.filter_size)
        self.w_history = False

    def run(
        self, desired_signal: np.ndarray, input_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filters multiple samples in a row.

        Args:
            desired_signal: desired signal (1-dimensional array)
            input_matrix: input_matrix (2-dimensional array). Rows are samples, columns are input arrays
        Returns:
            output_signal: output value (1 dimensional array).
                           The size corresponds with the desired signal.
            errors: filter error for every sample
            history_weights: history of all weights
        """
        n = len(input_matrix)
        if not len(desired_signal) == n:
            raise ValueError("The length of vector d and matrix x must agree.")
        # self.filter_size = len(input_matrix[0])
        # prepare data
        input_matrix = np.array(input_matrix)
        desired_signal = np.array(desired_signal)
        # create empty arrays
        output_signal = np.zeros(n)
        errors = np.zeros(n)
        self.w_history = np.zeros((n, len(input_matrix[0])))
        # adaptation loop
        for k in range(n):
            self.w_history[k, :] = self.weights
            output_signal[k] = np.dot(self.weights, input_matrix[k])
            errors[k] = desired_signal[k] - output_signal[k]
            dw = self.mu * errors[k] * input_matrix[k]
            self.weights += dw
        return output_signal, errors, self.w_history


class FilterNLMS(Filter):
    """
    Adaptive LMS filter
    """
    def __init__(
            self,
            filter_size: int,
            mu: float = 0.01,
            weights: Union[np.ndarray, str] = "random",
    ):
        """
        Args:
            filter_size: length of filter
            mu: learning rate.
            weights: initial weights of filter
        """
        super().__init__(filter_size=filter_size, mu=mu, weights=weights)
        if type(filter_size) == int:
            self.filter_size = filter_size
        else:
            raise ValueError("The size of filter must be an integer")

        self.mu = mu
        self.init_weights(weights, self.filter_size)
        self.w_history = False

    def run(
        self, desired_signal: np.ndarray, input_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filters multiple samples in a row.

        Args:
            desired_signal: desired signal (1-dimensional array)
            input_matrix: input_matrix (2-dimensional array). Rows are samples, columns are input arrays
        Returns:
            output_signal: output value (1 dimensional array).
                           The size corresponds with the desired signal.
            errors: filter error for every sample
            history_weights: history of all weights
        """
        n = len(input_matrix)
        if not len(desired_signal) == n:
            raise ValueError("The length of vector d and matrix x must agree.")
        self.filter_size = len(input_matrix[0])
        # prepare data
        input_matrix = np.array(input_matrix)
        desired_signal = np.array(desired_signal)
        # create empty arrays
        output_signal = np.zeros(n)
        errors = np.zeros(n)
        self.w_history = np.zeros((n, self.filter_size))
        # adaptation loop
        for k in range(n):
            self.w_history[k, :] = self.weights
            output_signal[k] = np.dot(self.weights, input_matrix[k])
            errors[k] = desired_signal[k] - output_signal[k]
            nu = self.mu / (self.eps + np.dot(input_matrix[k], input_matrix[k]))
            dw = nu * errors[k] * input_matrix[k]
            self.weights += dw
        return output_signal, errors, self.w_history


class FilterRLS(Filter):
    """
    Adaptive RLS filter
    """

    def __init__(
        self,
        filter_size: int,
        mu: float = 0.99,
        eps: float = 0.1,
        weights: Union[np.ndarray, str] = "random",
    ):
        """
        Args:
            filter_size: length of filter
            mu: forgetting factor. It is introduced to give exponentially
                less weight to older error samples. It is usually chosen
                between 0.98 and 1.
            eps: initialisation value (float). It is usually chosen
                 between 0.1 and 1.
            weights: initial weights of filter
        """
        super().__init__(filter_size=filter_size, mu=mu, eps=eps, weights=weights)
        if type(filter_size) == int:
            self.filter_size = filter_size
        else:
            raise ValueError("The size of filter must be an integer")

        # if mu == 0:
        #     self.mu = 0.01
        # else:
        self.mu = mu
        self.eps = eps
        self.init_weights(weights, self.filter_size)
        self.r = 1 / self.eps * np.identity(filter_size)
        self.w_history = False

    def run(
        self, desired_signal: np.ndarray, input_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filters multiple samples in a row.

        Args:
            desired_signal: desired signal (1-dimensional array)
            input_matrix: input_matrix (2-dimensional array). Rows are samples, columns are input arrays
        Returns:
            output_signal: output value (1 dimensional array).
                           The size corresponds with the desired signal.
            errors: filter error for every sample
            history_weights: history of all weights
        """
        n = len(input_matrix)
        if not len(desired_signal) == n:
            raise ValueError("The length of vector d and matrix x must agree.")
        self.filter_size = len(input_matrix[0])
        # prepare data
        input_matrix = np.array(input_matrix)
        desired_signal = np.array(desired_signal)
        # create empty arrays
        output_signal = np.zeros(n)
        errors = np.zeros(n)
        self.w_history = np.zeros((n, self.filter_size))
        # adaptation loop
        for k in range(n):
            self.w_history[k, :] = self.weights
            output_signal[k] = np.dot(self.weights, input_matrix[k])
            errors[k] = desired_signal[k] - output_signal[k]
            r_1 = np.dot(np.dot(np.dot(self.r, input_matrix[k]), input_matrix[k].T), self.r)
            r_2 = self.mu + np.dot(np.dot(input_matrix[k], self.r), input_matrix[k].T)
            self.r = 1 / self.mu * (self.r - r_1 / r_2)
            dw = np.dot(self.r, input_matrix[k].T) * errors[k]
            self.weights += dw
        return output_signal, errors, self.w_history
