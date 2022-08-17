import numpy as np


def input_preprocess(
    input_array: np.ndarray, matrix_size: int, bias: bool = False
) -> np.ndarray:
    """
    Creates the input matrix
    Args:
        input_array: 1 dimensional input array
        matrix_size: size of input matrix row It means how many samples \
                     of previous history you want to use \
                     as the filter input. It also represents the filter length.
        bias: decides if the bias is used (Boolean). If True, \
              array of all ones is appended as a last column to matrix `input_matrix`. \
              So matrix `input_matrix` has `n`+1 columns.
    Returns:
        input_matrix: input_matrix (2-dimensional array) \
                      constructed from an array `input_array`. The length of `input_matrix` \
                      is calculated as length of `input_array` - `matrix_size` + 1. \
                      If the `bias` is used, then the amount of columns is `matrix_size` if not then \
                      amount of columns is `matrix_size`+1).
    """
    if not type(matrix_size) == int:
        raise ValueError("The argument n must be int.")
    if not matrix_size > 0:
        raise ValueError("The argument n must be greater than 0")
    try:
        input_array = np.array(input_array, dtype="float64")
    except:
        raise ValueError("The argument a is not numpy array or similar.")
    input_matrix = np.array(
        [
            input_array[i : i + matrix_size]
            for i in range(len(input_array) - matrix_size + 1)
        ]
    )
    if bias:
        input_matrix = np.vstack((input_matrix.T, np.ones(len(input_matrix)))).T
    return input_matrix



def MAE(x1: np.ndarray) -> float:
    """
    Mean absolute error
    :param x1: series or error
    :return: MAE
    """
    error = np.array(x1)
    return np.sum(np.abs(error)) / float(len(error))

def MSE(x1: np.ndarray) -> float:
    """
    Mean squared error
    :param x1: series or error
    :return: MSE
    """
    error = np.array(x1)
    return np.dot(error, error) / float(len(error))

def RMSE(x1: np.ndarray) -> float:
    """
    Root-mean-square error
    :param x1: series or error
    :return: RMSE
    """
    error = np.array(x1)
    return np.sqrt(np.dot(error, error) / float(len(error)))


def get_mean_error(x1: np.ndarray, function: str = "MSE") -> float:
    """
    This function returns desired mean error. Options are: MSE, MAE, RMSE
    :param x1: series or error
    :param function:
    :return: mean error value
    """
    if function == "MSE":
        return MSE(x1)
    elif function == "MAE":
        return MAE(x1)
    elif function == "RMSE":
        return RMSE(x1)
    else:
        raise ValueError('The provided error function is not known')