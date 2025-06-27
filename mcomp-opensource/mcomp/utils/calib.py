## scipy hadamard
import torch
import numpy as np
import math

def hadamard(n, dtype=int):
    """
    Construct an Hadamard matrix.

    Constructs an n-by-n Hadamard matrix, using Sylvester's
    construction. `n` must be a power of 2.

    Parameters
    ----------
    n : int
        The order of the matrix. `n` must be a power of 2.
    dtype : dtype, optional
        The data type of the array to be constructed.

    Returns
    -------
    H : (n, n) ndarray
        The Hadamard matrix.

    Notes
    -----
    .. versionadded:: 0.8.0

    Examples
    --------
    >>> from scipy.linalg import hadamard
    >>> hadamard(2, dtype=complex)
    array([[ 1.+0.j,  1.+0.j],
           [ 1.+0.j, -1.-0.j]])
    >>> hadamard(4)
    array([[ 1,  1,  1,  1],
           [ 1, -1,  1, -1],
           [ 1,  1, -1, -1],
           [ 1, -1, -1,  1]])

    """

    # This function is a slightly modified version of the
    # function contributed by Ivo in ticket #675.

    if n < 1:
        lg2 = 0
    else:
        lg2 = int(math.log(n, 2))
    if 2 ** lg2 != n:
        raise ValueError("n must be an positive integer, and n must be "
                         "a power of 2")

    H = np.array([[1]])

    # Sylvester's construction
    for i in range(0, lg2):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

    H = torch.Tensor(H).to(dtype)
    return H