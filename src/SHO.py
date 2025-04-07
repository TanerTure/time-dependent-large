"""This SHO.py module generates the matrix form for several common operators in the energy eigenbasis of the harmonic oscillator.

Functions within can generate :math:`\\hat{x}, \\hat{p}, \\hat{a}^+, \\hat{a}, \\hat{H}`, and second moments of :math:`\\hat{x}` and :math:`\\hat{p}\\ (\\hat{x}^2,\\hat{x}\\hat{p}`, :math:`\\hat{p}\\hat{x}, \\hat{p}^2)`.

Example
-------
>>> import SHO
>>> SHO.a(3)
array([[0.        +0.j, 1.        +0.j, 0.        +0.j],
       [0.        +0.j, 0.        +0.j, 1.41421356+0.j],
       [0.        +0.j, 0.        +0.j, 0.        +0.j]])   
>>> SHO.a_dagger(3)
array([[0.        +0.j, 0.        +0.j, 0.        +0.j],
       [1.        +0.j, 0.        +0.j, 0.        +0.j],
       [0.        +0.j, 1.41421356+0.j, 0.        +0.j]])

The matrices above are used to calculate the energy using the relation :math:`H = a^+a + 1/2`

>>> import numpy as np
>>> SHO.a_dagger(3) @ SHO.a(3) + 0.5 * np.eye(3)
array([[0.5+0.j, 0. +0.j, 0. +0.j],
       [0. +0.j, 1.5+0.j, 0. +0.j],
       [0. +0.j, 0. +0.j, 2.5+0.j]])
        
"""
import numpy as np
import sympy as sp


def a(n=100):
    """
    Make the matrix representation of the lowering operator :math:`a`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrix

    Returns
    -------
    out : complex ndarray
        matrix of size n by n that represents the lowering operator :math:`\hat{a}`

    Raises
    ------
    ValueError
        If n is less than or equal to 1

    See Also
    --------
    SHO : overall description of module
    a_dagger : raising operator

    Notes
    -----
    The matrix :math:`\\hat{a}` in the eigenbasis of the Hamiltonian reads:
    
    .. math::

        \\hat{a} = \\begin{bmatrix}
            0 & \\sqrt{1} & 0 \\\\ 
            0 & 0        & \\sqrt{2} \\\\
            0 &   0      & 0 \\\\
            \\end{bmatrix}

    """
    if n <= 1:
        raise ValueError("n should be 2 or greater")
    matrix = np.zeros((n, n), dtype=np.complex128)
    for i in range(n - 1):
        matrix[i, i + 1] = sp.sqrt(i + 1)
    return matrix


def a_dagger(n=100):
    """
    Make the matrix representation of the raising operator :math:`\\hat{a}^\\dagger`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrix

    Returns
    -------
    out : complex ndarray
        matrix of size n by n that represents the raising operator :math:`\\hat{a}^\\dagger`

    Raises
    ------
    ValueError
        If n is less than or equal to 1

    See Also
    --------
    SHO : overall description of harmonic oscillator functions
    a : lowering operator

    Notes
    -----
    The matrix :math:`a^\\dagger` in the eigenbasis of the Hamiltonian reads
    
    .. math::

         a^\\dagger = \\begin{bmatrix}
             0 & 0 & 0 \\\\
             \\sqrt{1} & 0 & 0 \\\\
             0 &   \\sqrt{2} & 0 \\\\
             \\end{bmatrix}

    """
    if n <= 1:
        raise ValueError("n should be 2 or greater")
    matrix = np.zeros((n, n), dtype=np.complex128)
    for i in range(n - 1):
        matrix[i + 1, i] = np.sqrt(i + 1)
    return matrix


def p(n=100, w=1, hbar=1, m=1):
    """
    Make the matrix representation of the momentum operator :math:`\\hat{p}`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrix
    w : float, optional
        frequency :math:`\\omega` of the harmonic oscilator
    hbar : float, optional
        numerical value of physical constant :math:`\\hbar`
    m : float, optional
        mass :math:`m` of harmonic oscillator

    Returns
    -------
    out : complex ndarray
        matrix of size n by n that represents the momentum operator :math:`\\hat{p}`

    Raises
    ------
    ValueError
        If n is less than or equal to 1, or w, hbar, m <= 0

    See Also
    --------
    SHO : overall description of harmonic oscillator functions
    x : position operator
    

    Notes
    -----
    The matrix :math:`\\hat{p}` in the eigenbasis of the Hamiltonian reads
    
    .. math::

         \\hat{p}  =i\\sqrt{\\frac{m\\hbar\\omega}{2}} \\begin{bmatrix}
             0 & -\\sqrt{1} & 0 \\\\
             \\sqrt{1} & 0 & -\\sqrt{2} \\\\
             0 &   \\sqrt{2} & 0 \\\\
             \\end{bmatrix}

    """

    # if too time-consuming, may look to improve
    if n <= 1:
        raise ValueError("n should be 2 or greater")
    if any([w <= 0, hbar <= 0, m <= 0]):
        raise ValueError(
            " dimensional constants w,hbar, m should not be zero or negative"
        )
    matrix = -1j * (a(n=n) - a_dagger(n=n)) * np.sqrt(m * hbar * w / 2)
    return matrix


def x(n=100, w=1, hbar=1, m=1):
    """
    Make the matrix representation of the position operator :math:`\\hat{x}`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrix
    w : float, optional
        frequency :math:`\\omega` of the harmonic oscilator
    hbar : float, optional
        numerical value of physical constant :math:`\\hbar`
    m : float, optional
        mass :math:`m` of harmonic oscillator

    Returns
    -------
    out : complex ndarray
        matrix of size n by n that represents the position operator :math:`\\hat{x}`

    Raises
    ------
    ValueError
        If n is less than or equal to 1, or w, hbar, m <= 0

    See Also
    --------
    SHO : overall description of harmonic oscillator functions
    p : momentum operator
    

    Notes
    -----
    The matrix :math:`\\hat{x}` in the eigenbasis of the Hamiltonian reads
    
    .. math::

         \\hat{x}  =\\sqrt{\\frac{\\hbar}{2m\\omega}} \\begin{bmatrix}
             0 & \\sqrt{1} & 0 \\\\
             \\sqrt{1} & 0 & \\sqrt{2} \\\\
             0 &   \\sqrt{2} & 0 \\\\
             \\end{bmatrix}

    """
    
    if n <= 1:
        raise ValueError("n should be 2 or greater")
    if any([w <= 0, hbar <= 0, m <= 0]):
        raise ValueError(
            " dimensional constants w,hbar, m should not be zero or negative"
        )
    matrix = np.sqrt(hbar / m / w / 2) * (a(n=n) + a_dagger(n=n))
    return matrix


def H(n=100, w=1, hbar=1, m=1):
    """
    Make the matrix representation of the Hamiltonian operator :math:`\\hat{H}`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrix
    w : float, optional
        frequency :math:`\\omega` of the harmonic oscilator
    hbar : float, optional
        numerical value of physical constant :math:`\\hbar`
    m : float, optional
        mass :math:`m` of harmonic oscillator

    Returns
    -------
    out : complex ndarray
        matrix of size n by n that represents the Hamiltonian operator :math:`\\hat{H}`

    Raises
    ------
    ValueError
        If n is less than or equal to 1, or w, hbar, m <= 0

    See Also
    --------
    SHO : overall description of harmonic oscillator functions
    p : momentum operator
    x : position operator
    

    Notes
    -----
    The matrix :math:`\\hat{H}` in the eigenbasis of the Hamiltonian reads
    
    .. math::

         \\hat{H}  = \\hbar \\omega \\begin{bmatrix}
             \\frac{1}{2} & 0 & 0 \\\\
              0 & \\frac{3}{2} & 0 \\\\
             0 &   0 & \\frac{5}{2} \\\\
             \\end{bmatrix}

    """
    p_matrix = p(
        n=n + 1,
        w=w,
        hbar=hbar,
        m=m,
    )
    x_matrix = x(
        n=n + 1,
        w=w,
        hbar=hbar,
        m=m,
    )
    return (
        np.linalg.matrix_power(p_matrix, 2) / (2 * m)
        + 1 / 2 * m * w**2 * np.linalg.matrix_power(x_matrix, 2)
    )[:-1, :-1]


def SHO_distribution(T, n=50, hbar=1, w=1, k=1):
    if T < 0:
        raise ValueError(" T should not be negative")
    if any([n <= 0, hbar <= 0, w <= 0, k <= 0]):
        raise ValueError(" size of matrix and physical constants should be positive")
    if T == 0:
        probabilities = np.zeros(n, dtype=np.float64)
        probabilities[0] = 1
        return probabilities

    probabilities = np.zeros(n, dtype=np.float64)
    energies = np.arange(n, dtype=np.float64)
    energies = (energies + 0.5) * (hbar * w)
    probabilities[:] = np.exp(-energies / (k * T))
    return probabilities / sum(probabilities)


def get_second_moments(n=100, w=1, hbar=1, m=1):
    """
    Get all second moments of :math:`\\hat{x}` and :math:`\\hat{p}` :math:`(\\hat{x}^2, \\hat{x}\\hat{p}, \\hat{p}\\hat{x}, \\hat{p}^2)`

    Parameters
    ----------
    n : int, optional
        length and width of the resultant square matrices
    w : float, optional
        frequnecy :math:`\\omega` of the harmonic oscillator
    hbar : float, optional
        value of the physical constant :math: `\\hbar`
    m : float, optional
        mass :math: `m` of harmonic oscillator
    
    Returns
    -------
    out : list of four complex ndarrays
        out[0] corresponds to :math:`\\hat{x}^2`, out[1] corresponds to :math:`\\hat{x}\\hat{p}`,
        out[2] corresponds to :math:`\\hat{p}\\hat{x}`, and out[3] corresponds to :math:`\\hat{p}^2`

    Raises
    ------
    ValueError: if `n` <= 1 or any of `w`,`hbar`,`m` <= 0

    Notes
    -----
    The matrices for the second-order operators in the basis of the eigenvectors of the Hamiltonian are given by:

    .. math::

        \\hat{x}^2 =  \\frac{\\hbar}{2m\\omega} \\begin{bmatrix} 
                       1 & 0 & \\sqrt{2} \\\\
                       0 & 3 & 0 \\\\
                       \\sqrt{2} & 0 & 5 
                       \\end{bmatrix}
        \\hat {x}\\hat{p} = \\frac{i\\hbar}{2} \\begin{bmatrix}
                            1 & 0 & -\\sqrt{2} \\\\
                            0 & 1 & 0 \\\\
                            \\sqrt{2} & 0 & 1
                            \\end{bmatrix}

    .. math::

        \\hat{p}\\hat{x} = -\\frac{i\\hbar}{2} \\begin{bmatrix}
                           1 & 0 & \\sqrt{2} \\\\
                           0 & 1 & 0 \\\\
                           -\\sqrt{2} & 0 & 1 
                           \\end{bmatrix}
        \\hat{p}^2 = \\frac{m\\hbar\\omega}{2} \\begin{bmatrix}
                           1 & 0 & -\\sqrt{2} \\\\
                           0 & 3 & 0 \\\\
                           -\\sqrt{2} & 0 & 5 
                           \\end{bmatrix}

    """
    p_matrix = p(n=n + 1, w=w, hbar=hbar, m=m)
    x_matrix = x(n=n + 1, w=w, hbar=hbar, m=m)
    return (
        (x_matrix @ x_matrix)[:-1, :-1],
        (x_matrix @ p_matrix)[:-1, :-1],
        (p_matrix @ x_matrix)[:-1, :-1],
        (p_matrix @ p_matrix)[:-1, :-1],
    )


# def H(n=100,hbar=1,w=1,m=1):
#     matrix =sp.zeros(n,n)
#     for i in range(n):
#         matrix[i,i] = hbar*w*(i+1/2)
#     return matrix

# def a(n=100):
#     matrix = sp.zeros(n,n)
#     for i in range(n-1):
#         matrix[i,i+1] = sp.sqrt(i+1)
#     return matrix

# def a_dagger(n=100):
#     matrix = sp.zeros(n,n)
#     for i in range(n-1):
#         matrix[i+1,i] = sp.sqrt(i+1)
#     return matrix

# def p(n=100,w=1,hbar=1,m=1):
#     matrix = sp.I*(a_dagger(n)-a(n))
#     matrix *= 1/sp.sqrt(2)*sp.sqrt(m*hbar*w)
#     return matrix

# def x(n=100,w=1,hbar=1,m=1):
#     matrix = a_dagger(n) + a(n)
#     matrix *= 1/sp.sqrt(2)*sp.sqrt(hbar/(m*w))
#     return matrix
