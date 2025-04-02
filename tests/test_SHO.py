import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import SHO


def test_a():
    a = np.zeros((3, 3), dtype=np.complex128)
    a[0][1] = np.sqrt(1)
    a[1][2] = np.sqrt(2)
    assert (SHO.a(3) == a).all()
    try:
        SHO.a(1)
    except ValueError:
        assert True
    else:
        assert False


def test_a_dagger():
    a_dagger = np.zeros((3, 3), dtype=np.complex128)
    a_dagger[1][0] = np.sqrt(1)
    a_dagger[2][1] = np.sqrt(2)
    assert (SHO.a_dagger(3) == a_dagger).all()
    try:
        SHO.a_dagger(1)
    except ValueError:
        assert True
    else:
        assert False


def make_x(hbar=1, m=1, w=1):
    x = np.zeros((3, 3), dtype=np.complex128)
    x[0][1] = np.sqrt(1)
    x[1][2] = np.sqrt(2)
    x[1][0] = np.sqrt(1)
    x[2][1] = np.sqrt(2)
    x *= np.sqrt(hbar / m / w / 2)
    return x


def test_x():
    assert (SHO.x(3) == make_x()).all()
    assert (SHO.x(3, hbar=1, m=2, w=0.5) == make_x(hbar=1, m=2, w=0.5)).all()
    try:
        SHO.x(3, hbar=0)
    except ValueError:
        assert True
    else:
        assert False
    try:
        SHO.x(3, hbar=1, m=-1)
    except ValueError:
        assert True
    else:
        assert False


def make_p(hbar=1, m=1, w=1):
    p = np.zeros((3, 3), dtype=np.complex128)
    p[0][1] = -np.sqrt(1)
    p[1][2] = -np.sqrt(2)
    p[1][0] = np.sqrt(1)
    p[2][1] = np.sqrt(2)
    p *= 1j * np.sqrt(hbar * m * w / 2)
    return p


def test_p():
    assert (SHO.p(3) == make_p()).all()
    assert (SHO.p(3, hbar=1, m=2, w=2) == make_p(hbar=1, m=2, w=2)).all()
    try:
        SHO.p(3, hbar=0)
    except ValueError:
        assert True
    else:
        assert False
    try:
        SHO.p(3, hbar=1, m=-1)
    except ValueError:
        assert True
    else:
        assert False

def get_H(hbar=1, w=1):
    H = np.zeros((3, 3), dtype=np.complex128)
    H[0][0] = .5
    H[1][1] = 1.5
    H[2][2] = 2.5
    H *= hbar*w
    return H
    
def test_H():
    assert np.allclose(SHO.H(3), get_H(), atol=1e-15)
    assert np.allclose(SHO.H(3, hbar=2, w=1),get_H(hbar=2, w=1), atol=1e-15)
    try:
        SHO.H(0)
    except ValueError:
        assert True
    else:
        assert False
    try:
        SHO.H(3, hbar = 0)
    except ValueError:
        assert True
    else:
        assert False

def get_second_moments(hbar=1, m=1, w=1):
    p_squared = np.zeros((3, 3), dtype=np.complex128)
    p_squared[0][0] = -1
    p_squared[0][2] = np.sqrt(2)
    p_squared[1][1] = -3
    p_squared[2][0] = np.sqrt(2)
    p_squared[2][2] = -5
    p_squared *= -hbar * m * w / 2
    
    x_squared = np.zeros((3, 3), dtype=np.complex128)
    x_squared[0][0] = 1
    x_squared[0][2] = np.sqrt(2)
    x_squared[1][1] = 3
    x_squared[2][0] = np.sqrt(2)
    x_squared[2][2] = 5
    x_squared *= hbar / m / w / 2
    
    xp = np.zeros((3, 3), dtype=np.complex128)
    xp[0][0] = 1
    xp[0][1] = 0
    xp[0][2] = -np.sqrt(2)
    xp[1][1] = 1
    xp[1][2] = 0
    xp[2][0] = np.sqrt(2)
    xp[2][2] = 1
    xp *= 1j * hbar / 2
    
    px = xp.conj().T
    
    return x_squared, xp, px, p_squared
 

def test_get_second_moments():
    second_moments = get_second_moments()
    second_moments_SHO = SHO.get_second_moments(3)
    assert np.allclose(second_moments_SHO[0], second_moments[0], atol=1e-15) #x_squared
    assert np.allclose(second_moments_SHO[1], second_moments[1], atol=1e-15) #xp
    assert np.allclose(second_moments_SHO[2], second_moments[2], atol=1e-15) #px
    assert np.allclose(second_moments_SHO[3], second_moments[3], atol=1e-15) #p_squared


    
        
if __name__ == "__main__":
    pass

    
