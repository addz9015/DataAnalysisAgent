# layer2/utils/matrix_ops.py
"""
Matrix operations for stochastic processes
"""

import numpy as np
from typing import Tuple, Optional

def check_ergodicity(P: np.ndarray) -> Tuple[bool, str]:
    """
    Check if Markov chain is ergodic (irreducible and aperiodic)
    
    Returns:
        (is_ergodic, message)
    """
    n = P.shape[0]
    
    # Check irreducibility: P^n should have all positive entries for some n
    P_power = np.linalg.matrix_power(P, n)
    is_irreducible = np.all(P_power > 0)
    
    if not is_irreducible:
        return False, "Chain is not irreducible (multiple communicating classes)"
    
    # Check aperiodicity: self-loop or gcd of return times is 1
    # Simplified: check if any diagonal entry > 0
    has_self_loop = np.any(np.diag(P) > 0)
    
    if not has_self_loop:
        # More complex check needed, but assume periodic if no self-loops
        return False, "Chain may be periodic (no self-loops detected)"
    
    return True, "Chain is ergodic"

def compute_stationary_distribution(P: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute stationary distribution as left eigenvector
    
    Solves: pi * P = pi, sum(pi) = 1
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    
    # Find eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    
    # Normalize
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / np.sum(stationary)
    
    return stationary