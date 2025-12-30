import pytest
from typing import Annotated
import numpy as np
import sympy
from decimal import Decimal, localcontext, ROUND_UP, ROUND_DOWN


def is_accepted_solution(solution: Annotated[np.ndarray, "shape=(n, d)"], prec: int = 15) -> bool:
    '''
    Function to determine if a solution is accepted.
    The solution will be accepted if the minimum distance between any two distinct points in the solution is equal to
    or greater than the maximum distance from the origin to any point in the solution.
    i.e. |x_i - x_j| >= max(|x_k|) for all i != j and for all k

    Parameters
    ----------
    solution : np.ndarray
        An array of shape (n, d) representing the solution to be validated.
        The first dimension (n) represents the number of points, and the second dimension (d) represents the dimensionality of each point.
        
    prec : int, optional
        The precision for floating-point comparisons. Default is 15. Not used if the input is of sympy expressions.
        
    Returns
    -------
    bool
        True if the solution is accepted, False otherwise.
        
    Warnings
    --------
    False negative cases may occur due to floating-point precision issues.
    '''
    
    if not isinstance(solution, np.ndarray):
        raise TypeError("Input solution must be a numpy ndarray.")
    if solution.ndim != 2:
        raise ValueError("Input solution must be a 2D array.")
    n, d = solution.shape
    
    if any(isinstance(x, sympy.Expr) for row in solution for x in row):
        max_dist_sqaured = 0
        for point in solution:
            r_sq = sum(coord**2 for coord in point)
            if sympy.simplify(r_sq - max_dist_sqaured) > 0:
                max_dist_sqaured = r_sq
        
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = sum((solution[i, k] - solution[j, k])**2 for k in range(d))
                if sympy.simplify(dist_sq - max_dist_sqaured) < 0:
                    return False
    else:
        points = [[Decimal(coord) for coord in point] for point in solution]    
        with localcontext() as ctx:
            ctx.prec = prec
            # Compute maximum distance from origin
            ctx.rounding = ROUND_UP
            max_dist_from_origin_squared = max(sum(coord**2 for coord in point) for point in points)
            
            # Compute minimum distance between distinct points
            ctx.rounding = ROUND_DOWN
            for i in range(solution.shape[0]):
                for j in range(i + 1, solution.shape[0]):
                    dist_squared = sum((points[i][k] - points[j][k])**2 for k in range(solution.shape[1]))
                    if dist_squared < max_dist_from_origin_squared:
                        return False
    return True

# Unit tests

@pytest.mark.parametrize("solution, expected", [
    (np.array([[1.0, 0.0], [0.0, 1.0]]), True),
    (np.array([[1.0, 0.0], [0.5, 0.5]]), False),
    (np.array([
        [1, 0],
        [sympy.Rational(1, 2), sympy.sqrt(sympy.Rational(3, 4))],
        [sympy.Rational(-1, 2), sympy.sqrt(sympy.Rational(3, 4))],
        [-1, 0],
        [sympy.Rational(-1, 2), -sympy.sqrt(sympy.Rational(3, 4))],
        [sympy.Rational(1, 2), -sympy.sqrt(sympy.Rational(3, 4))]
    ], dtype=object), True),
        (np.array([
        [1, 0],
        [sympy.Rational(1, 2), sympy.S("sqrt(3)/2 - 1/1000")],
        [sympy.Rational(-1, 2), sympy.sqrt(sympy.Rational(3, 4))],
        [-1, 0],
        [sympy.Rational(-1, 2), -sympy.sqrt(sympy.Rational(3, 4))],
        [sympy.Rational(1, 2), -sympy.sqrt(sympy.Rational(3, 4))]
    ], dtype=object), False),
])
def test_is_accepted_solution(solution, expected):
    assert is_accepted_solution(solution, prec=35) == expected