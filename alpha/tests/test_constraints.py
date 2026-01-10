import numpy as np
from src.kissing.initializers import init_icosahedron_d3, init_24cell_d4
from src.kissing.constraints import check_constraints

def test_icosahedron_feasible_scaled():
    x = init_icosahedron_d3()
    ok, info = check_constraints(x, min_dist=1.0, max_norm=1.0, tol=1e-6)
    assert ok, info

def test_24cell_feasible_scaled():
    x = init_24cell_d4()
    ok, info = check_constraints(x, min_dist=1.0, max_norm=1.0, tol=1e-6)
    assert ok, info
