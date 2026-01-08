This is a research project on [Kissing Number Problem](https://en.wikipedia.org/wiki/Kissing_number).

# Materials
1. [Requirement File](docs/ML%20Term%20Project.pdf)
2. [Discussion Document Link](https://docs.google.com/document/d/1qDuhGE2B0Ftnie3CbB64CrlsF_eG_UEqJj3Fro7AHQw/edit?usp=sharing)

# Solvers
### [src/repulsion_solver.py](src/repulsion_solver.py)
First initialize points randomly and then optimize to make the distance bigger.

Achieve: 5 Dim 31 Points 

### [src/slack_solver.py](src/slack_solver.py)
First initialize n points as a simplex in (n - 1) dimension and then optimize a projection to d dimension.

Achieve: 5 Dim 36 Points