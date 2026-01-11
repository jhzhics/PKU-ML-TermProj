This is a research project on [Kissing Number Problem](https://en.wikipedia.org/wiki/Kissing_number).

# Materials
1. [Requirement File](docs/ML%20Term%20Project.pdf)
2. [Discussion Document Link](https://docs.google.com/document/d/1qDuhGE2B0Ftnie3CbB64CrlsF_eG_UEqJj3Fro7AHQw/edit?usp=sharing)

# Usage
1. use `conda` to setup and activate `kn_env` environment defined in [environment.yml](environment.yml).
```bash
# Create Environment
conda env create -f environment.yml

# Activate Environment
conda activate kn_env
```

2. use `python` run the solvers listed below directly. e.g. `python src/AALM.py`

# Solvers
### [src/repulsion_solver.py](src/repulsion_solver.py)
First initialize points randomly and then optimize to make the distance bigger.

Achieve: 5 Dim 31 Points 

### [src/slack_solver.py](src/slack_solver.py)
First initialize n points as a simplex in (n - 1) dimension and then optimize a projection to d dimension.

Achieve: 5 Dim 36 Points


### [src/AALM.py](src/AALM.py)
Augmented Lagrangian Method (ALM) with a hybrid Adam/L-BFGS optimizer. Initialize as Icosahedron.

Achieve: 5 Dim 36 Points

### [src/evolution_solver.py](src/evolution_solver.py)
Evolution algorithm on initial set. Optimize using Augmented Lagrangian Method. Initialize using a mix of symmetric and random sets.

Achieve: 5 Dim 38 Points