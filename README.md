# symbolic_ODE_approximation
15424 Cyber Physical Systems Final Project: Symbolic ODE Approximation

Language: python

Libraries: numpy, sympy, matplotlib (for plotting)

For examples on how to run see
- python_workflow.ipynb: uses ODE approximation assuming function representing ODE (x' = f(t,x)) is already python function
- string_workflow.ipynb: uses sympy parsing to parse a string representation of the ODE into a python function and then applies ODE approximation
- errors.ipynb: uses numerical method module to show how to compute timestep h given parameters of the function f for (x' = f(t,x)) and an error tolerance

Modules
- approximation.py: handles symbolic function approximation including computing timestep h, generating discrete approximation from numerical method and continuous approximation using linear interpolation
- numerical_method.py: numerical methods
- ODE_parser.py: sympy parser to parse ODEs