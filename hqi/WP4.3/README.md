# General description

The libraries here are designed to perform simulations of dynamics of spin systems on quantum computers or their emulators.
The following code was written based on the myQLM and Qaptiva libraries and to run on the TGCC machines, some dependencies
may change for other usages.

# Included files

## Libraries

`library_annealing.py`: build schedules for a quantum annealing protocol for the ferromagnetic Ising chain in transverse field

`library_measurments.py`: based on a schedule (analog quantum computation) or circuit (gate-based quantum computation) perform a simulation that measures one- or two-point correlation functions throughout the system.

## Tutorials

`Tutorial.ipynb`: demonstrates the use of the above libraries

## Tests

`test_lib.py`: script to test that the library operates as intended. Just run pytest in this folder.
