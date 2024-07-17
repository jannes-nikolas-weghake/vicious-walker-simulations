# Vicious Walker Simulations

This repository contains Python implementations of vicious walker models, including annihilating random walks in multiple dimensions and a zombie apocalypse variant. The simulations feature multi-threading and vectorization for improved performance.

## Features

- Vicious walker simulations in 1D, 2D, and 3D
- Diagonal movement option
- Probabilistic annihilation
- Zombie apocalypse variant
- Comparison to SIR (Susceptible, Infected, Recovered) models
- Multi-threaded processing for faster simulations
- Vectorized computations using NumPy

## Files

- `main.py`: Main script to run simulations
- `helperfunctions.py`: Helper functions for simulations and data analysis
- `zombies.py`: Implementation of the zombie apocalypse variant

## Requirements

Python 3.x
NumPy
Matplotlib
mpi4py
tqdm

## Results

The simulations produce various plots and data analyses, including:

Density decay of vicious walkers in different dimensions
Comparison of walker behavior with and without diagonal moves
Analysis of probabilistic annihilation
Zombie apocalypse progression compared to SIR models
